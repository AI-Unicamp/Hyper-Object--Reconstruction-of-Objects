from __future__ import annotations
from pathlib import Path
from typing import Any, Literal, Optional

from torch.utils.data import Dataset
import zarr
import numpy as np
import torch
import os

from .io import read_rgb_image, read_h5_cube, read_mosaic

# deterministic transforms - allows transforms to remain consistent in input/output even
# in separate dataloaders
class DeterministicTransforms:
    def __init__(self, track: Literal[1, 2], is_out: bool, base_seed: int = 0):
        self.base_seed = int(base_seed)
        self.track = track
        self.is_out = is_out
        self.transforms: dict[str, dict[str, Any]] = {}

    def add_transform(self, transform_name: str, **kwargs):
        self.transforms[transform_name] = kwargs

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _rng_for(self, idx: int):
        # combine base_seed, epoch, and idx into a 32-bit seed
        # using a simple mixing (avoid collisions)
        h = (self.base_seed * 1000003) ^ (self.epoch * 9176) ^ idx
        seed = h & 0xFFFFFFFF
        return np.random.default_rng(seed)  # or random.Random(seed)

    def _random_crop_det(self, img: torch.Tensor, rng, ps=256):
        H, W = img.shape[-2], img.shape[-1]
        if self.track==2 and self.is_out:
            # make sure the RNG call is exactly the same as for the input
            r = int(rng.integers(0, H//2 - ps))
            c = int(rng.integers(0, W//2 - ps))
        else:
            r = int(rng.integers(0, H - ps))
            c = int(rng.integers(0, W - ps))

        if self.track == 1:
            # ensure coordinates are even; i.e. crop aligns with bayer pattern
            r -= r % 2
            c -= c % 2
        if self.track == 2 and self.is_out:
            # output is twice the size of the input
            img = img[:, 2*r:2*r+2*ps, 2*c:2*c+2*ps]
        else:
            img = img[:, r:r+ps, c:c+ps]

        return img

    def _random_flip_det(self, img: torch.Tensor, rng):
        if rng.random() < 0.5:
            img = img.flip(dims=[2])
        return img

    def apply(self, img, idx: int):
        rng = self._rng_for(idx)

        if "random_crop" in self.transforms:
            img = self._random_crop_det(img, rng, **self.transforms["random_crop"])
        if "random_flip" in self.transforms:
            img = self._random_flip_det(img, rng, **self.transforms["random_flip"])

        return img

def read_rgb_np(path):
    return np.load(path).astype(np.float32)

class PartialDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        track: Literal[1, 2],
        dataset_type: Literal["train", "test-public"],
        img_type: Literal["hsi_61", "hsi_61_zarr", "mosaic", "rgb_2", "rgb"],
        transforms: Optional[DeterministicTransforms] = None,
    ) -> None:
        super().__init__()
        self.track = track 
        self.img_type = img_type

        path = f"{data_root}/track{track}/{dataset_type}/{img_type}"
        self.paths = sorted([Path(f"{path}/{f}") for f in os.listdir(path)])

        self.zarr_cache: list[zarr.Array | zarr.Group | None] = [None] * len(self.paths)

        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.paths)

    def _read_zarr(self, idx: int):
        if self.zarr_cache[idx] is None:
            self.zarr_cache[idx] = zarr.open(str(self.paths[idx]), mode='r')
        return self.zarr_cache[idx]["data"][:]

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        out = torch.tensor([])
        match self.img_type:
            case "hsi_61":
                cube = read_h5_cube(path, 'cube')                          # (H,W,C)
                out = torch.from_numpy(np.transpose(cube, (2, 0, 1)))    # C,H,W
            case "hsi_61_zarr":
                cube = self._read_zarr(idx)                          # (H,W,C)
                out = torch.from_numpy(cube)    # C,H,W
            case "mosaic":
                mosaic = read_mosaic(path)                                  # (H,W,1) float32 [0,1]
                out = torch.from_numpy(np.transpose(mosaic, (2, 0, 1)))    # 1,H,W
            case "rgb_2":
                rgb_2 = read_rgb_image(path)                             # (H,W,3) float32 [0,1]
                out = torch.from_numpy(np.transpose(rgb_2, (2, 0, 1)))  # C,H,W
            case "rgb_full":
                rgb = read_rgb_np(path)                             # (H,W,3) float32 [0,1]
                out = torch.from_numpy(np.transpose(rgb, (2, 0, 1)))  # C,H,W

        if self.transforms is not None:
            out = self.transforms.apply(out, idx)

        img_id = os.path.splitext(os.path.split(path)[1])[0]
        return out, img_id
