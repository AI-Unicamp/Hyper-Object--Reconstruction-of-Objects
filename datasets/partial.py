from __future__ import annotations
from pathlib import Path
from typing import Literal, Optional

from torch.utils.data import Dataset
import zarr
import numpy as np
import torch
import os

from .io import read_rgb_image, read_h5_cube, read_mosaic
from .det_transform import DeterministicTransforms

# R G
# G B
_bayer = torch.tensor([
    [[1, 0],
     [0, 0]],
    [[0, 1],
     [1, 0]],
    [[0, 0],
     [0, 1]]
]).float()


def read_rgb_np(path):
    return np.load(path).astype(np.float32)

class PartialDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        track: Literal[1, 2],
        dataset_type: Literal["train", "test-public", "test-private"],
        img_type: Literal["hsi_61", "hsi_61_zarr", "mosaic", "rgb_2", "rgb_full"],
        transforms: Optional[DeterministicTransforms] = None,
        old_mode: bool = False # whether to use old input mode for mosaic
    ) -> None:
        super().__init__()
        self.track = track 
        self.img_type = img_type

        path = f"{data_root}/track{track}/{dataset_type}/{img_type}"
        self.paths = sorted([Path(f"{path}/{f}") for f in os.listdir(path)])

        self.zarr_cache: list[zarr.Array | zarr.Group | None] = [None] * len(self.paths)

        self.transforms = transforms
        
        self.old_mode = old_mode

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

                if not self.old_mode:
                    _, H, W = out.shape
                    ry = H // 2 + 1
                    rx = W // 2 + 1
                    repeated_bayer = _bayer.repeat(1, ry, rx)[:, :H, :W]

                    # concat bayer with input to ensure alignment in transforms
                    out = torch.cat([out, repeated_bayer]) # (4, H, W)
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
