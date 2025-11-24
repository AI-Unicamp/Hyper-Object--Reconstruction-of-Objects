import torch
import numpy as np


from typing import Literal, Any

class DeterministicTransforms:
    """FOR ICASSP CHALLENGE:
        Deterministic transforms - allows transforms to remain consistent in input/output even
        in separate data-loaders. Every transform is given an `rng` seeded on: 1. the current epoch,
        2. the datum index. This allows transforms to be applied consistently to inputs and outputs.
    """
    def __init__(self, track: Literal[1, 2], is_out: bool, base_seed: int = 0, old_mode: bool=False):
        self.base_seed = int(base_seed)
        self.track = track
        self.is_out = is_out
        self.old_mode = old_mode
        self.transforms: dict[str, dict[str, Any]] = {}

        self.epoch = 0

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
            r = int(rng.integers(0, H//2 - ps, endpoint=True))
            c = int(rng.integers(0, W//2 - ps, endpoint=True))
        else:
            r = int(rng.integers(0, H - ps, endpoint=True))
            c = int(rng.integers(0, W - ps, endpoint=True))

        if self.track == 1 and self.old_mode:
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
        if rng.random() < 0.5: # horizontal flip
            img = img.flip(dims=[2])
        if rng.random() < 0.5: # vertical flip
            img = img.flip(dims=[1])
        return img

    def _random_rot90_det(self, img: torch.Tensor, rng):
        k = rng.integers(0, 4)
        if k > 0:
            img = img.rot90(k, dims=(1, 2))
        return img

    def _add_input_gaussian_noise_det(self, img: torch.Tensor, _, sigma=0.02):
        # only applied to input - don't need to care about matching RNG
        if self.is_out:
            return img

        if self.track==1 and not self.old_mode:
            # img is 1-channel input + 3-channel mosaic pattern
            noise = torch.randn_like(img[0, :, :]) * sigma
            img[0, :, :] += noise
        else:
            # can apply noise directly to input
            noise = torch.randn_like(img) * sigma
            img += noise
        return img.clamp(0, 1)

    def _spectral_jitter_det(self, img: torch.Tensor, _, sigma=0.02):
        # only applied to output - don't need to care about matching RNG
        if not self.is_out:
            return img

        C = img.size(0)

        jitter = 1.0 + sigma * torch.randn(C, 1, 1, dtype=img.dtype, device=img.device)
        img *= jitter
        return img

    def apply(self, img, idx: int):
        if "random_crop" in self.transforms:
            img = self._random_crop_det(img, self._rng_for(idx), **self.transforms["random_crop"])
        if "random_flip" in self.transforms:
            img = self._random_flip_det(img, self._rng_for(idx), **self.transforms["random_flip"])
        if "random_rot90" in self.transforms:
            img = self._random_rot90_det(img, self._rng_for(idx), **self.transforms["random_rot90"])
        if "rgb_gaussian_noise" in self.transforms:
            img = self._add_input_gaussian_noise_det(img, self._rng_for(idx), **self.transforms["rgb_gaussian_noise"])
        if "spectral_jitter" in self.transforms:
            img = self._spectral_jitter_det(img, self._rng_for(idx), **self.transforms["spectral_jitter"])

        return img
