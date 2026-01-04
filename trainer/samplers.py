import torch
import numpy as np
from torch.utils.data import Sampler

class SimpleShuffleSampler(Sampler[int]):
    """Shuffles indices deterministically."""
    def __init__(self, data_len: int, base_seed: int = 0):
        self.base_seed = int(base_seed)
        self.epoch = 0
        self._indices = torch.arange(data_len)

    def set_epoch(self, epoch: int):
        """Reshuffle indices deterministically for this epoch."""
        self.epoch = int(epoch)
        g = torch.Generator()
        g.manual_seed(self.base_seed + self.epoch)

        perm = torch.randperm(len(self._indices), generator=g)
        self._indices = self._indices[perm]

    def __iter__(self):
        # DataLoader in the main process will call this each epoch
        return iter(self._indices.tolist())

    def __len__(self):
        return len(self._indices)

class LossBalancedSampler(Sampler[int]):
    def __init__(self, sample_names: list[str], base_seed: int = 0):
        self.base_seed = int(base_seed)
        self.epoch = 0
        self._indices = [i for i in range(len(sample_names))]

        self._samples = sample_names
        self._samples_by_cat = {
                "cat1": np.array([sample for sample in sample_names if "Category-1" in sample]),
                "cat2": np.array([sample for sample in sample_names if "Category-2" in sample]),
                "cat3": np.array([sample for sample in sample_names if "Category-3" in sample]),
                "cat4": np.array([sample for sample in sample_names if "Category-4" in sample]),
                }
        
        # start with balanced category weights
        self._cat_weights = np.array([0.25, 0.25, 0.25, 0.25])

    def last_cat_losses(self, cat_losses: list[float]):
        self._cat_weights = np.array(cat_losses) / np.array(cat_losses).sum()

    def set_epoch(self, epoch: int):
        """Draw len(samples) indices balanced by category loss, deterministically per epoch."""
        self.epoch = int(epoch)

        h = (self.base_seed * 1000003) ^ (self.epoch * 9176)
        seed = h & 0xFFFFFFFF
        g = np.random.default_rng(seed)  # or random.Random(seed)

        indices = []
        for _ in range(len(self._samples)):
            cat = g.choice(["cat1", "cat2", "cat3", "cat4"], p=self._cat_weights)
            sample = g.choice(self._samples_by_cat[cat])
            idx = self._samples.index(sample)
            indices.append(idx)

        self._indices = indices

    def __iter__(self):
        # DataLoader in the main process will call this each epoch
        return iter(self._indices)

    def __len__(self):
        return len(self._samples)
