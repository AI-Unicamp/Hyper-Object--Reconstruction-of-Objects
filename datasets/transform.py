import random
import torch

def random_flip(batch):
    """
    Random horizontal and vertical flips applied consistently across all modalities.
    Applies to: rgb, cube, rgb_2, rgb_4, mosaic
    """
    keys = ["input_data", "output_data"]

    # horizontal flip
    if random.random() < 0.5:
        for k in keys:
            if k in batch and isinstance(batch[k], torch.Tensor):
                batch[k] = torch.flip(batch[k], dims=[2])  # flip width

    # vertical flip
    if random.random() < 0.5:
        for k in keys:
            if k in batch and isinstance(batch[k], torch.Tensor):
                batch[k] = torch.flip(batch[k], dims=[1])  # flip height

    return batch

def random_crop(batch, ps=256, track=1):
    """
    Random crop applied consistently across all modalities.
    
    Must set `track=2` for track 2.
    """
    keys = ["input_data", "output_data"]

    # assume all tensors have shape (C,H,W)
    _, H, W = batch["input_data"].shape
    if H >= ps and W >= ps:
        r = random.randint(0, H - ps)
        c = random.randint(0, W - ps)

        if track == 1:
            # ensure coordinates are even; i.e. crop aligns with bayer pattern
            r -= r % 2
            c -= c % 2
            for k in keys:
                if k in batch and isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k][:, r:r+ps, c:c+ps]
        elif track == 2:
            if "input_data" in batch and isinstance(batch["input_data"], torch.Tensor):
                batch["input_data"] = batch["input_data"][:, r:r+ps, c:c+ps]

            if "output_data" in batch and isinstance(batch["output_data"], torch.Tensor):
                batch["output_data"] = batch["output_data"][:, 2*r:2*r+2*ps, 2*c:2*c+2*ps]

    return batch
