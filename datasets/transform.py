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
            for k in keys:
                if k in batch and isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k][:, r:r+ps, c:c+ps]
        elif track == 2:
            if "input_data" in batch and isinstance(batch["input_data"], torch.Tensor):
                batch["input_data"] = batch["input_data"][:, r:r+ps, c:c+ps]

            if "output_data" in batch and isinstance(batch["output_data"], torch.Tensor):
                batch["output_data"] = batch["output_data"][:, 2*r:2*r+2*ps, 2*c:2*c+2*ps]

    return batch





###### DATA AUGMENTATION ########
# ============================================================
#  Helper: work with HyperObjectDataset transform batch
#  batch = {"input_data": tensor(C,H,W), "output_data": tensor(C_hsi,H,W), "id": str}
# ============================================================

def random_rot90(batch):
    """
    Random rotation by 0/90/180/270 degrees.
    Applied jointly to input_data and output_data.
    """
    inp = batch["input_data"]          # [Cin, H, W]
    out = batch["output_data"]         # [Cout, H, W] (may be empty tensor for submission)

    k = random.randint(0, 3)  # 0,1,2,3 * 90°
    if k > 0:
        inp = torch.rot90(inp, k, dims=[1, 2])
        if out.numel() > 0:
            out = torch.rot90(out, k, dims=[1, 2])

    batch["input_data"] = inp
    batch["output_data"] = out
    return batch


def add_input_gaussian_noise(batch, sigma=0.01):
    """
    Add small Gaussian noise to the input (RGB or mosaic).
    Keeps values in [0,1].
    """
    inp = batch["input_data"]          # [Cin, H, W]

    noise = torch.randn_like(inp) * sigma
    inp = torch.clamp(inp + noise, 0.0, 1.0)

    batch["input_data"] = inp
    return batch


def spectral_jitter_output(batch, sigma=0.02):
    """
    Spectral jitter for the HSI cube:
    multiplicative band-wise noise.
    """
    out = batch["output_data"]         # [Cout, H, W]

    if out.numel() == 0:
        # submission mode: no GT
        return batch

    C = out.shape[0]
    jitter = 1.0 + sigma * torch.randn(C, 1, 1, dtype=out.dtype, device=out.device)
    out = out * jitter

    batch["output_data"] = out
    return batch
