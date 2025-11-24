import torch
import torch.nn as nn
import torch.nn.functional as F

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

class MosaicUp(nn.Module):
    def __init__(self, old_mode: bool = False):
        """FOR ICASSP CHALLENGE:
            'Upscales' RGB input by a factor of two while turning it into a mosaic input
             compatible with RevSCI.
             TODO: Better way to do this?
        """
        super().__init__()
        self.old_mode = old_mode

    def forward(self, img: torch.Tensor):
        out = F.interpolate(img, scale_factor=2, mode='bicubic')
        
        mask_r = torch.zeros((out.shape[-2], out.shape[-1])).to(out.device)
        mask_g = torch.zeros((out.shape[-2], out.shape[-1])).to(out.device)
        mask_b = torch.zeros((out.shape[-2], out.shape[-1])).to(out.device)

        mask_r[0::2, 0::2] = 1.0
        mask_g[0::2, 1::2] = 1.0
        mask_g[1::2, 0::2] = 1.0
        mask_b[1::2, 1::2] = 1.0

        out = out[:, 0] * mask_r + out[:, 1] * mask_g + out[:, 2] * mask_b
        out = out.unsqueeze(1)

        if not self.old_mode:
            B, _, H, W = out.shape
            ry = H // 2 + 1
            rx = W // 2 + 1
            repeated_bayer = _bayer.to(out.device).unsqueeze(0).repeat(B, 1, ry, rx)[:, :, :H, :W]

            # concat bayer with input to ensure alignment in transforms
            out = torch.cat([out, repeated_bayer], dim=1) # (B, 4, H, W)

        return out


