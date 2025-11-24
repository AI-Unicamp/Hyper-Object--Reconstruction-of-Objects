import torch
import torch.nn as nn
import torch.nn.functional as F

class MosaicUp(nn.Module):
    def __init__(self):
        """FOR ICASSP CHALLENGE:
            'Upscales' RGB input by a factor of two while turning it into a mosaic.
             TODO: Better way to do this?
        """
        super().__init__()

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

        return out.unsqueeze(1)


