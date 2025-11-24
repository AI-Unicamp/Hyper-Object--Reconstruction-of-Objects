import torch
import torch.nn as nn

from typing import Callable, Optional 
from torch import Tensor # makes typing a little nicer
from torch.optim import Optimizer

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

def mosaic(imgs: Tensor) -> Tensor:
    """
    Turns 1-channel batched images into sparse 3-channel with Bayer pattern.
    """
    assert imgs.shape[1] == 1, f"Expected `img` with 1 channel, got {imgs.shape[1]} channels. You might need to change old_mode to False in the configs."

    repeated_x = imgs.expand(-1, 3, -1, -1)

    _, _, H, W = imgs.shape
    ry = H // 2 + 1
    rx = W // 2 + 1

    repeated_bayer = _bayer.to(imgs.device).repeat(1, rx, ry)[:, :H, :W]

    repeated_bayer = repeated_bayer.unsqueeze(0).expand(imgs.shape[0], -1, -1, -1)

    return repeated_x * repeated_bayer

def split_n_features(x: Tensor, n: int):
    """Splits the tenosr's channels into n groups."""
    x_list = list(torch.chunk(x, n, dim=1))
    return x_list

class f_g_layer(nn.Module):
    """Single part of a RevBlock.

    Applies two Conv3d's with 3x3 kernel and padding 1, with leaky ReLU in the middle.
    The shape of the tensor is preserved throughout.

    Args:
        channels (int): number of input (and output) channels
    """

    def __init__(self, channels: int):
        super(f_g_layer, self).__init__()
        self.nn_layer = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.nn_layer(x)
        return x

class RevBlock(nn.Module):
    """A reversible block of the RevSCI.

    Splits the image into `n` features, and applies `f_g_layer` successively,
    concatenating the results as it goes. As such, there will be in total `in_channels // n` layers inside the block.
    `n` must divide `in_channels`. The shape of the tensor is preserved throughout

    TODO (?): make it so `n` doesn't need to divide in_channels

    Args:
        in_channels (int): number of input (and output) channels
        n (int): number of features to split into
    """

    def __init__(self, in_channels: int, n: int):
        super(RevBlock, self).__init__()
        self.f = nn.ModuleList()
        self.n = n
        self.ch = in_channels
        for _ in range(n):
            self.f.append(f_g_layer(in_channels // n))

    def forward(self, x: Tensor) -> Tensor:
        feats = split_n_features(x, self.n)
        h_new = feats[-1] + self.f[0](feats[0])
        h_curr = h_new
        for i in range(1, self.n):
            h_new = feats[-(i+1)] + self.f[i](h_new)
            h_curr = torch.cat([h_curr, h_new], dim=1)
        return h_curr

    def reverse(self, y: Tensor) -> Tensor:
        """Applies the reverse transformation from `forward`.

        Used for reverse mode training, generally with `torch.no_grad`.
        """
        l = split_n_features(y, self.n)
        h_new = l[-1] - self.f[-1](l[-2])
        h_curr = h_new
        for i in range(2, self.n):
            h_new = l[-i] - self.f[-i](l[-(i+1)])
            h_curr = torch.cat([h_curr, h_new], dim=1)

        # we slice h_curr to obtain the first layer
        h_new = l[0] - self.f[0](h_curr[:, 0:(self.ch // self.n), ::])
        h_curr = torch.cat([h_curr, h_new], dim=1)
        return h_curr



class Rev3DCNN(nn.Module):
    """Implementation of RevSCI, from:
    Z. Cheng et al., "Memory-Efficient Network for Large-scale Video Compressive Sensing," doi: 10.1109/CVPR46437.2021.01598.

    And code based on: https://github.com/BoChenGroup/RevSCI-net.

    Args:
        n_blocks (int): number of reversible blocks.
        n_split (int): number of splits in each rev block.

        FOR ICASSP CHALLENGE:
        old_mode (bool): if True, takes in 1-channel inputs instead of 1-channel concat'ed with RGB mosaic. 
    """

    def __init__(self, n_blocks: int, n_split: int, old_mode: bool = False):
        super().__init__()

        self.old_mode = old_mode

        # encoding / feature extraction
        self.conv1 = nn.Sequential(
            # input shape: B x 1 x D x W x H
            #         bands go here^
            nn.Conv3d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.LeakyReLU(inplace=True),
        )
        
        # rev blocks
        self.layers = nn.ModuleList()
        for _ in range(n_blocks):
            self.layers.append(RevBlock(64, n_split))

        # decoding
        self.conv2 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1),
                               output_padding=(0, 1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, 16, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(16, 1, kernel_size=3, stride=1, padding=1),
        )

        # save output of conv1 and blocks when doing rev passes
        self._last_out_conv1 = torch.tensor([])
        self._last_out_revblocks = torch.tensor([])

    def forward(self, inp: Tensor) -> Tensor:
        """
        Args:
            FOR ICASSP CHALLENGE:
            inp (Tensor): raw single-channel concatenated with an RGB mask (shape: B x 4 x W x H).

            Used to take in the single-channel input (shape: B x 1 x W x H) only. To use it like this, you
            must set old_mode to True.

        Returns:
            Tensor: the demosaicked image (shape: B x C x W x H).
        """
        out: Tensor
        if not self.old_mode:
            assert inp.size(1) == 4, "Input size for TRevSCI is incorrect. If you are using a model from an older run, try setting 'old_mode' to True in the config."

            raw = inp[:, 0, :, :].unsqueeze(1)
            mask = inp[:, 1:, :, :]

            out = raw * mask
        else:
            out = mosaic(inp)

        # input shape: B x 1 x D x W x H
        # bands go on D
        out = self.conv1(out.unsqueeze(1))
        for layer in self.layers:
            out = layer(out)
        out = self.conv2(out).squeeze(1)
        return out

    def rev_pass_forward(self, inp: Tensor) -> Tensor:
        """Forward pass skipping gradients in rev blocks."""
        out: Tensor
        if not self.old_mode:
            raw = inp[:, 0, :, :].unsqueeze(1)
            mask = inp[:, 1:, :, :]

            out = raw * mask
        else:
            out = mosaic(inp)

        out = self.conv1(out.unsqueeze(1))
        self._last_out_conv1 = out 

        with torch.no_grad():
            for layer in self.layers:
                out = layer(out)
        out = out.requires_grad_()
        self._last_out_revblocks = out

        pred = self.conv2(out)
        return pred.squeeze(1)

    def rev_pass_backward(self, loss: Tensor):
        loss.backward()

        out_curr = self._last_out_revblocks
        last_grad = out_curr.grad

        for layer in reversed(self.layers):
            with torch.no_grad():
                out_pre = layer.reverse(out_curr)
            out_pre.requires_grad_()

            out_curr_with_grad: Tensor = layer(out_pre)
            out_curr_with_grad.backward(gradient=last_grad)

            last_grad = out_pre.grad
            out_curr = out_pre

        self._last_out_conv1.backward(gradient=last_grad)
