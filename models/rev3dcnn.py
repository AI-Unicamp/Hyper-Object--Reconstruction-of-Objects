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
    assert imgs.shape[1] == 1, f"Expected `img` with 1 channel, got {imgs.shape[1]} channels."

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
        msfa (torch.Tensor): MSFA to use against the raw data (shape: C x X x Y).
        n_blocks (int): number of reversible blocks.
        n_split (int): number of splits in each rev block.
    """

    def __init__(self, n_blocks: int, n_split: int):
        super().__init__()

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

    def forward(self, raw: Tensor) -> Tensor:
        """
        Args:
            raw (Tensor): the raw single-channel image (shape: B x 1 x W x H).

        Returns:
            Tensor: the demosaicked image (shape: B x C x W x H).
        """
        out: Tensor = mosaic(raw)

        # input shape: B x 1 x D x W x H
        # bands go on D
        out = self.conv1(out.unsqueeze(1))
        for layer in self.layers:
            out = layer(out)
        out = self.conv2(out).squeeze(1)
        return out

    # TODO: allow reverse-mode training (preserves memory)
    def for_backward(self,
                     raw: Tensor,
                     gt: Tensor,
                     loss_fn: Callable[[Tensor, Tensor], Tensor],
                     opt: Optional[Optimizer]) -> tuple[Tensor, Tensor]:
        """Memory-efficient training using reversability. Uses less memory, but takes longer.

        Executes backpropagation for one batch, stepping the optimizer, if passed.

        Args:
            raw (torch.Tensor): batch to run (shape: B x 1 x W x H)
            gt (torch.Tensor): ground-truth/target (shape: B x C x W x H)
            loss_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): loss function to use
            opt (torch.Optimizer): optimizer to use (optional)

        Returns:
            torch.Tensor: the model's predcition
            torch.Tensor: the loss from prediction vs. ground-truth
        """
        unf: Tensor = self.unflatten(raw).unsqueeze(1)

        # compute conv1, with grads. we keep it saved
        out_conv1: Tensor = self.conv1(unf)

        # skip grads for rev-blocks
        out = out_conv1
        with torch.no_grad():
            for layer in self.layers:
                out = layer(out)
        out = out.requires_grad_()

        # get grads for conv2, save the prediction
        pred: Tensor = self.conv2(out)

        # back-propagate, only until right before conv2
        loss = loss_fn(pred, gt.unsqueeze(1))
        loss.backward()

        # setting up reversal
        out_curr = out
        last_grad = out.grad
        # we go through layers in reverse, saving only the gradients we need and
        # thus saving up on memory
        for layer in reversed(self.layers):
            # we reverse, so we can forward again and get the grads
            with torch.no_grad():
                out_pre = layer.reverse(out_curr)
            out_pre.requires_grad_()

            # the values on this tensor are the same as out_curr, but they have gradients now
            # TODO: maybe something more manual would be faster? we already have out_cur without
            # gradients, maybe there's a quicker way to set them up
            out_curr_with_grad: Tensor = layer(out_pre)
            out_curr_with_grad.backward(gradient=last_grad)

            # set up next iteration
            last_grad = out_pre.grad
            out_curr = out_pre

        # then we do the back-prop for conv1
        out_conv1.backward(gradient=last_grad)

        # step optimizer
        if opt is not None:
            opt.step()
            opt.zero_grad()

        return pred.squeeze(1), loss
