import torch
import numpy as np

from csv import reader

# number of bands used for render (first 31)
N_VIS_BANDS = 31

def hsi2xyz(hsi: torch.Tensor):
    illu = None
    cmf = None

    with open(f"data/render_srgb/illuminant_d65.csv", "r", newline='') as f:
        for row in reader(f):
            illu = np.array(row, dtype=np.float64)

    with open(f"data/render_srgb/std_observer_CIE_1931_2deg.csv", "r", newline='') as f:
        cmf = np.ndarray((N_VIS_BANDS, 3), dtype=np.float64)
        for i, row in enumerate(reader(f)):
            cmf[i] = row

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    illu = torch.from_numpy(illu).to(dev)
    cmf = torch.from_numpy(cmf).to(dev)

    xbar = cmf[:, 0]
    ybar = cmf[:, 1]
    zbar = cmf[:, 2]
    d_lambda = 10.0 # 10 nm between each band 

    N = torch.sum(illu * ybar * d_lambda)

    hsi_hwc = hsi[0:N_VIS_BANDS, :, :].permute((1, 2, 0)).to(dev)
    X = 1/N * torch.sum(hsi_hwc * illu * xbar * d_lambda, dim=2)
    Y = 1/N * torch.sum(hsi_hwc * illu * ybar * d_lambda, dim=2)
    Z = 1/N * torch.sum(hsi_hwc * illu * zbar * d_lambda, dim=2)

    # render in CIE XYZ colorspace
    XYZ = torch.stack([X, Y, Z], dim=2).clamp(0.0, None)

    return XYZ

def xyz2rgb(xyz: torch.Tensor):
    M = torch.tensor([[ 3.2406, -1.5372, -0.4986],
                      [-0.9689,  1.8758,  0.0415],
                      [ 0.0557, -0.2040,  1.0570]], dtype=torch.float64).to(xyz.device)

    # convert to linear RGB
    rgb = xyz @ M.T
    return rgb

def rgb2srgb(rgb: torch.Tensor):
    # convert to sRGB
    a = 0.055; threshold = 0.0031308
    sRGB = torch.where(rgb <= threshold, 12.92 * rgb,
                   (1 + a) * torch.pow(rgb, 1/2.4) - a)
    return sRGB

def hsi2srgb(hsi: torch.Tensor):
    xyz = hsi2xyz(hsi)
    rgb = xyz2rgb(xyz)
    sRGB = rgb2srgb(rgb)

    # output as HWC
    return sRGB.permute((2, 0, 1)).to(torch.float32)
