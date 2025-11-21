from __future__ import annotations
from typing import Dict, Optional, Tuple

import torch
import numpy as np

from utils.render_srgb import hsi2srgb

from torchmetrics.functional.image import spectral_angle_mapper as sam_metric
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr_metric
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_metric
from torchmetrics.functional.image import error_relative_global_dimensionless_synthesis as ergas_metric

# --- scaling helpers (all return [0,1]) ---
# def _exp_score(x_mean: float, tau: float) -> float:
#     return float(np.exp(-float(x_mean) / float(tau)))

def _exp_score(x_mean: float, tau: float, min_val: float = 1e-6) -> float:
    score = np.exp(-float(x_mean) / float(tau))
    return float(np.clip(score, min_val, 1.0))  # avoid 0.0

def _normalize_psnr(psnr_db: float, lo: float = 20.0, hi: float = 50.0) -> float:
    return float(np.clip((psnr_db - lo) / (hi - lo), 0.0, 1.0))

def _deltaE00_mean(rgb1: np.ndarray, rgb2: np.ndarray) -> float:
    import colour
    XYZ1 = colour.sRGB_to_XYZ(rgb1); XYZ2 = colour.sRGB_to_XYZ(rgb2)
    Lab1 = colour.XYZ_to_Lab(XYZ1);  Lab2 = colour.XYZ_to_Lab(XYZ2)
    dE = colour.difference.delta_E(Lab1.reshape(-1,3), Lab2.reshape(-1,3), method="CIE 2000")
    return float(np.mean(dE))

def sid_metric(pred: torch.Tensor, gt: torch.Tensor, eps = 1e-12):
    # normalize like probability distribution
    norm_gt = gt / (gt.sum(dim=0, keepdim=True) + eps)    
    norm_pred = pred / (pred.sum(dim=0, keepdim=True) + eps)    

    # kullback-leibler divergences
    D_pq = torch.sum(norm_gt * torch.log((norm_gt + eps)/(norm_pred + eps)), dim=0)
    D_qp = torch.sum(norm_pred * torch.log((norm_pred + eps)/(norm_gt + eps)), dim=0)

    return torch.mean(D_pq + D_qp)

# TODO (IMPORTANT): implement masking for all metrics
def evaluate_pair_ssc(
    gt_cube: torch.Tensor,       # HxWxC or CxHxW reflectance [0,1]
    pr_cube: torch.Tensor,       # HxWxC or CxHxW
    # wl_nm: np.ndarray,         # (C,)
    mask: Optional[np.ndarray] = None,  # HxW bool

    # weights for the 3 groups (spectral, spatial, color)
    weights: Tuple[float, float, float] = (0.5, 0.35, 0.15),

    # spectral scaling τ
    taus = dict(sam=5.0, sid=0.02, ergas=3.0, de=3.0),

    # PSNR normalization range
    psnr_range: Tuple[float, float] = (20.0, 50.0),
) -> Dict[str, float]:
    """
    Returns all subscores in [0,1] and the final SSC in [0,1].
    """
    # gt_cube_np = gt_cube.detach().cpu().numpy()
    # pr_cube_np = pr_cube.detach().cpu().numpy()

    # --- spectral metrics (means over mask) ---
    with torch.no_grad():
        sam_mean = sam_metric(pr_cube.unsqueeze(0), gt_cube.unsqueeze(0)).clamp(-1.0, 1.0).rad2deg().item()
        sid_mean = sid_metric(pr_cube, gt_cube).item()
        erg_val  = ergas_metric(pr_cube.unsqueeze(0), gt_cube.unsqueeze(0), ratio=1.0).item()

    S_SAM    = _exp_score(sam_mean, taus["sam"])
    S_SID    = _exp_score(sid_mean, taus["sid"])
    S_ERGAS  = _exp_score(erg_val,  taus["ergas"])
    S_spec   = (S_SAM * S_SID * S_ERGAS) ** (1/3)

    gt_rgb_t = hsi2srgb(gt_cube)
    pr_rgb_t = hsi2srgb(pr_cube)
    
    gt_rgb = gt_rgb_t.permute(1, 2, 0).cpu().numpy()
    pr_rgb = pr_rgb_t.permute(1, 2, 0).cpu().numpy()

    with torch.no_grad():
        psnr_val = psnr_metric(pr_rgb_t, gt_rgb_t, data_range=1.0).item()
        ssim_val = ssim_metric(pr_rgb_t.unsqueeze(0), gt_rgb_t.unsqueeze(0), data_range=1.0, reduction="elementwise_mean").item()

    S_PSNR   = _normalize_psnr(psnr_val, *psnr_range)
    S_spat   = 0.5 * (S_PSNR + float(ssim_val))

    dE_mean  = _deltaE00_mean(gt_rgb, pr_rgb)
    S_color  = _exp_score(dE_mean, taus["de"])

    # --- final SSC (weighted geometric mean) ---
    ws, wp, wc = weights
    SSC_geom = (S_spec**ws * S_spat**wp * S_color**wc) ** (1.0 / (ws + wp + wc))
    SSC_arith = (ws * S_spec + wp * S_spat + wc * S_color) / (ws + wp + wc)

    return dict(
        # raw metrics
        SAM_deg=float(sam_mean), SID=float(sid_mean), ERGAS=float(erg_val),
        PSNR_dB=float(psnr_val), SSIM=float(ssim_val), DeltaE00=float(dE_mean),

        # group subscores (0..1)
        S_SAM=float(S_SAM), S_SID=float(S_SID), S_ERGAS=float(S_ERGAS),
        S_PSNR=float(S_PSNR), S_SSIM=float(ssim_val),
        S_SPEC=float(S_spec), S_SPAT=float(S_spat), S_COLOR=float(S_color),

        # final
        SSC_arith=float(SSC_arith), SSC_geom=float(SSC_geom)
    )


# ################ USAGE EXAMPLE ################
# # gt_cube, pr_cube: (H,W,61) reflectance in [0,1]
# # wl_61: np.array of wavelengths
# # mask: optional HxW bool
# from utils.leaderboard_ssc import evaluate_pair_ssc

# res = evaluate_pair_ssc(gt_cube, pr_cube, wl_61, mask=None)
# print(res["SSC"], res)
# ################ USAGE EXAMPLE ################
