from typing import Tuple, Literal
import numpy as np
from dataclasses import dataclass, field

# TODO: make trainer configs less messy. separate trainer configs from model configs

@dataclass
class TrainerCfg:
    out_dir: str = "runs/track2/mstpp_up_baseline"
    epochs: int = 14
    amp: bool = True
    save_best: bool = True
    psnr_range: Tuple[float, float] = (20.0, 50.0)  # for reporting scale only
    log_csv_name: str = "train_log.csv"
    wl_61: np.ndarray = field(default_factory=lambda: np.arange(400, 1001, 10))  # 61 bands

    # how many epochs to wait before evaluating all metrics
    metrics_report_interval: int = 5

    # Optimizer & scheduler settings
    # TODO: allow use of Adam instead of AdamW (the original track2 script used Adam)
    lr: float = 4e-4
    weight_decay: float = 1e-4
    scheduler_type: Literal["cosine", "none"] = "cosine"
    eta_min: float = 1e-6
    lambda_sam: float = 0.1  # SAM loss weight

    # model parameters
    in_channels: int = 3
    out_channels: int = 61
    n_feat: int =  61
    stage: int = 3
    upscale_factor: int  = 2
