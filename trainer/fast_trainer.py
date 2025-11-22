from __future__ import annotations
import wandb
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import csv
import time
import numpy as np

import torch
from torch.utils.data import DataLoader

# ---- bring your stuff ----
# - loss: ReconLoss (L1 + SAM) or any callable loss(pred, target, mask=None)
# - metrics: rmse/sam/sid/ergas/psnr/ssim
# - render: function to convert HSI cube -> sRGB under D65
from .losses import ReconLoss
from models.rev3dcnn import Rev3DCNN

from utils.leaderboard_ssc import evaluate_pair_ssc

from .trainer import TrainerCfg

class FastTrainer:
    """
    Generic trainer for RAW mosaic -> HSI models.

    Expects each batch dict with:
      - "mosaic": (N,1,H,W) float in [0,1]
      - "cube":   (N,C,H,W) float in [0,1] (C=61 for your case)
      - Optional "mask": (N,1,H,W) bool/float (ROI), used only for metrics if present
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader_in: DataLoader,
        train_loader_out: DataLoader,
        val_loader_in: DataLoader,
        val_loader_out: DataLoader,
        loss_fn: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        cfg: TrainerCfg = TrainerCfg(),
        wandb_run: Optional[wandb.Run] = None
    ):
        self.cfg = cfg

        self.train_loader_in = train_loader_in
        self.train_loader_out = train_loader_out
        self.val_loader_in = val_loader_in
        self.val_loader_out = val_loader_out

        self.device = device 

        self.model = model
        self.model.to(self.device)
        self.scaler = torch.cuda.amp.GradScaler() if (self.cfg.amp and self.device.type == "cuda") else None
        self.loss_fn = loss_fn if loss_fn is not None else ReconLoss(lambda_sam=0.1)
        
        if cfg.optim == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        elif cfg.optim == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        else:
            raise ValueError(f"'{cfg.optim}' is not a valid optimizer. Must be: 'adam' or 'adamw'")


        if self.cfg.scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.cfg.epochs, eta_min=self.cfg.eta_min)
        else:
            self.scheduler = None

        # I/O
        self.out_dir = Path(self.cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_best = self.out_dir / "model_best.tar"
        self.ckpt_last = self.out_dir / "model_last.tar"
        self.log_csv = self.out_dir / "train_log.csv"

        # CSV header
        if not self.log_csv.exists():
            with open(self.log_csv, "w", newline="") as f:
                w = csv.writer(f)
                header = [
                    "epoch", "lr", "train_loss", "val_loss",
                    "SAM_deg", "SID", "ERGAS", # spectral metrics
                    "PSNR_dB", "SSIM",         # spatial metrics
                    "E00",                     # color metric
                    "SSC(arith)", "SSC(geom)"  # final score
                ]
                w.writerow(header)

        self.best_val = float("inf")

        self.wandb_run: Optional[wandb.Run] = wandb_run
        if self.wandb_run is not None:
            # align every metric to epoch
            self.wandb_run.define_metric("*", step_metric="epoch")

    def _forward_loss(self, input_img: torch.Tensor, output_cube: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.scaler is None:
            pred = self.model(input_img)
            loss = self.loss_fn(pred, output_cube)
            return pred, loss
        else:
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                pred = self.model(input_img)
                loss = self.loss_fn(pred, output_cube)
            return pred, loss

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        running = 0.0
        n_samples = 0
        total_samples = len(self.train_loader_in.dataset)

        self.train_loader_in.dataset.transforms.set_epoch(epoch)
        self.train_loader_out.dataset.transforms.set_epoch(epoch)
        self.train_loader_in.sampler.set_epoch(epoch) # same instance for in/out

        total_io_time = 0.0
        total_time_start = time.perf_counter()

        it_out = iter(self.train_loader_out)
        gt = next(it_out)[0].to(self.device, non_blocking=True)
        first_iter = True

        for input_img, _ in self.train_loader_in:
            input_img = input_img.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if self.scaler is not None:
                with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                    pred = self.model(input_img)

                    # we're not counting I/O for the input image but that's small anyway
                    io_time_start = time.perf_counter()
                    # get gt while GPU computes
                    if first_iter:
                       first_iter = False
                    else:
                       gt = next(it_out)[0].to(self.device, non_blocking=True)
                    total_io_time += time.perf_counter() - io_time_start

                    loss = self.loss_fn(pred, gt)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            elif not self.cfg.rev_mode:
                # begin forward pass BEFORE fetching gt
                pred = self.model(input_img)

                # get gt while GPU computes
                if first_iter:
                    first_iter = False
                else:
                    # we're not counting I/O for the input image but that's small anyway
                    io_time_start = time.perf_counter()
                    gt = next(it_out)[0].to(self.device, non_blocking=True)
                    total_io_time += time.perf_counter() - io_time_start

                loss = self.loss_fn(pred, gt)
                loss.backward()
                self.optimizer.step()
            else:
                self.model: Rev3DCNN
                pred = self.model.rev_pass_forward(input_img)

                # we're not counting I/O for the input image but that's small anyway
                io_time_start = time.perf_counter()
                # get gt while GPU computes
                if first_iter:
                   first_iter = False
                else:
                   gt = next(it_out)[0].to(self.device, non_blocking=True)
                total_io_time += time.perf_counter() - io_time_start

                loss = self.loss_fn(pred, gt)
                self.model.rev_pass_backward(loss)
                self.optimizer.step()

            running += float(loss.item()) * input_img.size(0)
            n_samples += input_img.size(0)
            if n_samples > 0:
                print(f"[{epoch:03d}] (training...) running avg loss: {(running / n_samples):7.6f}   [ {n_samples:3d} / {total_samples:3d}]", end = '\r')
        
                     
        avg = running / max(n_samples, 1)

        if self.wandb_run is not None:
            self.wandb_run.log({"epoch": epoch, "lr": self._current_lr(), "train loss": avg})

        if self.scheduler is not None:
            self.scheduler.step()

        total_time = time.perf_counter() - total_time_start
        print(f"\n[{epoch:03d}] training finished.",
              f"Total time: {total_time:.3f}s",
              f"/ Train time: {total_time - total_io_time:.3f}s",
              f"/ I/O lag: {(total_io_time):.3f}s")

        return avg

    @torch.no_grad()
    def validate(self, epoch: int, validate_metrics: bool) -> Dict[str, float]:
        """
        Returns dict with: val_loss, SAM_deg, SID, ERGAS, PSNR_dB, SSIM, (DeltaE00 optional).
        """
        self.model.eval()

        loss_sum = 0.0
        n_samples = 0
        total_samples = len(self.val_loader_in.dataset)

        lists_dict: Dict[str, List[float]] = {}
        metric_keys = [
                # raw metrics
                "SAM_deg", "SID", "ERGAS", # spectral
                "PSNR_dB", "SSIM",         # spatial
                "DeltaE00",                # color

                # normalized metrics
                "S_SAM", "S_SID", "S_ERGAS", "S_PSNR", "S_SSIM",

                # grouped scores
                "S_SPEC", "S_SPAT", "S_COLOR",

                # final scores, using arithmetic and geometric mean
                "SSC_arith", "SSC_geom"
        ]

        for k in metric_keys:
            lists_dict[k] = []

        total_io_time = 0.0
        total_time_start = time.perf_counter()

        out_it = iter(self.val_loader_out)
        gt_cube = next(out_it)[0].to(self.device, non_blocking=True)
        first_iter = True

        for input_img, _ in self.val_loader_in:
            input_img = input_img.to(self.device, non_blocking=True)

            # forward (no grad)
            with torch.no_grad():
                pred_cube = self.model(input_img).clamp(0, 1)
                if first_iter:
                    first_iter = False
                else:
                    io_time_start = time.perf_counter()
                    gt_cube = next(out_it)[0].to(self.device, non_blocking=True)
                    total_io_time += time.perf_counter() - io_time_start
                loss = self.loss_fn(pred_cube, gt_cube)

            loss_sum += float(loss.item()) * input_img.size(0)
            n_samples += input_img.size(0)

            if n_samples > 0:
                print(f"[{epoch:03d}] (validating...) running avg loss: {(loss_sum / n_samples):7.6f}   [ {n_samples:3d} / {total_samples:3d}]", end = '\r')

            # per-sample metrics
            if validate_metrics:
                for i in range(pred_cube.size(0)):
                    # --- spectral metrics (means over mask) ---
                    scores = evaluate_pair_ssc(gt_cube[i].detach(), pred_cube[i].detach())

                    for k in metric_keys:
                        lists_dict[k].append(scores[k])

        total_time = time.perf_counter() - total_time_start
        print(f"\n[{epoch:03d}] validation finished.",
              f"Time: {total_time:.3f}s",
              f"/ Val time: {total_time - total_io_time:.3f}s",
              f"/ I/O lag: {(total_io_time):.3f}s")

        out: Dict[str, float] = {}
        out["val_loss"] = loss_sum / max(n_samples, 1)
        for k in metric_keys:
            if lists_dict[k]:
                out[k] = float(np.mean(lists_dict[k]))

        if self.wandb_run is not None:
            wandb_out = out.copy()
            wandb_out["epoch"] = epoch
            self.wandb_run.log(wandb_out)

        return out

    def _current_lr(self) -> float:
        if self.optimizer.param_groups:
            return float(self.optimizer.param_groups[0].get("lr", 0.0))
        return 0.0

    def _log_csv(self, epoch: int, train_loss: float, val_stats: Dict[str, float]):
        row = [
            epoch,
            self._current_lr(),
            train_loss,
            val_stats.get("val_loss", " "),
            val_stats.get("SAM_deg", " "),
            val_stats.get("SID", " "),
            val_stats.get("ERGAS", " "),
            val_stats.get("PSNR_dB", " "),
            val_stats.get("SSIM", " "),
            val_stats.get("E00", " "),
            val_stats.get("SSC_arith",  " "),
            val_stats.get("SSC_geom", " ")
        ]
        with open(self.log_csv, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def _print_epoch(self, epoch: int, train_loss: float, val_stats: Dict[str, float], report_metrics: bool):
        parts = [
            # --- Epoch and Learning Info ---
            f"[{epoch:03d}] epoch finished |",
            f"lr: {self._current_lr():.2e}",
            f"train: {train_loss:7.4f}",
            f"val: {val_stats.get('val_loss', float('nan')):7.4f}"
        ]

        if report_metrics:
            parts += [
                # --- Core Reconstruction Metrics ---
                f"\n[{epoch:03d}] RAW METRICS    |",
                f"SAM(deg): {val_stats.get('SAM_deg', float('nan')):6.2f}",
                f"SID: {val_stats.get('SID', float('nan')):7.4f}",
                f"ERGAS: {val_stats.get('ERGAS', float('nan')):6.3f}",
                f"PSNR(dB): {val_stats.get('PSNR_dB', float('nan')):6.2f}",
                f"SSIM: {val_stats.get('SSIM', float('nan')):5.3f}",
                f"DE00: {val_stats.get('DeltaE00', float('nan')):5.3f}",

                f"\n[{epoch:03d}] SPECTRAL SCORE |",
                f"S_SPEC:  {val_stats.get('S_SPEC', float('nan')):5.5f} |",
                f"S_SAM: {val_stats.get('S_SAM', float('nan')):5.5f}",
                f"S_SID: {val_stats.get('S_SID', float('nan')):5.5f}",
                f"S_ERGAS: {val_stats.get('S_ERGAS', float('nan')):5.5f}",

                f"\n[{epoch:03d}] SPATIAL SCORE  |",
                f"S_SPAT:  {val_stats.get('S_SPAT', float('nan')):5.5f} |",
                f"S_PSNR: {val_stats.get('S_PSNR', float('nan')):5.5f}",
                f"S_SSIM: {val_stats.get('SSIM', float('nan')):5.5f}",

                f"\n[{epoch:03d}] COLOR SCORE    |",
                f"S_COLOR: {val_stats.get('S_COLOR', float('nan')):5.5f}",

                # --- Final Score ---
                f"\n[{epoch:03d}] FINAL SCORE    |",
                f"SSC_arith: {val_stats.get('SSC_arith', float('nan')):5.5f}",
                f"SSC_geom: {val_stats.get('SSC_geom', float('nan')):5.5f}",
            ]
        print("  ".join(parts))

    def _save_checkpoint(self, epoch: int, is_best: bool):
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
        }
        torch.save(state, self.ckpt_last)
        if is_best:
            torch.save(state, self.ckpt_best)
            print(f"[Saved BEST model @ epoch {epoch}] → {self.ckpt_best}")

    def fit(self):
        print(f"Start training for {self.cfg.epochs} epochs. Logs → {self.log_csv}")
        for ep in range(1, self.cfg.epochs + 1):
            train_loss = self.train_epoch(ep)

            validate_metrics: bool = (ep % self.cfg.metrics_report_interval) == 0 if \
                                     self.cfg.metrics_report_interval > 0 else False
            val_stats = self.validate(ep, validate_metrics=validate_metrics)

            self._print_epoch(ep, train_loss, val_stats, report_metrics=validate_metrics)
            self._log_csv(ep, train_loss, val_stats)

            is_best = val_stats["val_loss"] < self.best_val
            if is_best:
                self.best_val = val_stats["val_loss"]
            self._save_checkpoint(ep, is_best)
