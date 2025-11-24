import os
import torch
import numpy as np
import yaml
import argparse
from tqdm import tqdm

from datasets.partial import PartialDataset
from models import setup_model
from utils.leaderboard_ssc import evaluate_pair_ssc, evaluate_reconstruction

from typing import Dict, Any, List

# use batch size 1
BATCH_SIZE = 1

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--track", type=int, required=True, help="track to run")
parser.add_argument("-d", "--data_dir", type=str, default="./data", required=False, help="path to dataset directory")
parser.add_argument("-c", "--config", type=str, required=False, help="path of config file to use (defaults to baselines for each track)")
parser.add_argument("-i", "--data_in", type=str, required=False, help="dataset to use for input loader (e.g. rgb_2, mosaic, etc.)")
parser.add_argument("-o", "--data_out", type=str, default="hsi_61_zarr", required=False, help="dataset to use for output loader (e.g. hsi_61, hsi_61_zarr, etc.)")

parser.add_argument("--model", type=str, required=True, help="path to model to submit")

args = parser.parse_args()

model_path = args.model

if args.config is None:
    # if no config given, assume that it is in the same folder as the model
    config_path = f"{os.path.dirname(model_path)}/config.yaml"
    print(f"No config given, assuming config is at {config_path}")
else:
    config_path = args.config

config: Dict[str, Any]
with open(config_path, "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

if "model" not in config:
    raise ValueError("No model settings found.")
elif "model_name" not in config["model"]:
    raise ValueError("No model name found.")

model_config = config["model"]

print(f"Preparing to evaluate model '{model_config['model_name']}'...")
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print("WARNING: GPU not found, using CPU to evaluate.")
    device = torch.device("cpu")

if args.data_in is not None:
    img_type_in = args.data_in
elif args.track is not None:
    img_type_in = "mosaic" if args.track==1 else "rgb_2"
else:
    raise ValueError("Must use at least one of the flags '--track' or '--data_in'.")

img_type_out = args.data_out

ds_val_in = PartialDataset(
    data_root=f"{args.data_dir}",
    track=args.track,  # 1 for mosaic, 2 for rgb_2
    dataset_type="test-public",
    img_type=img_type_in,
)

ds_val_out = PartialDataset(
    data_root=f"{args.data_dir}",
    track=args.track,  # 1 for mosaic, 2 for rgb_2
    dataset_type="test-public",
    img_type=img_type_out,
)

val_loader_in  = torch.utils.data.DataLoader(
        ds_val_in,
        batch_size=BATCH_SIZE,
        num_workers=2,
        shuffle=False,
        pin_memory=True,
    )

val_loader_out  = torch.utils.data.DataLoader(
        ds_val_out,
        batch_size=BATCH_SIZE,
        num_workers=2,
        shuffle=False,
        pin_memory=True,
    )

print(f"Loading checkpoint from: {model_path}")

model = setup_model(model_config)
checkpoint = torch.load(model_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model"])
model.to(device)
model.eval()

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

it_out = iter(val_loader_out)
for input_img, _ in tqdm(val_loader_in, desc="Evaluating test data..."):
    input_img: torch.Tensor
    input_img = input_img.to(device, non_blocking=True)

    # forward (no grad)
    with torch.no_grad():
        pred_cube = model(input_img).clamp(0, 1)

    output_cube: torch.Tensor = next(it_out)[0].to(device, non_blocking=True) 

    # per-sample metrics
    for i in range(pred_cube.size(0)):
        # --- spectral metrics (means over mask) ---

        if "rgb" not in img_type_out:
            scores = evaluate_pair_ssc(output_cube[i].detach(), pred_cube[i].detach())
        else:
            scores = evaluate_reconstruction(output_cube[i].detach(), pred_cube[i].detach())

        for k in metric_keys:
            if k in scores:
                lists_dict[k].append(scores[k])

val_stats: Dict[str, float] = {}
for k in metric_keys:
    if not lists_dict[k]:
        val_stats[k] = float("nan")
    else:
        val_stats[k] = float(np.mean(lists_dict[k]))

if "rgb" not in img_type_out:
    parts = [
        # --- Core Reconstruction Metrics ---
        f"\nRAW METRICS    |",
        f"SAM(deg): {val_stats.get('SAM_deg', float('nan')):6.2f}",
        f"SID: {val_stats.get('SID', float('nan')):7.4f}",
        f"ERGAS: {val_stats.get('ERGAS', float('nan')):6.3f}",
        f"PSNR(dB): {val_stats.get('PSNR_dB', float('nan')):6.2f}",
        f"SSIM: {val_stats.get('SSIM', float('nan')):5.3f}",
        f"DE00: {val_stats.get('DeltaE00', float('nan')):5.3f}",

        f"\nSPECTRAL SCORE |",
        f"S_SPEC:  {val_stats.get('S_SPEC', float('nan')):5.5f} |",
        f"S_SAM: {val_stats.get('S_SAM', float('nan')):5.5f}",
        f"S_SID: {val_stats.get('S_SID', float('nan')):5.5f}",
        f"S_ERGAS: {val_stats.get('S_ERGAS', float('nan')):5.5f}",

        f"\nSPATIAL SCORE  |",
        f"S_SPAT:  {val_stats.get('S_SPAT', float('nan')):5.5f} |",
        f"S_PSNR: {val_stats.get('S_PSNR', float('nan')):5.5f}",
        f"S_SSIM: {val_stats.get('SSIM', float('nan')):5.5f}",

        f"\nCOLOR SCORE    |",
        f"S_COLOR: {val_stats.get('S_COLOR', float('nan')):5.5f}",

        # --- Final Score ---
        f"\nFINAL SCORE    |",
        f"SSC_arith: {val_stats.get('SSC_arith', float('nan')):5.5f}",
        f"SSC_geom: {val_stats.get('SSC_geom', float('nan')):5.5f}",
    ]
else:
    parts = [
            f"\nMETRICS:",
            f"\n   SAM(deg): {val_stats.get('SAM_deg', float('nan')):6.3f}",
            f"\n   SID:      {val_stats.get('SID', float('nan')):6.4f}",
            f"\n   ERGAS:    {val_stats.get('ERGAS', float('nan')):6.3f}",
            f"\n   PSNR(dB): {val_stats.get('PSNR_dB', float('nan')):6.2f}",
            f"\n   SSIM:     {val_stats.get('SSIM', float('nan')):6.6f}",
            f"\n   DE00:     {val_stats.get('DeltaE00', float('nan')):6.3f}",
        ]

print("  ".join(parts))
