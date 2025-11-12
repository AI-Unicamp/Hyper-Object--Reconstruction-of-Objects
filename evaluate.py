import torch
import numpy as np
import yaml
import argparse
from tqdm import tqdm

from datasets.hyper_object import HyperObjectDataset
from models import setup_model
from utils.leaderboard_ssc import evaluate_pair_ssc

from typing import Dict, Any, List

# use batch size 1
BATCH_SIZE = 1

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--track", type=int, required=True, help="track to run")
parser.add_argument("-d", "--data_dir", type=str, default="./data", required=False, help="path to dataset directory")
parser.add_argument("-c", "--config", type=str, required=False, help="path of config file to use (defaults to baselines for each track)")

parser.add_argument("--model", type=str, required=True, help="path to model to submit")

# TODO: implement
# parser.add_argument("-o", "--out_dir", type=str, default="blahblah", required=False, help="path to save generated files")

args = parser.parse_args()

model_path = args.model

if args.config is None:
    if args.track == 1:
        config_path = "config/raw2hsi_baseline.yaml"
    elif args.track == 2:
        config_path = "config/mst_plus_plus_up_baseline.yaml"
    else:
        raise ValueError(f"'{args.track}' is invalid value for track: must be 1 or 2.")
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

print(f"Preparing to train model '{model_config["model"]}'...")
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print("WARNING: GPU not found, using CPU to evaluate.")
    device = torch.device("cpu")

test_dataset = HyperObjectDataset(
    data_root=f'{args.data_dir}/track{args.track}',
    track=args.track,
    train=False,
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
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

for data in tqdm(test_loader, desc="Evaluating test data..."):
    input_img  :   torch.Tensor = data["input"].to(device, non_blocking=True)
    output_cube:   torch.Tensor = data["output"].to(device, non_blocking=True)

    # forward (no grad)
    with torch.no_grad():
        pred_cube = model(input_img).clamp(0, 1)

    # per-sample metrics
    for i in range(pred_cube.size(0)):
        # --- spectral metrics (means over mask) ---

        scores = evaluate_pair_ssc(output_cube[i].detach(), pred_cube[i].detach())

        for k in metric_keys:
            lists_dict[k].append(scores[k])

val_stats: Dict[str, float] = {}
for k in metric_keys:
    if not lists_dict[k]:
        val_stats[k] = float("nan")
    else:
        val_stats[k] = float(np.mean(lists_dict[k]))

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

print("  ".join(parts))
