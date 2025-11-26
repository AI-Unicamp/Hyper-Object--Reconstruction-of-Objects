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
parser.add_argument("--train", default=False, required=False, help="use train set to evaluate", action="store_true")

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
    dataset_type="train" if args.train else "test-public",
    img_type=img_type_in,
    old_mode=model_config.get("old_mode", False)
)

ds_val_out = PartialDataset(
    data_root=f"{args.data_dir}",
    track=args.track,  # 1 for mosaic, 2 for rgb_2
    dataset_type="train" if args.train else "test-public",
    img_type=img_type_out,
    old_mode=model_config.get("old_mode", False)
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

from trainer.losses import ReconLoss

loss_fn = ReconLoss(lambda_sam=config.get("train", {}).get("lambda_sam", 0.1))
loss_fn.to(device)

cats_dict: Dict[str, Dict[str, List[float]]] = {}
cat_keys = [
        "Category-1",
        "Category-2",
        "Category-3",
        "Category-4"
        ]

for cat in cat_keys:
    cats_dict[cat] = {}

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
        "SSC_arith", "SSC_geom", "loss"
]

for cat in cat_keys:
    for metric in metric_keys:
        cats_dict[cat][metric] = []

it_out = iter(val_loader_out)
for input_img, img_ids in tqdm(val_loader_in, desc="Evaluating test data..."):
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

        loss = loss_fn(pred_cube, output_cube).item()
        scores["loss"] = loss

        for cat in cat_keys:
            if cat in img_ids[i]:
                for metric in metric_keys:
                    if metric in scores:
                        cats_dict[cat][metric].append(scores[metric])
                break

cats_stats: Dict[str, Dict[str, float]] = {}
for cat in cat_keys:
    cats_stats[cat] = {}

for cat in cat_keys:
    for metric in metric_keys:
        if cats_dict[cat][metric]: # if list is not empty
            cats_stats[cat][metric] = float(np.mean(cats_dict[cat][metric]))
        else:
            cats_stats[cat][metric] = 0.0

from matplotlib import pyplot as plt

raw_metrics = metric_keys[:6]
metrics_fig, metrics_ax = plt.subplots(2, 3) # raw metrics

for i, metric in enumerate(raw_metrics):
    col = i % 3
    row = (i // 3)

    metrics_ax[row, col].set_title(metric)
    metrics_ax[row, col].set_xticks([1, 2, 3, 4])

    results = [cats_stats[cat][metric] for cat in cats_stats]
    metrics_ax[row, col].bar([1, 2, 3, 4], results)

metrics_fig.tight_layout()

if "rgb" not in img_type_out:
    norm_metrics = metric_keys[6:6+5]
    scores_names = metric_keys[6+5:]

    norm_fig, norm_ax = plt.subplots(2, 3)       # normalized metrics
    score_fig, score_ax = plt.subplots(2, 3)     # score

    for i, metric in enumerate(norm_metrics):
        col = i % 3
        row = i % 2

        norm_ax[row, col].set_title(metric)
        norm_ax[row, col].set_xticks([1, 2, 3, 4])

        results = [cats_stats[cat][metric] for cat in cats_stats]
        norm_ax[row, col].bar([1, 2, 3, 4], results)

    for i, metric in enumerate(scores_names):
        col = i % 3
        row = i % 2

        score_ax[row, col].set_title(metric)
        score_ax[row, col].set_xticks([1, 2, 3, 4])

        results = [cats_stats[cat][metric] for cat in cats_stats]
        score_ax[row, col].bar([1, 2, 3, 4], results)

    norm_fig.tight_layout()
    score_fig.tight_layout()

plt.show()
