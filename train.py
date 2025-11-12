import torch
import yaml
import os
import argparse
from datetime import datetime

from torch.utils.data import DataLoader
from trainer.losses import ReconLoss
from trainer.trainer import Trainer, TrainerCfg

from datasets.hyper_object import HyperObjectDataset
from datasets.transform import random_crop, random_flip

from models import setup_model

from typing import Dict, Any

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--track", type=int, required=True, help="track to run")
parser.add_argument("-d", "--data_dir", type=str, default="./data", required=False, help="path to dataset directory")
parser.add_argument("-c", "--config", type=str, required=False, help="path of config file to use (defaults to baselines for each track)")

# TODO: implement
# parser.add_argument("--continue_from", type=str, required=False, help="checkpoint to start from")
# parser.add_argument("--use_wandb", type=bool, default=False, required=False, help="log to wandb")

args = parser.parse_args()

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
train_config = config.get("train", {})
transforms_config = config.get("transforms", {})

print(f"Preparing to train model '{model_config}'...")
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print("WARNING: GPU not found, using CPU to train.")
    device = torch.device("cpu")

# TODO: improve transforms
transforms = []
if transforms_config.get("random_crop", False):
    transforms.append(lambda batch: random_crop(batch, ps=transforms_config.get("crop_size", 320), track=args.track))
if transforms_config.get("random_flip", False):
    transforms.append(random_flip)

def transform(batch):
    for t in transforms:
        batch = t(batch)
    return batch

ds_train = HyperObjectDataset(
    data_root=f"{args.data_dir}/track{args.track}",
    track=args.track,  # 1 for mosaic, 2 for rgb_2
    train=True,
    transforms=transform,
)

ds_val = HyperObjectDataset(
    data_root=f"{args.data_dir}/track{args.track}",
    track=args.track,  # 1 for mosaic, 2 for rgb_2
    train=False,
)

train_loader = DataLoader(
        ds_train,
        batch_size=train_config.get("batch_size_train", 4),
        num_workers=train_config.get("num_workers_train", 4),
        shuffle=True,
        pin_memory=True
    )
val_loader  = DataLoader(
        ds_val,
        batch_size=train_config.get("batch_size_test", 4),
        num_workers=train_config.get("num_workers_test", 4),
        shuffle=False,
        pin_memory=False
    )

config_name = os.path.splitext(os.path.basename(config_path))[0]
out_dir = f"runs/track{args.track}/{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_{config_name}"
cfg = TrainerCfg(
        out_dir=out_dir,
        epochs=train_config.get("epochs", 1000),
        amp=train_config.get("amp", True),
        optim=train_config.get("optim", "adamw"),
        lr=train_config.get("lr", 2e-4),
        weight_decay=train_config.get("weight_decay", 1e-4),
        scheduler_type=train_config.get("scheduler","cosine"),
        eta_min=train_config.get("eta_min", 1e-6),
        lambda_sam=train_config.get("lambda_sam", 0.1),     
        metrics_report_interval=train_config.get("metrics_report_interval", 5)
    )

loss_fn = ReconLoss(lambda_sam=cfg.lambda_sam)

model = setup_model(model_config)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=loss_fn,
    cfg=cfg,
    device=device
)

trainer.fit()
