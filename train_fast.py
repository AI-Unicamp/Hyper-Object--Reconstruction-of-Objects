import torch
import yaml
import os
import argparse
from datetime import datetime
from shutil import copy2

from torch.utils.data import DataLoader
from trainer.losses import ReconLoss
from trainer.trainer import TrainerCfg
from utils.tools_wandb import ToolsWandb

from datasets.partial import PartialDataset
from datasets.det_transform import DeterministicTransforms

from models import setup_model

from typing import Dict, Any

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--track", type=int, required=False, help="track to run")
parser.add_argument("-d", "--data_dir", type=str, default="./data", required=False, help="path to dataset directory")
parser.add_argument("-c", "--config", type=str, required=False, help="path of config file to use (defaults to baselines for each track)")
parser.add_argument("-i", "--data_in", type=str, required=False, help="dataset to use for input loader (e.g. rgb_2, mosaic, etc.)")
parser.add_argument("-o", "--data_out", type=str, default="hsi_61_zarr", required=False, help="dataset to use for output loader (e.g. hsi_61, hsi_61_zarr, etc.)")
parser.add_argument("-s", "--seed", type=str, required=False, help="seed for transform/shuffler RNG")

# TODO: implement
# parser.add_argument("--continue_from", type=str, required=False, help="checkpoint to start from")
parser.add_argument("--use_wandb", default=False, required=False, help="log to wandb", action="store_true")

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

if args.seed is not None:
    rng_seed = args.seed
else:
    rng_seed = int(datetime.now().timestamp())

transforms_in = DeterministicTransforms(track=args.track, is_out=False, base_seed=rng_seed)
transforms_out = DeterministicTransforms(track=args.track, is_out=True, base_seed=rng_seed)
if transforms_config.get("random_crop", False):
    transforms_in.add_transform("random_crop", ps=transforms_config.get("crop_size", 320))
    transforms_out.add_transform("random_crop", ps=transforms_config.get("crop_size", 320))
if transforms_config.get("random_flip", False):
    transforms_in.add_transform("random_flip")
    transforms_out.add_transform("random_flip")

if args.data_in is not None:
    img_type_in = args.data_in
elif args.track is not None:
    img_type_in = "mosaic" if args.track==1 else "rgb_2"
else:
    raise ValueError("Must use at least one of the flags '--track' or '--data_in'.")

img_type_out = args.data_out

ds_train_in = PartialDataset(
    data_root=f"{args.data_dir}",
    track=args.track,  # 1 for mosaic, 2 for rgb_2
    dataset_type="train",
    img_type=img_type_in,
    transforms=transforms_in,
    old_mode=model_config.get("old_mode", False)
)

ds_train_out = PartialDataset(
    data_root=f"{args.data_dir}",
    track=args.track,  # 1 for mosaic, 2 for rgb_2
    dataset_type="train",
    img_type=img_type_out,
    transforms=transforms_out,
    old_mode=model_config.get("old_mode", False)
)

ds_val_in = PartialDataset(
    data_root=f"{args.data_dir}",
    track=args.track,  # 1 for mosaic, 2 for rgb_2
    dataset_type="test-public",
    img_type=img_type_in,
    old_mode=model_config.get("old_mode", False)
)

ds_val_out = PartialDataset(
    data_root=f"{args.data_dir}",
    track=args.track,  # 1 for mosaic, 2 for rgb_2
    dataset_type="test-public",
    img_type=img_type_out,
    old_mode=model_config.get("old_mode", False)
)

from torch.utils.data import Sampler
class SharedShuffledSampler(Sampler[int]):
    def __init__(self, data_len: int, base_seed: int = 0):
        self.base_seed = int(base_seed)
        self.epoch = 0
        self._indices = torch.arange(data_len)

    def set_epoch(self, epoch: int):
        """Reshuffle indices deterministically for this epoch."""
        self.epoch = int(epoch)
        g = torch.Generator()
        g.manual_seed(self.base_seed + self.epoch)
        perm = torch.randperm(len(self._indices), generator=g)
        self._indices = self._indices[perm]

    def __iter__(self):
        # DataLoader in the main process will call this each epoch
        return iter(self._indices.tolist())

    def __len__(self):
        return len(self._indices)

shared_sampler = SharedShuffledSampler(len(ds_train_in), base_seed=rng_seed)

train_loader_in = DataLoader(
        ds_train_in,
        batch_size=train_config.get("batch_size_train", 4),
        num_workers=train_config["fast"].get("num_workers_train_in", 4),
        shuffle=False,
        sampler=shared_sampler,
        pin_memory=True,
        persistent_workers=True
    )

train_loader_out = DataLoader(
        ds_train_out,
        batch_size=train_config.get("batch_size_train", 4),
        num_workers=train_config["fast"].get("num_workers_train_out", 4),
        shuffle=False,
        sampler=shared_sampler,
        pin_memory=True,
        persistent_workers=True
    )

val_loader_in  = DataLoader(
        ds_val_in,
        batch_size=train_config.get("batch_size_test", 4),
        num_workers=train_config["fast"].get("num_workers_test_in", 4),
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
    )

val_loader_out  = DataLoader(
        ds_val_out,
        batch_size=train_config.get("batch_size_test", 4),
        num_workers=train_config["fast"].get("num_workers_test_out", 4),
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
    )

config_name = os.path.splitext(os.path.basename(config_path))[0]
run_name = f"{config_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
out_dir = f"runs/track{args.track}/{run_name}"

os.makedirs(out_dir)

# save the config that was used
copy2(config_path, f"{out_dir}/config.yaml")

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
        metrics_report_interval=train_config.get("metrics_report_interval", 5),
        rev_mode=train_config.get("rev_mode", False)
    )

loss_fn = ReconLoss(lambda_sam=cfg.lambda_sam)

model = setup_model(model_config)

run_wandb = None
flattened_config = {}
ToolsWandb.config_flatten(config, flattened_config)
if args.use_wandb:
    run_wandb = ToolsWandb.init_wandb_run(
            f_configurations=flattened_config,
            run_name=run_name
            )

from trainer.fast_trainer import FastTrainer
trainer = FastTrainer(
    model=model,
    train_loader_in=train_loader_in,
    train_loader_out=train_loader_out,
    val_loader_in=val_loader_in,
    val_loader_out=val_loader_out,
    loss_fn=loss_fn,
    cfg=cfg,
    device=device,
    wandb_run=run_wandb
)

try:
    trainer.fit()
except KeyboardInterrupt:
    if run_wandb is not None:
        run_wandb.finish()
