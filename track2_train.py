from torch.utils.data import DataLoader
from baselines.mstpp_up import MST_Plus_Plus_LateUpsample
from trainer.losses import ReconLoss
from trainer.trainer import Trainer

from datasets.hyper_object import HyperObjectDataset
from datasets.transform import random_crop

from config.track2_cfg_default import TrainerCfg

# TODO: add command-line arguments

ds_train = HyperObjectDataset(
    data_root=f'data/track2',
    track=2,
    train=True,
    transforms=lambda batch: random_crop(batch, ps=128, track=2),
)

ds_val = HyperObjectDataset(
    data_root=f'data/track2',
    track=2,
    train=False,
)

train_loader = DataLoader(ds_train, batch_size=2, shuffle=True, num_workers=0, pin_memory=False)
val_loader   = DataLoader(ds_val,   batch_size=2, shuffle=False, num_workers=0, pin_memory=False)

cfg = TrainerCfg()
model = MST_Plus_Plus_LateUpsample(in_channels=cfg.in_channels,
                                   out_channels=cfg.out_channels,
                                   n_feat=cfg.n_feat,
                                   stage=cfg.stage,
                                   upscale_factor=cfg.upscale_factor)

# enable upscaling
model.return_hr = True

loss_fn = ReconLoss(lambda_sam=cfg.lambda_sam)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=loss_fn,
    cfg=cfg,
)

trainer.fit()
