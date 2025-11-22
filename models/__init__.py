from baselines.raw2hsi import Raw2HSI
from baselines.mstpp_up import MST_Plus_Plus_LateUpsample

from .rev3dcnn import Rev3DCNN
# to add a model that is in this folder:
# from .file import ModelName
from .example import Example

import torch
from typing import Dict, Any

def setup_model(config: Dict[str, Any]) -> torch.nn.Module:
    model: torch.nn.Module
    match config["model_name"]:
        case "raw2hsi":
            model = Raw2HSI(base_ch=config.get("base_ch", 64),
                            n_blocks=config.get("n_blocks", 8),
                            out_bands=config.get("out_bands", 61))

        case "mst_plus_plus":
            model = MST_Plus_Plus_LateUpsample(in_channels=config.get("in_channels", 3),
                                               out_channels=config.get("out_channels", 61),
                                               n_feat=config.get("n_feat", 61),
                                               stage=config.get("stage", 3),
                                               upscale_factor=config.get("upscale_factor", 1))
            model.return_hr = False
            model.force_direct_lr = config.get("force_direct_lr", True)
        case "mst_plus_plus_up":
            model = MST_Plus_Plus_LateUpsample(in_channels=config.get("in_channels", 3),
                                               out_channels=config.get("out_channels", 61),
                                               n_feat=config.get("n_feat", 61),
                                               stage=config.get("stage", 3),
                                               upscale_factor=config.get("upscale_factor", 2))
            model.return_hr = True
        case "revsci_mstpp":
            demosaic = Rev3DCNN(n_blocks=config.get("n_blocks", 12), n_split=config.get("n_split", 2))

            mstpp = MST_Plus_Plus_LateUpsample(in_channels=config.get("in_channels", 3),
                                               out_channels=config.get("out_channels", 61),
                                               n_feat=config.get("n_feat", 61),
                                               stage=config.get("stage", 3),
                                               upscale_factor=config.get("upscale_factor", 1))
            mstpp.return_hr = False
            mstpp.force_direct_lr = config.get("force_direct_lr", True)

            model = torch.nn.Sequential(
                demosaic, mstpp
                )
        case "revsci_mstpp_up":
            demosaic = Rev3DCNN(n_blocks=config.get("n_blocks", 12), n_split=config.get("n_split", 2))

            mstpp = MST_Plus_Plus_LateUpsample(in_channels=config.get("in_channels", 3),
                                               out_channels=config.get("out_channels", 61),
                                               n_feat=config.get("n_feat", 61),
                                               stage=config.get("stage", 3),
                                               upscale_factor=config.get("upscale_factor", 1))
            mstpp.return_hr = True

            model = torch.nn.Sequential(
                demosaic, mstpp
                )
        case "revsci_rgb":
            model = Rev3DCNN(n_blocks=config.get("n_blocks", 12), n_split=config.get("n_split", 2))
            return model

        # to add a new model:
        case "example":
            model = Example(param1=config.get("param1", ...),
                            param2=config.get("param2", ...))

            # if using pre-trained data, do something like
            pretrained = torch.load(config.get("pretrained_path", ...), weights_only=True)
            model.load_state_dict(pretrained)

            # and so on

        case _:
            raise ValueError(f"'{config['model_name']}'' is not a valid model")

    return model
