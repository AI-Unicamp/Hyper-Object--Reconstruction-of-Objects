from baselines.raw2hsi import Raw2HSI
from baselines.mstpp_up import MST_Plus_Plus_LateUpsample

from .rev3dcnn import Rev3DCNN
from .rev2dcnn import Rev2DCNN
from .rgb2mosaic import MosaicUp
# to add a model that is in this folder:
# from .file import ModelName
from .example import Example

import torch
from typing import Dict, Any

def setup_model(config: Dict[str, Any]) -> torch.nn.Module:
    model: torch.nn.Module
    match config["model_name"]:
        case "raw2hsi": # track 1 baseline
            model = Raw2HSI(base_ch=config.get("base_ch", 64),
                            n_blocks=config.get("n_blocks", 8),
                            out_bands=config.get("out_bands", 61))

        case "mst_plus_plus": # MST++, no upsample
            model = MST_Plus_Plus_LateUpsample(in_channels=config.get("in_channels", 3),
                                               out_channels=config.get("out_channels", 61),
                                               n_feat=config.get("n_feat", 61),
                                               stage=config.get("stage", 3),
                                               upscale_factor=config.get("upscale_factor", 1))
            model.return_hr = False
            model.force_direct_lr = config.get("force_direct_lr", True)
        case "mst_plus_plus_up": # MST++ with upsample, track 2 baseline
            model = MST_Plus_Plus_LateUpsample(in_channels=config.get("in_channels", 3),
                                               out_channels=config.get("out_channels", 61),
                                               n_feat=config.get("n_feat", 61),
                                               stage=config.get("stage", 3),
                                               upscale_factor=config.get("upscale_factor", 2))
            model.return_hr = True
        case "revsci_rgb":
            # TRevSCI for RGB demosaicing ("mosaic" -> "rgb_full")
            model = Rev3DCNN(n_blocks=config.get("n_blocks", 12),
                             n_split=config.get("n_split", 2),
                             old_mode=config.get("old_mode", False))
            return model

        case "revsci_rgb_up":
            # TRevSCI for RGB demosaicing adapted to track 2 ("rgb_2" -> "rgb_full")
            to_mosaic = MosaicUp(old_mode=config.get("old_mode", False))
            revsci = Rev3DCNN(n_blocks=config.get("n_blocks", 12),
                             n_split=config.get("n_split", 2),
                             old_mode=config.get("old_mode", False))

            model = torch.nn.Sequential(
                    to_mosaic, revsci
                    )
            
            return model

        case "revsci_mstpp":
            # pre-trained TRevSCI (frozen) with MST++
            revsci = Rev3DCNN(n_blocks=config.get("n_blocks", 12),
                             n_split=config.get("n_split", 2),
                             old_mode=config.get("old_mode", False))
            data = torch.load(config["pretrained_revsci_path"], weights_only=True)
            revsci.load_state_dict(data["model"])

            for p in revsci.parameters():
                p.requires_grad_(False)

            mstpp = MST_Plus_Plus_LateUpsample(in_channels=config.get("in_channels", 3),
                                               out_channels=config.get("out_channels", 61),
                                               n_feat=config.get("n_feat", 61),
                                               stage=config.get("stage", 3),
                                               upscale_factor=config.get("upscale_factor", 1))
            mstpp.return_hr = False
            mstpp.force_direct_lr = config.get("force_direct_lr", True)

            model = torch.nn.Sequential(
                    revsci, mstpp
                    )

            return model

        case "revsci_mstpp_no_pretrain":
            revsci = Rev3DCNN(n_blocks=config.get("n_blocks", 12),
                             n_split=config.get("n_split", 2),
                             old_mode=config.get("old_mode", False))

            mstpp = MST_Plus_Plus_LateUpsample(in_channels=config.get("in_channels", 3),
                                               out_channels=config.get("out_channels", 61),
                                               n_feat=config.get("n_feat", 61),
                                               stage=config.get("stage", 3),
                                               upscale_factor=config.get("upscale_factor", 1))
            mstpp.return_hr = False
            mstpp.force_direct_lr = config.get("force_direct_lr", True)

            model = torch.nn.Sequential(
                    revsci, mstpp
                    )

            return model

        case "revsci_mstpp_up":
            # pre-trained TRevSCI (frozen) with MST++, adapted to track 2
            to_mosaic = MosaicUp(old_mode=config.get("old_mode", False))
        
            revsci = Rev3DCNN(n_blocks=config.get("n_blocks", 12),
                             n_split=config.get("n_split", 2),
                             old_mode=config.get("old_mode", False))

            revsci_rgb_up = torch.nn.Sequential(to_mosaic, revsci)
            data = torch.load(config["pretrained_revsci_path"], weights_only=True)
            revsci_rgb_up.load_state_dict(data["model"])

            for p in revsci_rgb_up.parameters():
                p.requires_grad_(False)

            mstpp = MST_Plus_Plus_LateUpsample(in_channels=config.get("in_channels", 3),
                                               out_channels=config.get("out_channels", 61),
                                               n_feat=config.get("n_feat", 61),
                                               stage=config.get("stage", 3),
                                               upscale_factor=config.get("upscale_factor", 1))
            mstpp.return_hr = False
            mstpp.force_direct_lr = config.get("force_direct_lr", True)

            model = torch.nn.Sequential(
                    revsci_rgb_up, mstpp
                    )

            return model

        case "revsci2_rgb":
            # TRevSCI with Conv2D's. Doesn't perform as well.
            model = Rev2DCNN(n_blocks=config.get("n_blocks", 12), n_split=config.get("n_split", 2))
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
