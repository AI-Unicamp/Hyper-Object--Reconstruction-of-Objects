from baselines.raw2hsi import Raw2HSI
from baselines.mstpp_up import MST_Plus_Plus_LateUpsample

# to add a model that is in this folder:
# from . import ModelName

from torch.nn import Module
from typing import Dict, Any

def setup_model(config: Dict[str, Any]) -> Module:
    model: Module
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

        # NOTE: to add a new model:
        # case "[model_name]":
        #     model = Model(parameter1=config.get("parameter1", default_value1),
        #                   parameter2=config.get("parameter2", default_value2),
        #                   ...)
        # use parameters that you have added to the model's config file

        case _:
            raise ValueError(f"'{config["model_name"]}'' is not a valid model")

    return model
