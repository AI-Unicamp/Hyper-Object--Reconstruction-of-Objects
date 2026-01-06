import wandb
import os

class ToolsWandb:

    @staticmethod
    def config_flatten(config, fconfig):
        for key in config:
            if isinstance(config[key], dict):
                fconfig = ToolsWandb.config_flatten(config[key], fconfig)
            else:
                fconfig[key] = config[key]
        return fconfig

    @staticmethod
    def init_wandb_run(f_configurations,
                       run_name,
                       name_project="your-project-name",
                       reinit=True,
                       notes="Testing wandb implementation",
                       entity="YourEntityName"):
        project = name_project or os.getenv("WANDB_PROJECT")
        entity = entity or os.getenv("WANDB_ENTITY")

        return wandb.init(project=project,
                          name=run_name,
                          reinit=reinit,
                          config=f_configurations,
                          notes=notes,
                          entity=entity)
