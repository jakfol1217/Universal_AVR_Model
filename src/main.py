import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from lightning.pytorch.loggers import WandbLogger

from src.model.models.STSN import SlotAttentionAutoEncoder
from wandb_agent import WandbAgent


# from config import register_resolvers
from model.avr_datasets import IRAVENdataset


# from model.data_modules.common_modules import CombinedModuleSequential

# register_resolvers()


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def _test(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision(cfg.torch.matmul_precision)  # 'medium' | 'high'
    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision

    data_module = instantiate(cfg.data.datamodule, cfg)

    # TODO: checkpoint mechanism (param in config + loading from checkpoint)
    # TODO: datamodules (combination investiagtion)

    wandb_logger = WandbLogger(project="AVR_universal", log_model="all")
    module = instantiate(cfg.model, cfg)
    # print(module)
    # print(cfg.trainer)
    trainer: pl.Trainer = instantiate(cfg.trainer)
    trainer.logger = wandb_logger
    wandb_logger.watch(module)
    trainer.fit(module, data_module)
    #trainer.test(module, data_module)

    # example loading best model from newest run
    wandb_agent = WandbAgent("AVR_universal")
    checkpoint_path = wandb_agent.get_newest_checkpoint()
    new_model = SlotAttentionAutoEncoder.load_from_checkpoint(checkpoint_path)
    print(new_model)


if __name__ == "__main__":
    _test()
