import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


class Sequential(pl.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        models: list[nn.Module],
    ):
        super().__init__()
        _models = []

        for model in models:
            if isinstance(model, (dict, DictConfig)):
                module=instantiate(DictConfig(model))
            else:
                module=model
            _models.append(module)
        self.model = nn.Sequential(*_models)

    def forward(self, x):
        return self.model(x)
