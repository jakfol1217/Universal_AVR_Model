from itertools import permutations

import pytorch_lightning as pl
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.checkpoint import checkpoint

from .scoring_model_v1 import ScoringModel
from .WReN_average import WReN_average
from .WReN_each import WReN_each
from .WReN_in_order import WReN_in_order


class ScoringModelWReN(ScoringModel):
    def __init__(
        self,
        cfg: DictConfig,
        context_norm: bool,
        object_size: int,
        slot_model: pl.LightningModule,
        wren_type: str,
        additional_metrics: dict = {},
        save_hyperparameters=True,
        freeze_slot_model=True,
        **kwargs,
    ):
        super().__init__(
            cfg, 
            context_norm=context_norm, 
            num_correct=1, 
            in_dim=-1, 
            slot_model=slot_model,
            transformer=None, 
            pos_emb=None, 
            additional_metrics=additional_metrics,
            save_hyperparameters=save_hyperparameters, 
            freeze_slot_model=freeze_slot_model
                         )
        self.wren_type = wren_type
        if self.wren_type == "averaged":
            self.wren_model = WReN_average(
                object_size=object_size,
                use_layer_norm=context_norm
            )
        elif self.wren_type == "order":
            self.wren_model = WReN_in_order(
                object_size=object_size,
                use_layer_norm=context_norm
            )
        elif self.wren_type == "each":
            self.wren_model = WReN_each(
                object_size=object_size,
                use_layer_norm=context_norm
            )


       

    def forward(self, given_panels, answer_panels):
        # TODO: implement with slot_model (image as input instead of slots)
        if self.wren_type == "averaged":
            given_panels = given_panels.mean(2)
            answer_panels = answer_panels.mean(2)
        scores = self.wren_model(given_panels.squeeze(), answer_panels.squeeze())
        return scores


