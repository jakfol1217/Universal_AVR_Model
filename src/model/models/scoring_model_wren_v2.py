from itertools import permutations
import json

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import autocast
import timm 
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.checkpoint import checkpoint

from .scoring_model_v1 import ScoringModel
from .WReN_average import WReN_average
from .WReN_each import WReN_each
from .WReN_in_order import WReN_in_order
from .WReN_vit import WReN_vit
from .VASR_model import VASR_model
from .yoloWrapper import YOLOwrapper


class ScoringModelWReN(ScoringModel):
    def __init__(
        self,
        cfg: DictConfig,
        context_norm: bool,
        in_dim: int,
        slot_model: pl.LightningModule,
        wren_type: str,
        hidden_dim: int = 512,
        additional_metrics: dict = {},
        save_hyperparameters=True,
        freeze_slot_model=True,
        g_depth=3,
        f_depth=2,
        use_caption_linear: bool = False,
        num_correct: int = 2,
        increment_dataloader_idx: int = 0,
        **kwargs,
    ):
        super().__init__(
            cfg, 
            context_norm=context_norm, 
            num_correct=num_correct, 
            in_dim=in_dim, 
            slot_model=slot_model,
            transformer=None, 
            pos_emb=None, 
            additional_metrics=additional_metrics,
            save_hyperparameters=save_hyperparameters, 
            freeze_slot_model=freeze_slot_model
                         )
        # choosing the type of WReN model (affects how STSN slots are processed, e.g. if we take the average of all slots or slots unchanged)
        self.wren_type = wren_type
        if self.wren_type == "averaged" or self.wren_type == "vit_pooled":
            self.wren_model = WReN_average(
                object_size=in_dim,
                use_layer_norm=context_norm,
                g_depth=g_depth,
                f_depth=f_depth
            )
        elif self.wren_type == "vit":
            self.wren_model = WReN_vit(
                object_size=in_dim,
                use_layer_norm=context_norm,
                hidden_size=hidden_dim
            )
        elif self.wren_type == "order":
            self.wren_model = WReN_in_order(
                object_size=in_dim,
                use_layer_norm=context_norm
            )
        elif self.wren_type == "each":
            self.wren_model = WReN_each(
                object_size=in_dim,
                use_layer_norm=context_norm
            )
        elif self.wren_type == "basic":
            self.wren_model = VASR_model(
                object_size=in_dim
            )

        # defining feature transformer (for embedding), default: vit_large_patch32_384
        
        task_metrics_idxs = [ # loading additional metrics for different tasks from configuration files
            int(_it.removeprefix("task_metric_"))
            for _it in kwargs.keys()
            if _it.startswith("task_metric_")
        ]

        multi_corrects = [
            int(_it.removeprefix("num_correct_"))
            for _it in kwargs.keys()
            if _it.startswith("num_correct_")
        ]
        self.num_correct = [num_correct] + [
            kwargs.get(f"num_correct_{_ix}") for _ix in multi_corrects
        ]

        def create_module_dict(metrics_dict):
            return nn.ModuleDict(
                {
                    metric_nm: (
                        instantiate(metric_func)
                        if isinstance(metric_func, DictConfig)
                        else metric_func
                    )
                    for metric_nm, metric_func in metrics_dict.items()
                }
            )

        self.task_metrics = nn.ModuleList( # loading additional metrics for different tasks
            [
                create_module_dict(kwargs.get(f"task_metric_{_ix}"))
                for _ix in task_metrics_idxs
            ]
        )

        if len(self.task_metrics) > 0:
            self.additional_metrics = self.task_metrics

        # optional: activity detection model (using captions to find activities as presented on image, e.g. man running etc)    

        self.increment_dataloader_idx = increment_dataloader_idx



    @torch.no_grad()
    def init_detection_model(self):
        # workaround for creating object detection model
        detection_model = YOLOwrapper('/app/yolo/yolov8m.pt')
        detection_model.yolo.eval()
        return detection_model
            


    def forward(self, given_panels, answer_panels, idx=0):
        __num_correct = self.num_correct[idx]
        __disc_pos_emb = self.disc_pos_emb[idx]
        # creating scores with the use of WReN model
        if self.wren_type == "averaged":
            given_panels = given_panels.mean(2)
            answer_panels = answer_panels.mean(2)

        if self.use_disc_pos_emb:
            disc_pos_embed = __disc_pos_emb()
            disc_pos_embed = disc_pos_embed.repeat(given_panels.shape[0], 1, 1)
            disc_pos_embed_context = disc_pos_embed[:, :-__num_correct, :]
            disc_pos_embed_answers = disc_pos_embed[:, -__num_correct:, :].repeat(1,__num_correct,1)
            given_panels = torch.cat([given_panels, disc_pos_embed_context], dim=-1)
            answer_panels = torch.cat([answer_panels, disc_pos_embed_answers], dim=-1)

        scores = self.wren_model(given_panels, answer_panels)
        return scores
    
       
    

    def _step(self, step_name, batch, batch_idx, dataloader_idx=0):

        img, target = batch
        slot_model_loss = None

        
        context_panels_cnt = self.cfg.data.tasks[ # number of task context panels
                self.task_names[dataloader_idx]
            ].num_context_panels

        given_panels = img[:, :context_panels_cnt] # context panels
        answer_panels = img[:, context_panels_cnt:] # answer panels

        
        scores = self(given_panels, answer_panels, idx=dataloader_idx + self.increment_dataloader_idx)


        pred = scores.argmax(1)

        ce_loss = self.loss(scores, target)

        current_metrics = self.additional_metrics[dataloader_idx] # computing and reporting metrics
        for metric_nm, metric_func in current_metrics.items():
            value = metric_func(pred, target)
            self.log(
                f"{step_name}/{self.task_names[dataloader_idx]}/{metric_nm}",
                value,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                add_dataloader_idx=False,
            )
        if self.auxiliary_loss_ratio > 0:
            self.log(
                f"{step_name}/{self.task_names[dataloader_idx]}/mse_loss",
                slot_model_loss,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                add_dataloader_idx=False,
            )
            self.log(
                f"{step_name}/{self.task_names[dataloader_idx]}/cross_entropy_loss",
                ce_loss,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                add_dataloader_idx=False,
            )

        if slot_model_loss is not None:
            loss = ce_loss + self.auxiliary_loss_ratio * slot_model_loss
        else:
            loss = ce_loss
        return loss

