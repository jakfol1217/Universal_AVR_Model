import math
import os
import json
from itertools import permutations


import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
import timm
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from numpy import dot
from numpy.linalg import norm 
from transformers import pipeline

from .base import AVRModule
from .yoloWrapper import YOLOwrapper


class ScoringModelFeatureTransformer(AVRModule):
    def __init__(
        self,
        cfg: DictConfig,
        context_norm: bool,
        num_correct: int,  # number of correct answers, raven - 1, bongard - 2
        in_dim,
        transformer: pl.LightningModule,
        pos_emb: pl.LightningModule | None = None,
        disc_pos_emb: pl.LightningModule | None = None,
        additional_metrics: dict = {},
        save_hyperparameters=True,
        increment_dataloader_idx: int = 0,
        **kwargs,
    ):
        super().__init__(cfg)
        if save_hyperparameters:
            self.save_hyperparameters(
                OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            )

        # self.automatic_optimization = False # use when slot and transoformer have separate optimizers
        self.in_dim = in_dim
        # self.row_column_fc = nn.Linear(6, in_dim)
        multi_corrects = [
            int(_it.removeprefix("num_correct_"))
            for _it in kwargs.keys()
            if _it.startswith("num_correct_")
        ]
        self.num_correct = [num_correct] + [
            kwargs.get(f"num_correct_{_ix}") for _ix in multi_corrects
        ]

        if context_norm: # use context norm
            self.contextnorm = True
            self.gamma = nn.Parameter(torch.ones(in_dim))
            self.beta = nn.Parameter(torch.zeros(in_dim))
        else:
            self.contextnorm = False

        # TODO: Add option to train slot_model as well (may require configuring multiple optimizers)
        multi_pos_emb = [
            int(_it.removeprefix("pos_emb_"))
            for _it in kwargs.keys()
            if _it.startswith("pos_emb_")
        ]
        self.transformer = transformer
        if len(multi_pos_emb) == 0:
            self.pos_emb = nn.ModuleList([pos_emb])
        if len(multi_pos_emb) > 0:
            self.pos_emb = nn.ModuleList(
                [pos_emb] + [kwargs.get(f"pos_emb_{_ix}") for _ix in multi_pos_emb]
            )
        
        multi_disc_pos_emb = [
            int(_it.removeprefix("disc_pos_emb_"))
            for _it in kwargs.keys()
            if _it.startswith("disc_pos_emb_")
        ]
        if len(multi_disc_pos_emb) == 0:
            self.disc_pos_emb = nn.ModuleList([disc_pos_emb])
        if len(multi_disc_pos_emb) > 0:
            self.disc_pos_emb = nn.ModuleList(
                [disc_pos_emb] + [kwargs.get(f"disc_pos_emb_{_ix}") for _ix in multi_disc_pos_emb]
            )
        
        self.use_disc_pos_emb = pos_emb is None and not disc_pos_emb is None

        self.loss = instantiate(cfg.metrics.cross_entropy)
        self.val_losses = []
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

        if len(multi_pos_emb) == 0:
            self.additional_metrics = nn.ModuleList(
                [create_module_dict(additional_metrics)]
            )
        else:
            self.additional_metrics = nn.ModuleList(
                [create_module_dict(additional_metrics)]
                + [
                    create_module_dict(kwargs.get(f"additional_metrics_{_ix}"))
                    for _ix in multi_pos_emb
                ]
            )
        

        self.increment_dataloader_idx = increment_dataloader_idx

    @torch.no_grad()
    def init_detection_model(self):
        # workaround for creating object detection model
        detection_model = YOLOwrapper('/app/yolo/yolov8m.pt')
        detection_model.yolo.eval()
        return detection_model


    def apply_context_norm(self, z_seq):
        eps = 1e-8
        z_mu = z_seq.mean(1)
        z_sigma = (z_seq.var(1) + eps).sqrt()

        z_seq = (z_seq - z_mu.unsqueeze(1)) / z_sigma.unsqueeze(1)
        z_seq = (z_seq * self.gamma.unsqueeze(0).unsqueeze(0)) + self.beta.unsqueeze(
            0
        ).unsqueeze(0)
        return z_seq

    def forward(self, given_panels, answer_panels, idx=0):
        # computing scores with multiple possible transformer models
        __pos_emb = self.pos_emb[idx]
        __transformer = self.transformer
        __num_correct = self.num_correct[idx]
        __disc_pos_emb = self.disc_pos_emb[idx]

        scores = []
        pos_emb_score = (
            __pos_emb(given_panels) if __pos_emb is not None and not self.use_disc_pos_emb else torch.tensor(0.0)
        )

        disc_pos_embed = __disc_pos_emb() if self.use_disc_pos_emb else torch.rand(0)
        if self.use_disc_pos_emb:
            disc_pos_embed = disc_pos_embed.repeat(given_panels.shape[0], 1, 1)



        for d in permutations(range(answer_panels.shape[1]), __num_correct):

            x_seq = torch.cat([given_panels, answer_panels[:, d]], dim=1)

            if self.contextnorm:

                x_seq = self.apply_context_norm(x_seq)
            if self.use_disc_pos_emb:
                print(x_seq.shape)
                print(disc_pos_embed.shape)
                x_seq = torch.cat([x_seq, disc_pos_embed], dim=-1)
            else:
                x_seq = x_seq + pos_emb_score  # TODO: add positional embeddings

            score = __transformer(x_seq)
            scores.append(score)
        scores = torch.cat(scores, dim=1)
        return scores

    # TODO: Separate optimizers for different modules
    def configure_optimizers(self):
        return instantiate(self.cfg.optimizer, params=self.parameters())
 

    def _step(self, step_name, batch, batch_idx, dataloader_idx=0):
        img, target = batch

        context_panels_cnt = self.cfg.data.tasks[ # number of task context panels
            self.task_names[dataloader_idx]
        ].num_context_panels

        given_panels = img[:, :context_panels_cnt] # context panels
        answer_panels = img[:, context_panels_cnt:] # answer panels

        scores = self(given_panels, answer_panels, idx=dataloader_idx + self.increment_dataloader_idx)
        softmax = nn.Softmax(dim=1)
        scores = softmax(scores)

        pred = scores.argmax(1)

        current_metrics = self.additional_metrics[dataloader_idx + self.increment_dataloader_idx] # computing and reporting metrics
        for metric_nm, metric_func in current_metrics.items():
            value = metric_func(pred, target)
            self.log(
                f"{step_name}/{self.task_names[dataloader_idx]}/{metric_nm}",
                value,
                on_epoch=True,
                prog_bar=True if step_name == "val" else False,
                logger=True,
                add_dataloader_idx=False,
            )

        loss = self.loss(scores, target)
        return loss

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self.module_step("train", batch, batch_idx, dataloader_idx)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self.module_step("val", batch, batch_idx, dataloader_idx)
        self.val_losses.append(loss)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self.module_step("test", batch, batch_idx, dataloader_idx)
        return loss

    def on_validation_epoch_end(self) -> None:
        val_losses = torch.tensor(self.val_losses)
        val_loss = val_losses.mean()
        self.log(
            "val/loss",
            val_loss.to(self.device),
            on_epoch=True,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )
        self.val_losses.clear()

    



