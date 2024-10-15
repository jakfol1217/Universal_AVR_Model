import io
import math
import os
from itertools import permutations
from typing import OrderedDict

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.checkpoint import checkpoint

from model.models import STSN, ESNBv2, STSNv3

from .base import AVRModule

# import sys

# print(f">>>pl::{pl.__version__}")
# print(f">>>torch::{torch.__version__}")
# sys.setrecursionlimit(50000)

class ScoringModelEsnb(AVRModule):
    def __init__(
        self,
        cfg: DictConfig,
        encoders: list[pl.LightningDataModule],
        relation_module: pl.LightningModule,
        scoring_module: pl.LightningModule,
        save_hyperparameters=True,
        auxiliary_loss_ratio: float = 0.0,
        increment_dataloader_idx: int = 0,
        **kwargs,
    ):
        super().__init__(cfg)

        if save_hyperparameters:
            self.save_hyperparameters(
                OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            )

        _encoders = []
        cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True) # loading encoder models
        for _i in range(len(encoders)):
            conf = cfg.model.encoders[_i]
            if conf is None:
                module=None
            elif (
                ckpt_path := conf.ckpt_path
            ) is not None and cfg.checkpoint_path is None:
                _cfg_dict = cfg_dict["model"]["encoders"][_i]
                model_cfg = {
                    k: v
                    for k, v in _cfg_dict.items()
                    if k != "_target_"
                }

                module_class = get_class(conf._target_)
                module = module_class.load_from_checkpoint(
                    ckpt_path, cfg=cfg, **model_cfg
                )
            else:
                if isinstance(encoders[_i], DictConfig):
                    _args = [cfg] if encoders[_i].get("_target_") != "timm.create_model" else []
                    module=instantiate(encoders[_i], *_args)
                else:
                    module=encoders[_i]

            if conf and conf.get('freeze', True):
                for param in module.parameters():
                    param.requires_grad = False
            _encoders.append(module)

        self.encoders = nn.ModuleList(_encoders)
        self.relation_module =  relation_module
        self.scoring_module = scoring_module

        self.auxiliary_loss_ratio = auxiliary_loss_ratio

        self.loss = instantiate(cfg.metrics.cross_entropy)
        self.val_losses = []
        self.increment_dataloader_idx = increment_dataloader_idx

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


        task_metrics_idxs = sorted([ # loading additional metrics for different tasks from configuration files
            int(_it.removeprefix("task_metric_"))
            for _it in kwargs.keys()
            if _it.startswith("task_metric_")
        ])
        self.task_metrics = nn.ModuleList( # loading additional metrics for different tasks
            [
                create_module_dict(kwargs.get(f"task_metric_{_ix}"))
                for _ix in task_metrics_idxs
            ]
        )

        if len(self.task_metrics) > 0:
            self.additional_metrics = self.task_metrics

    def apply_context_norm(self, z_seq):
        eps = 1e-8
        z_mu = z_seq.mean(1)
        z_sigma = (z_seq.var(1) + eps).sqrt()
        z_seq = (z_seq - z_mu.unsqueeze(1)) / z_sigma.unsqueeze(1)
        z_seq = (z_seq * self.gamma.unsqueeze(0).unsqueeze(0)) + self.beta.unsqueeze(
            0
        ).unsqueeze(0)
        return z_seq

    def forward(self, relations, n_answers=None):
        if isinstance(self.relation_module, ESNBv2.ESNB):
            scores = []
            relations = torch.stack(relations, dim=1).squeeze(2)
            # print(f"{relations.shape=}")
            relations_view = relations.view(relations.shape[0], n_answers, -1, relations.shape[-1]) # add different strategies?
            # print(f"{relations_view.shape=}")
            for i in range(n_answers):
                scores.append(self.scoring_module(relations_view[:, i]))
            return torch.cat(scores, dim=1)
        else:
            return self.scoring_module(relations)

    # TODO: Separate optimizers for different modules
    def configure_optimizers(self):
        return instantiate(self.cfg.optimizer, params=self.parameters())

    def _step(self, step_name, batch, batch_idx, dataloader_idx=0):
        img, target = batch
        encoder = self.encoders[dataloader_idx + self.increment_dataloader_idx]
        results = []

        if encoder is None:
            encoder_model_loss = 0.0
            panels = img
        elif isinstance(encoder, (STSN.SlotAttentionAutoEncoder, STSNv3.SlotAttentionAutoEncoder)): # creating slots using STSN
            recon_combined_seq = []
            recons_seq = []
            masks_seq = []
            for idx in range(img.shape[1]):
                recon_combined, recons, masks, slots, attn = encoder(img[:, idx])
                recons_seq.append(recons)
                recon_combined_seq.append(recon_combined)
                masks_seq.append(masks)
                results.append(slots)
                del recon_combined, recons, masks, slots, attn
            pred_img = torch.stack(recon_combined_seq, dim=1).contiguous()
            if pred_img.shape[2] != img.shape[2]:
                pred_img = pred_img.repeat(1, 1, 3, 1, 1)
            encoder_model_loss = encoder.loss(pred_img, img)
            panels = torch.stack(results, dim=1)
        else:
            for idx in range(img.shape[1]): # creating embeddings with feature transformer
            #     with torch.autocast(device_type='cuda', dtype=torch.float16):
                res = encoder(img[:, idx])
                results.append(res)
            encoder_model_loss = 0.0
            panels = torch.stack(results, dim=1)


        relations = self.relation_module(panels) # computing relations
        # print(f"{relations=}")
        num_context_panels=self.cfg.data.tasks[self.task_names[dataloader_idx]].num_context_panels
        scores = self(relations, n_answers=img.shape[1]-num_context_panels)

        pred = scores.argmax(1)
        # print(scores)
        # print(scores.shape) # batch x num_choices
        # print(f"Prediction: {pred}, Target: {target}")
        ce_loss = self.loss(scores, target) # cross entropy loss for slot image reconstruction

        current_metrics = self.additional_metrics[dataloader_idx + self.increment_dataloader_idx] # computing and reporting metrics
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
                encoder_model_loss,
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

        loss = ce_loss + self.auxiliary_loss_ratio * encoder_model_loss
        # print(f"{encoder_model_loss=}")
        # print(f"{loss=}")
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
        val_loss = val_losses.nanmean()
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
