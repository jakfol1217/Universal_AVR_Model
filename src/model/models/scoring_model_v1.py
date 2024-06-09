import math
import os
from itertools import permutations

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.checkpoint import checkpoint

from .base import AVRModule


class ScoringModel(AVRModule):
    def __init__(
        self,
        cfg: DictConfig,
        context_norm: bool,
        num_correct: int,  # number of correct answers, raven - 1, bongard - 2
        in_dim,
        slot_model: pl.LightningModule,
        transformer: pl.LightningModule,
        pos_emb: pl.LightningModule | None = None,
        additional_metrics: dict = {},
        save_hyperparameters=True,
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
        self.num_correct = num_correct

        if context_norm:
            self.contextnorm = True
            self.gamma = nn.Parameter(torch.ones(in_dim))
            self.beta = nn.Parameter(torch.zeros(in_dim))
        else:
            self.contextnorm = False

        slot_model.freeze()  # TODO: Add option to train slot_model as well (may require configuring multiple optimizers)
        self.slot_model = slot_model
        self.transformer = transformer
        self.pos_emb = pos_emb

        self.loss = instantiate(cfg.metrics.cross_entropy)
        self.val_losses = []
        self.additional_metrics = nn.ModuleDict(
            {
                metric_nm: (
                    instantiate(metric_func)
                    if isinstance(metric_func, DictConfig)
                    else metric_func
                )
                for metric_nm, metric_func in additional_metrics.items()
            }
        )

    # TODO: overwrite load_from_checkpoint to load slot_model and transformer (and pos_emb if exists)
    # TODO: does it work GPU vs CPU?
    def apply_context_norm(self, z_seq):
        eps = 1e-8
        z_mu = z_seq.mean(1)
        z_sigma = (z_seq.var(1) + eps).sqrt()
        # print("seq, mean, var shape>>",z_seq.shape,z_mu.shape,z_sigma.shape)
        z_seq = (z_seq - z_mu.unsqueeze(1)) / z_sigma.unsqueeze(1)
        z_seq = (z_seq * self.gamma.unsqueeze(0).unsqueeze(0)) + self.beta.unsqueeze(
            0
        ).unsqueeze(0)
        return z_seq

    def forward(self, given_panels, answer_panels):
        # TODO: implement with slot_model (image as input instead of slots)

        scores = []
        pos_emb_score = (
            self.pos_emb(given_panels)
            if self.pos_emb is not None
            else torch.tensor(0.0)
        )
        # Loop through all choices and compute scores
        for d in permutations(range(answer_panels.shape[1]), self.num_correct):

            # print(AB,C_choices[:,d,:],AB.shape,C_choices[:,d,:].shape)
            # x_seq = torch.cat([given_panels_posencoded_seq,torch.cat((answer_panels[:,d],self.row_fc(third).unsqueeze(1).repeat((1,answer_panels.shape[2],1)), self.column_fc(third).unsqueeze(1).repeat((1,answer_panels.shape[2],1))),dim=2).unsqueeze(1)],dim=1)
            # print(given_panels.shape)
            # print(answer_panels[:, d].shape)
            x_seq = torch.cat([given_panels, answer_panels[:, d]], dim=1)
            # print("seq min and max>>",torch.min(x_seq),torch.max(x_seq))
            # x_seq = torch.cat([AB,C_choices[:,d,:].unsqueeze(1)],dim=1)
            x_seq = torch.flatten(x_seq, start_dim=1, end_dim=2)
            if self.contextnorm:

                x_seq = self.apply_context_norm(x_seq)

            x_seq = x_seq + pos_emb_score  # TODO: add positional embeddings
            # x_seq = torch.cat((x_seq,all_posemb_concat_flatten),dim=2)
            score = self.transformer(x_seq)
            scores.append(score)
        scores = torch.cat(scores, dim=1)
        return scores

    # TODO: Separate optimizers for different modules
    # def configure_optimizers(self):
    #     return instantiate(self.cfg.optimizer, params=self.parameters())

    def _step(self, step_name, batch, batch_idx, dataloader_idx=0):
        img, target = batch
        recon_combined_seq = []
        recons_seq = []
        masks_seq = []
        slots_seq = []
        for idx in range(img.shape[1]):
            recon_combined, recons, masks, slots, attn = self.slot_model(img[:, idx])
            recons_seq.append(recons)
            recon_combined_seq.append(recon_combined)
            masks_seq.append(masks)
            slots_seq.append(slots)
            del recon_combined, recons, masks, slots, attn
        pred_img = torch.stack(recon_combined_seq, dim=1).contiguous()
        if pred_img.shape[2] != img.shape[2]:
            pred_img = pred_img.repeat(1, 1, 3, 1, 1)
        slot_model_loss = self.slot_model.loss(pred_img, img)

        context_panels_cnt = self.cfg.data.tasks[
            self.task_names[dataloader_idx]
        ].num_context_panels

        given_panels = torch.stack(slots_seq, dim=1)[:, :context_panels_cnt]
        answer_panels = torch.stack(slots_seq, dim=1)[:, context_panels_cnt:]

        scores = self(given_panels, answer_panels)
        # print("scores and target>>",scores,target)
        pred = scores.argmax(1)
        # print(scores)
        # print(scores.shape) # batch x num_choices
        # print(f"Prediction: {pred}, Target: {target}")

        for metric_nm, metric_func in self.additional_metrics.items():
            value = metric_func(pred, target)
            self.log(
                f"{step_name}/{self.task_names[dataloader_idx]}/{metric_nm}",
                value,
                on_epoch=True,
                prog_bar=True if step_name == "val" else False,
                logger=True,
                add_dataloader_idx=False,
            )
        # acc = torch.eq(pred,target).float().mean().item() * 100.0

        # print("mse loss>>>",mse_criterion(torch.stack(recon_combined_seq,dim=1).squeeze(4), img))
        # print("ce loss>>",ce_criterion(scores,target))
        # print("recon combined seq shape>>",torch.stack(recon_combined_seq,dim=1).shape)
        # loss = 1000*mse_criterion(torch.stack(recon_combined_seq,dim=1), img) + ce_criterion(scores,target)
        # loss = ce_criterion(scores,target)
        # TODO: which loss
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
