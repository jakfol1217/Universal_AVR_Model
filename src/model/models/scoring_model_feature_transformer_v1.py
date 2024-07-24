import math
import os
import json
from itertools import permutations

from ultralytics import YOLO


import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import timm
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn

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
        transformer_name: str,
        use_detection: bool,
        pooling: bool,
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

        # TODO: Add option to train slot_model as well (may require configuring multiple optimizers)

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
        if pooling:
            self.feature_transformer = timm.create_model(transformer_name, pretrained=True, num_classes=0)
        else:
            self.feature_transformer = timm.create_model(transformer_name, pretrained=True, num_classes=0, global_pool='')
        for param in self.feature_transformer.parameters():
                param.requires_grad = False
        self.pooling = pooling

        self.detection_model = None
        if use_detection:
            self.detection_model = self.init_detection_model()

    @torch.no_grad()
    def init_detection_model(self):
        detection_model = YOLOwrapper('yolo/yolov8m.pt')
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

    def forward(self, given_panels, answer_panels):

        scores = []
        pos_emb_score = (
            self.pos_emb(given_panels)
            if self.pos_emb is not None
            else torch.tensor(0.0)
        )

        for d in permutations(range(answer_panels.shape[1]), self.num_correct):


            x_seq = torch.cat([given_panels, answer_panels[:, d]], dim=1)

            if not self.pooling:     
                x_seq = torch.flatten(x_seq, start_dim=1, end_dim=2)
            if self.contextnorm:

                x_seq = self.apply_context_norm(x_seq)
            x_seq = x_seq + pos_emb_score  # TODO: add positional embeddings

            score = self.transformer(x_seq)
            scores.append(score)
        scores = torch.cat(scores, dim=1)
        return scores

    # TODO: Separate optimizers for different modules
    def configure_optimizers(self):
        return instantiate(self.cfg.optimizer, params=self.parameters())
    
    def forward_detection_model(self, given_panels, answer_panels, context_groups, answer_groups):
        detection_scores = [[] for _ in range(given_panels.shape[0])]
        for i in range(given_panels.shape[0]):
            for c_g in context_groups:
                chosen_given = given_panels[i, c_g, :]
                if len(chosen_given.shape) < 4:
                    chosen_given = chosen_given.unsqueeze(0)
                context_detected = self.get_detected_classes(chosen_given)
                for a_g in answer_groups:
                    chosen_answer = answer_panels[i, a_g, :]
                    if len(chosen_answer) < 4:
                        chosen_answer = chosen_answer.unsqueeze(0)
                    answer_detected = self.get_detected_classes(chosen_answer)
                    detection_scores[i].append(self.score_function(context_detected, answer_detected))
        return torch.Tensor(detection_scores)

    def _step(self, step_name, batch, batch_idx, dataloader_idx=0):
        img, target = batch
        results = []
        for idx in range(img.shape[1]):
            results.append(self.feature_transformer(img[:, idx]))

        context_panels_cnt = self.cfg.data.tasks[
            self.task_names[dataloader_idx]
        ].num_context_panels

        given_panels = torch.stack(results, dim=1)[:, :context_panels_cnt]
        answer_panels = torch.stack(results, dim=1)[:, context_panels_cnt:]

        given_imgs = img[:, :context_panels_cnt]
        answer_imgs = img[:, context_panels_cnt:]

        scores = self(given_panels, answer_panels)

        if self.detection_model is not None:
            context_groups = self.cfg.data.tasks[
                self.task_names[dataloader_idx]
            ].context_groups
            answer_groups = self.cfg.data.tasks[
                self.task_names[dataloader_idx]
            ].answer_groups
            det_scores = self.forward_detection_model(given_imgs, answer_imgs, context_groups, answer_groups)
            det_scores = det_scores.to(scores, non_blocking=True)
            scores += det_scores

        pred = scores.argmax(1)


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

    def on_save_checkpoint(self, checkpoint):
        keys_to_delete = []
        for key in checkpoint['state_dict']:
            if key.startswith('slot_model') or key.startswith('feature_transformer') or key.startswith('detection_model'):
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del checkpoint['state_dict'][key]



    def get_detected_classes(self, images, confidence_level=0.8):
        results = self.detection_model(images)
        classes = np.array([0 for _ in range(len(results[0].names))], dtype='float64')
        for r in results:
            json_res = json.loads(r.tojson())
            for detection in json_res:
                if detection['confidence'] >= confidence_level:
                    classes[detection['class']] += 1
        classes /= len(results)
        return classes

    def score_function(self, context, answers):
        x = np.sum(context - answers)
        context_scale = np.sum(context)
        
        res = max(-(abs(x)**(3/2))/100 + 0.1, -0.15)
        if res < 0:
            numbing_param = 2/context_scale if context_scale != 0 else 1
            res *= numbing_param
        return res