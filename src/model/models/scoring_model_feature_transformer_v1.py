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
import spacy 
from numpy import dot
from numpy.linalg import norm 
from transformers import pipeline
from sentence_transformers import SentenceTransformer

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
        use_captions: bool,
        pooling: bool,
        pos_emb: pl.LightningModule | None = None,
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
        
        # defining feature transformer (for embedding), default: vit_large_patch32_384
        if pooling: # use pooling in feature transformer
            self.feature_transformer = timm.create_model(transformer_name, pretrained=True, num_classes=0)
        else:
            self.feature_transformer = timm.create_model(transformer_name, pretrained=True, num_classes=0, global_pool='')
        for param in self.feature_transformer.parameters():
                param.requires_grad = False
        self.pooling = pooling
        
        # optional: detection model for object detection
        self.detection_model = None
        if use_detection:
            self.detection_model = [self.init_detection_model()]
        
        # optional: activity detection model (using captions to find activities as presented on image, e.g. man running etc)
        self.use_captions = use_captions
        self.cos_sim = torch.nn.CosineSimilarity(dim=0)
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

        scores = []
        pos_emb_score = (
            __pos_emb(given_panels) if __pos_emb is not None else torch.tensor(0.0)
        )


        for d in permutations(range(answer_panels.shape[1]), __num_correct):


            x_seq = torch.cat([given_panels, answer_panels[:, d]], dim=1)

            if not self.pooling:     
                x_seq = torch.flatten(x_seq, start_dim=1, end_dim=2)
            if self.contextnorm:

                x_seq = self.apply_context_norm(x_seq)
            x_seq = x_seq + pos_emb_score  # TODO: add positional embeddings

            score = __transformer(x_seq)
            scores.append(score)
        scores = torch.cat(scores, dim=1)
        return scores

    # TODO: Separate optimizers for different modules
    def configure_optimizers(self):
        return instantiate(self.cfg.optimizer, params=self.parameters())
    
    def forward_detection_model(self, given_panels, answer_panels, context_groups, answer_groups):
        # function performing optional object detection step (returns scores that are added to model scores)
        detection_scores = [[] for _ in range(given_panels.shape[0])]

        context_panels = given_panels[:, context_groups[0], :]
        context_panels_flat = context_panels.flatten(0, 1)

        used_answer_panels = answer_panels[:, answer_groups, :]
        used_answer_panels_flat = used_answer_panels.flatten(0, 1)

        context_detected = self.get_detected_classes(context_panels_flat).view((*context_panels.shape[:2], -1))
        answer_detected = self.get_detected_classes(used_answer_panels_flat).view((*used_answer_panels.shape[:2], -1))

        for i in range(context_detected.shape[0]):
            for a_g in answer_groups:
                detection_scores[i].append(self.activity_score_function(context_detected[i,:], answer_detected[i, a_g, :]))
        return torch.Tensor(detection_scores)


    def forward_activities_model(self, given_panels, answer_panels, context_groups, answer_groups):
        # function performing optional activity detection step (returns scores that are added to model scores)
        activity_scores = [[] for _ in range(given_panels.shape[0])]

        context_panels = given_panels[:, context_groups[0], :]

        used_answer_panels = answer_panels[:, answer_groups, :]

        
        for i in range(context_panels.shape[0]):
            for a_g in answer_groups:
                activity_scores[i].append(self.activity_score_function(context_panels[i,:], used_answer_panels[i, a_g, :]))

        return torch.Tensor(activity_scores)
 

    def _step(self, step_name, batch, batch_idx, dataloader_idx=0):
        if self.use_captions: # if activity detection is used, we load in additional pre-defined captions for images
            img, target, img_cap = batch
        else:
            img, target = batch
        results = []
        for idx in range(img.shape[1]): # creating embeddings with feature transformer
            results.append(self.feature_transformer(img[:, idx]))

        context_panels_cnt = self.cfg.data.tasks[ # number of task context panels
            self.task_names[dataloader_idx]
        ].num_context_panels

        given_panels = torch.stack(results, dim=1)[:, :context_panels_cnt] # context panels
        answer_panels = torch.stack(results, dim=1)[:, context_panels_cnt:] # answer panels

        given_imgs = img[:, :context_panels_cnt] # context original images (pre embedding)
        answer_imgs = img[:, context_panels_cnt:] # answer original images (pre-embedding)

        context_groups = self.cfg.data.tasks[ # context groups (e.g. using only 1 group from bongard instead of all context images)
                self.task_names[dataloader_idx]
            ].context_groups
        
        answer_groups = self.cfg.data.tasks[ # answer groups, contain all answers in case of bongard and analogy making problems
                self.task_names[dataloader_idx]
            ].answer_groups

        scores = self(given_panels, answer_panels, idx=dataloader_idx + self.increment_dataloader_idx)
        softmax = nn.Softmax(dim=1)
        scores = softmax(scores)

        if self.detection_model is not None:
            # optional object detection step
            det_scores = self.forward_detection_model(given_imgs, answer_imgs, context_groups, answer_groups)
            det_scores = det_scores.to(scores, non_blocking=True)
            
            scores_adjusted = scores + det_scores
            scores = scores_adjusted

        if self.use_captions:
            # optional activity detection step
            given_imgs_cap = img_cap[:, :context_panels_cnt]
            answer_imgs_cap = img_cap[:, context_panels_cnt:]

            act_scores = self.forward_activities_model(given_imgs_cap, answer_imgs_cap, context_groups, answer_groups)
            act_scores = act_scores.to(scores, non_blocking=True)

            scores_adjusted = scores + act_scores
            scores = scores_adjusted   
            del scores_adjusted, act_scores, given_imgs_cap, answer_imgs_cap 
        del given_imgs, answer_imgs

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

    def on_save_checkpoint(self, checkpoint):
        # deleting saved weights of non-trained models to save EDEN disk space
        keys_to_delete = []
        for key in checkpoint['state_dict']:
            if key.startswith('slot_model') or key.startswith('feature_transformer') or key.startswith('detection_model'):
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del checkpoint['state_dict'][key]



    def get_detected_classes(self, images, confidence_level=0.8):
        # function computing the number of detected objects in object detection step
        results = self.detection_model[0](images)
        classes = np.array([[0 for _ in range(len(results[0].names))] for i in range(len(results))], dtype='float64')
        for i, r in enumerate(results):
            json_res = json.loads(r.tojson())
            for detection in json_res:
                if detection['confidence'] >= confidence_level:
                    classes[i][detection['class']] += 1
        return torch.from_numpy(classes)

    def detection_score_function(self, context, answers):
        # function computing object detection scores (which are added to model scores to influence model decision)
        x = np.sum(np.average(context, axis=0) - answers)
        context_scale = np.sum(np.average(context, axis=0))
        
        res = max(-(abs(x)**(3/2))/90 + 0.15, -0.2)
        if res < 0:
            numbing_param = 2/context_scale if context_scale != 0 else 1
            res *= numbing_param
        return res
    

    def activity_score_function(self, ac1_em, ac2_em):
        # function computing activity detection scores (which are added to model scores to influence model decision)
        cos_sim = 0
        for i in range(ac1_em.shape[0]):
            cos_sim += self.cos_sim(ac1_em[i,:], ac2_em)
        cos_sim /= ac1_em.shape[0]
        return cos_sim/2 - 0.12
    

