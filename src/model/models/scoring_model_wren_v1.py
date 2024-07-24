from itertools import permutations
import json

from ultralytics import YOLO

import pytorch_lightning as pl
import torch
import torch.nn as nn
import timm 
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.checkpoint import checkpoint

from .scoring_model_v1 import ScoringModel
from .WReN_average import WReN_average
from .WReN_each import WReN_each
from .WReN_in_order import WReN_in_order
from .WReN_vit import WReN_vit
from .yoloWrapper import YOLOwrapper


class ScoringModelWReN(ScoringModel):
    def __init__(
        self,
        cfg: DictConfig,
        context_norm: bool,
        in_dim: int,
        slot_model: pl.LightningModule,
        wren_type: str,
        transformer_name: str,
        use_detection: bool,
        hidden_dim: int = 512,
        additional_metrics: dict = {},
        save_hyperparameters=True,
        freeze_slot_model=True,
        **kwargs,
    ):
        super().__init__(
            cfg, 
            context_norm=context_norm, 
            num_correct=1, 
            in_dim=in_dim, 
            slot_model=slot_model,
            transformer=None, 
            pos_emb=None, 
            additional_metrics=additional_metrics,
            save_hyperparameters=save_hyperparameters, 
            freeze_slot_model=freeze_slot_model
                         )
        
        self.wren_type = wren_type
        if self.wren_type == "averaged" or self.wren_type == "vit_pooled":
            self.wren_model = WReN_average(
                object_size=in_dim,
                use_layer_norm=context_norm
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
        self.feature_transformer = None
        if self.wren_type == "vit":
            self.feature_transformer = timm.create_model(transformer_name, pretrained=True, num_classes=0, global_pool='')
        if self.wren_type == "vit_pooled":
            self.feature_transformer= timm.create_model(transformer_name, pretrained=True, num_classes=0)
        
        if self.feature_transformer is not None:
            for param in self.feature_transformer.parameters():
                param.requires_grad = False

        self.detection_model = None
        if use_detection:
            self.detection_model = [self.init_detection_model()]

    @torch.no_grad()
    def init_detection_model(self):
        detection_model = YOLOwrapper('yolo/yolov8m.pt')
        detection_model.yolo.eval()
        return detection_model
            


    def forward(self, given_panels, answer_panels):
        if self.wren_type == "averaged":
            given_panels = given_panels.mean(2)
            answer_panels = answer_panels.mean(2)
        scores = self.wren_model(given_panels, answer_panels)
        return scores
    
    def forward_detection_model(self, given_panels, answer_panels, context_groups, answer_groups):
        detection_scores = [[] for _ in range(given_panels.shape[0])]
        for i in range(given_panels.shape[0]):
            for c_g in context_groups:
                context_detected = self.get_detected_classes(given_panels[i, c_g, :])
                for a_g in answer_groups:
                    answer_detected = self.get_detected_classes(answer_groups[i, a_g, :])
                    detection_scores[i].append(self.score_function(context_detected, answer_detected))
        return torch.Tensor(detection_scores)
                    
    

    def _step(self, step_name, batch, batch_idx, dataloader_idx=0):
        img, target = batch
        results = []
        slot_model_loss = None
        if self.feature_transformer is None:
            for idx in range(img.shape[1]):
                recon_combined, recons, masks, slots, attn = self.slot_model(img[:, idx])
                results.append(slots)
                del recon_combined, recons, masks, slots, attn
            pred_img = torch.stack(results, dim=1).contiguous()
            if pred_img.shape[2] != img.shape[2]:
                pred_img = pred_img.repeat(1, 1, 3, 1, 1)
            slot_model_loss = self.slot_model.loss(pred_img, img)

        else:
            for idx in range(img.shape[1]):
                results.append(self.feature_transformer(img[:, idx]))
        
        context_panels_cnt = self.cfg.data.tasks[
                self.task_names[dataloader_idx]
            ].num_context_panels

        given_panels = torch.stack(results, dim=1)[:, :context_panels_cnt]
        answer_panels = torch.stack(results, dim=1)[:, context_panels_cnt:]

        
        scores = self(given_panels, answer_panels)

        if self.detection_model is not None:
            context_groups = self.cfg.data.tasks[
                self.task_names[dataloader_idx]
            ].context_groups
            answer_groups = self.cfg.data.tasks[
                self.task_names[dataloader_idx]
            ].answer_groups
        
            scores += self.forward_detection_model(given_panels, answer_panels, context_groups, answer_groups)
            

        pred = scores.argmax(1)

        ce_loss = self.loss(scores, target)

        current_metrics = self.additional_metrics[dataloader_idx]
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

    def on_save_checkpoint(self, checkpoint):
        keys_to_delete = []
        for key in checkpoint['state_dict']:
            if key.startswith('slot_model') or key.startswith('feature_transformer') or key.startswith('detection_model'):
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del checkpoint['state_dict'][key]


    def get_detected_classes(self, images, confidence_level=0.8):
        results = self.detection_model[0](images)
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
