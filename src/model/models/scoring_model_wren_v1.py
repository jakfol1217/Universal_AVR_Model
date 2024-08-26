from itertools import permutations
import json

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import autocast
import timm 
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.checkpoint import checkpoint
import spacy 
from numpy import dot
from numpy.linalg import norm 
from transformers import pipeline
from sentence_transformers import SentenceTransformer

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
        transformer_name: str,
        use_detection: bool,
        use_captions: bool,
        hidden_dim: int = 512,
        additional_metrics: dict = {},
        save_hyperparameters=True,
        freeze_slot_model=True,
        g_depth=3,
        f_depth=2,
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
        self.feature_transformer = None
        if self.wren_type == "vit":
            self.feature_transformer = timm.create_model(transformer_name, pretrained=True, num_classes=0, global_pool='')
        if self.wren_type == "vit_pooled" or self.wren_type == "basic":
            self.feature_transformer= timm.create_model(transformer_name, pretrained=True, num_classes=0)
        
        if self.feature_transformer is not None:
            for param in self.feature_transformer.parameters():
                param.requires_grad = False

        self.detection_model = None
        if use_detection:
            self.detection_model = [self.init_detection_model()]
        
        task_metrics_idxs = [
            int(_it.removeprefix("task_metric_"))
            for _it in kwargs.keys()
            if _it.startswith("task_metric_")
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

        self.task_metrics = nn.ModuleList(
            [
                create_module_dict(kwargs.get(f"task_metric_{_ix}"))
                for _ix in task_metrics_idxs
            ]
        )

        if len(self.task_metrics) > 0:
            self.additional_metrics = self.task_metrics

            
        self.use_captions = use_captions
        self.cos_sim = torch.nn.CosineSimilarity(dim=0)



    @torch.no_grad()
    def init_detection_model(self):
        detection_model = YOLOwrapper('/app/yolo/yolov8m.pt')
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
        activity_scores = [[] for _ in range(given_panels.shape[0])]

        context_panels = given_panels[:, context_groups[0], :]

        used_answer_panels = answer_panels[:, answer_groups, :]

        
        for i in range(context_panels.shape[0]):
            for a_g in answer_groups:
                activity_scores[i].append(self.activity_score_function(context_panels[i,:], used_answer_panels[i, a_g, :]))

        return torch.Tensor(activity_scores) 
       
    

    def _step(self, step_name, batch, batch_idx, dataloader_idx=0):
        if self.use_captions:
            img, target, img_cap = batch
        else:
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
                with autocast(device_type='cuda', dtype=torch.float16):
                    res = self.feature_transformer(img[:, idx])
                results.append(res)
        
        context_panels_cnt = self.cfg.data.tasks[
                self.task_names[dataloader_idx]
            ].num_context_panels

        given_panels = torch.stack(results, dim=1)[:, :context_panels_cnt]
        answer_panels = torch.stack(results, dim=1)[:, context_panels_cnt:]

        given_imgs = img[:, :context_panels_cnt]
        answer_imgs = img[:, context_panels_cnt:]
        context_groups = self.cfg.data.tasks[
                self.task_names[dataloader_idx]
            ].context_groups
        answer_groups = self.cfg.data.tasks[
                self.task_names[dataloader_idx]
            ].answer_groups
        
        scores = self(given_panels, answer_panels)

        if self.detection_model is not None:
        
            det_scores = self.forward_detection_model(given_imgs, answer_imgs, context_groups, answer_groups)
            det_scores = det_scores.to(scores, non_blocking=True)

            scores_adjusted = scores + det_scores
            scores = scores_adjusted
        
        if self.use_captions:
            given_imgs_cap = img_cap[:, :context_panels_cnt]
            answer_imgs_cap = img_cap[:, context_panels_cnt:]

            act_scores = self.forward_activities_model(given_imgs_cap, answer_imgs_cap, context_groups, answer_groups)
            act_scores = act_scores.to(scores, non_blocking=True)

            scores_adjusted = scores + act_scores
            scores = scores_adjusted

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
        classes = np.array([[0 for _ in range(len(results[0].names))] for i in range(len(results))], dtype='float64')
        for i, r in enumerate(results):
            json_res = json.loads(r.tojson())
            for detection in json_res:
                if detection['confidence'] >= confidence_level:
                    classes[i][detection['class']] += 1
        return torch.from_numpy(classes)

    def detection_score_function(self, context, answers):
        x = np.sum(np.average(context, axis=0) - answers)
        context_scale = np.sum(np.average(context, axis=0))
        
        res = max(-(abs(x)**(3/2))/90 + 0.15, -0.2)
        if res < 0:
            numbing_param = 2/context_scale if context_scale != 0 else 1
            res *= numbing_param
        return res
    
    def activity_score_function(self, ac1_em, ac2_em):
        cos_sim = 0
        for i in range(ac1_em.shape[0]):
            cos_sim += self.cos_sim(ac1_em[i,:], ac2_em)
        cos_sim /= ac1_em.shape[0]
        return cos_sim/2 - 0.12
