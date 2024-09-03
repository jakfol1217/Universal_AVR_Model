from itertools import permutations


import numpy as np
import pytorch_lightning as pl
import torch
import timm 
import torch.nn as nn
from torch import autocast
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn

from .scoring_model_v1 import ScoringModel
from .relational_module import RelationalScoringModule, relationalModelConstructor


class CombinedModel(ScoringModel):

    def __init__(
            self,
            cfg: DictConfig,
            context_norm: bool,
            slot_model: pl.LightningModule,
            #slot_model_v3: pl.LightningModule,
            transformer_name: str,
            use_answers_only: bool,
            relational_in_dim: int,
            relational_asymetrical: bool,
            relational_activation_func: str,
            relational_context_norm: bool,
            relational_hierarchical: bool,
            separate_relationals: bool,
            scoring_in_dim: int,
            scoring_hidden_dim: int = 256,
            scoring_pooling_type: str = "max",
            real_idxes: list = [0],
            relational_in_dim_2: int = 1280,
            relational_asymetrical_2: bool = None,
            relational_activation_func_2: str = None,
            relational_context_norm_2: bool = None,
            relational_hierarchical_2: bool = None,
            additional_metrics: dict = {},
            save_hyperparameters=True,
            freeze_slot_model=True,
            auxiliary_loss_ratio: float = 0.0,
            **kwargs,
    ):
        
        super().__init__(cfg,
            context_norm=context_norm, 
            num_correct=1, 
            in_dim=relational_in_dim, 
            slot_model=slot_model,
            transformer=None, 
            pos_emb=None, 
            additional_metrics=additional_metrics,
            save_hyperparameters=save_hyperparameters, 
            freeze_slot_model=freeze_slot_model,
            auxiliary_loss_ratio=auxiliary_loss_ratio)
        
        self.relationalScoringModule = RelationalScoringModule(
            in_dim=scoring_in_dim,
            hidden_dim=scoring_hidden_dim,
            pooling=scoring_pooling_type
        )

        self.relationalModule_real = relationalModelConstructor(
            use_answers_only=use_answers_only,
            object_size=relational_in_dim,
            asymetrical=relational_asymetrical,
            rel_activation_func=relational_activation_func,
            context_norm=relational_context_norm,
            hierarchical=relational_hierarchical
        )
        if separate_relationals:
            self.pooling = nn.AdaptiveMaxPool2d((1, relational_in_dim_2))
            self.relationalModule_abstract = relationalModelConstructor(
                use_answers_only=use_answers_only,
                object_size=relational_in_dim_2,
                asymetrical=relational_asymetrical_2,
                rel_activation_func=relational_activation_func_2,
                context_norm=relational_context_norm_2,
                hierarchical=relational_hierarchical_2 
            )
        else:
            self.pooling = nn.AdaptiveMaxPool2d((1, relational_in_dim))
            self.relationalModule_abstract = self.relationalModule_real

        self.feature_transformer= timm.create_model(transformer_name, pretrained=True, num_classes=0)
        self.real_idxes = real_idxes

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
        


        #self.slot_model_v3 = slot_model_v3
        #if (
        #    slot_ckpt_path := cfg.model.slot_model_v3.ckpt_path
        #) is not None and cfg.checkpoint_path is None:
        #    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        #    model_cfg = {
        #        k: v
        #        for k, v in cfg_dict["model"]["slot_model_v3"].items()
        #        if k != "_target_"
        #    }
        #    self.slot_model_v3 = slot_model_v3.__class__.load_from_checkpoint(
        #        slot_ckpt_path, cfg=cfg, **model_cfg
        #    )
        #if self.freeze_slot_model:
        #    self.slot_model_v3.freeze()
        #else:
        #    self.slot_model_v3.unfreeze()
#
    def is_task_abstract(self, image): # todo: how to detect if task real or abstract? for now it's hard-coded
        return image not in self.real_idxes

    def forward(self, given_panels, answer_panels, isAbstract):
        if not isAbstract:
            rel_matrix = self.relationalModule_real(given_panels, answer_panels)
        else:
            given_panels_rel = given_panels.flatten(-2).unsqueeze(-2)
            answer_panels_rel = answer_panels.flatten(-2).unsqueeze(-2)
            given_panels_rel = self.pooling(given_panels_rel).squeeze(-2)
            answer_panels_rel = self.pooling(answer_panels_rel).squeeze(-2)
            rel_matrix = self.relationalModule_abstract(given_panels_rel, answer_panels_rel)
        
        scores = self.relationalScoringModule(rel_matrix)
        return scores

        
    def _step(self, step_name, batch, batch_idx, dataloader_idx=0):
        img, target = batch
        
        results = []
        isAbstract = self.is_task_abstract(dataloader_idx) # pass img, for now its hard coded

        slot_model_loss = None
        if isAbstract:
            recon_combined_seq = []
            if img.shape[2] > 1:
                slot_model = self.slot_model
            else:
                slot_model = self.slot_model
            for idx in range(img.shape[1]):
                recon_combined, recons, masks, slots, attn = slot_model(img[:, idx])
                results.append(slots)
                recon_combined_seq.append(recon_combined)
                del recon_combined, recons, masks, slots, attn
            pred_img = torch.stack(recon_combined_seq, dim=1).contiguous()
            if pred_img.shape[2] != img.shape[2]:
                pred_img = pred_img.repeat(1, 1, 3, 1, 1)
            slot_model_loss = slot_model.loss(pred_img, img)

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

        scores = self(given_panels, answer_panels, isAbstract=isAbstract)
        softmax = nn.Softmax(dim=1)
        scores = softmax(scores)

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

        if isAbstract and slot_model_loss is not None:
            loss = ce_loss + self.auxiliary_loss_ratio * slot_model_loss
        else:
            loss = ce_loss
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
