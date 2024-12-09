import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from hydra.utils import instantiate

from .scoring_model_feature_transformer_v1 import ScoringModelFeatureTransformer

class BaselineScoringModel(ScoringModelFeatureTransformer):
    def __init__(self, 
                cfg: DictConfig,
                context_norm: bool,
                base_type: str, 
                base_in_dim: int, 
                base_out_dim: int, 
                base_hid_dim: int, 
                additional_metrics: dict = {},
                *args, 
                **kwargs):
        super().__init__(cfg=cfg, context_norm=context_norm, additional_metrics=additional_metrics, 
                         *args, **kwargs)

        self.base_type = base_type
        if base_type == "mlp":
            self.scoring = MLPscoringModule(base_in_dim, 1, base_hid_dim)
        if base_type == "avg_pool":
            self.scoring = PoolingScoringModule(base_out_dim, "avg")
        if base_type == "max_pool":
            self.scoring = PoolingScoringModule(base_out_dim, "max")

        self.pooling = nn.AdaptiveMaxPool2d((1, base_in_dim))

        self.dataloader_idx = kwargs.get("dataloader_idx")

        multi_metric = [
            int(_it.removeprefix("additional_metrics_"))
            for _it in kwargs.keys()
            if _it.startswith("additional_metrics_")
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

        if len(multi_metric) == 0:
            self.additional_metrics = nn.ModuleList(
                [create_module_dict(additional_metrics)]
            )
        else:
            self.additional_metrics = nn.ModuleList(
                [create_module_dict(additional_metrics)]
                + [
                    create_module_dict(kwargs.get(f"additional_metrics_{_ix}"))
                    for _ix in multi_metric
                ]
            )

    def forward(self, given_panels, answer_panels, idx=0):


        answer_scores = []
        for ans_i in range(answer_panels.shape[1]):
            x_seq = torch.cat([given_panels, answer_panels[:,ans_i].unsqueeze(1)], dim=1)

            if self.contextnorm:
                x_seq = self.apply_context_norm(x_seq)

            if self.base_type == "mlp":     
                x_seq = torch.flatten(x_seq, start_dim=1, end_dim=2)
                x_seq = x_seq.unsqueeze(-2)
                x_seq = self.pooling(x_seq).squeeze(-2)
                answer_scores.append(self.scoring(x_seq))
                 
        if self.base_type == "mlp": 
            scores = torch.cat(answer_scores, dim=1)
        else:
            scores = self.scoring(x_seq)

        return scores
    
    def _step(self, step_name, batch, batch_idx, dataloader_idx=0):
        img, target = batch

        print(self.task_names)

        context_panels_cnt = self.cfg.data.tasks[ # number of task context panels
            self.task_names[dataloader_idx]
        ].num_context_panels

        given_panels = img[:, :context_panels_cnt] # context panels
        answer_panels = img[:, context_panels_cnt:] # answer panels


        scores = self(given_panels, answer_panels, idx=dataloader_idx)
        softmax = nn.Softmax(dim=1)
        scores = softmax(scores)

        pred = scores.argmax(1)

        current_metrics = self.additional_metrics[self.dataloader_idx] # computing and reporting metrics
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
    



class MLPscoringModule(pl.LightningModule):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is not None:
            self.mlp_module = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim)
            )
        else:
            self.mlp_module = nn.Sequential(
                nn.Linear(in_dim, out_dim)
            )
    
    def forward(self, panels):
        return self.mlp_module(panels)

class PoolingScoringModule(pl.LightningModule):
    def __init__(self, out_dim: int, pool_type: str = "max"):
        super().__init__()
        if pool_type == "max":
            self.pooling = nn.AdaptiveMaxPool2d((1, out_dim))
        elif pool_type == "avg":
            self.pooling = nn.AdaptiveAvgPool2d((1, out_dim))

    def forward(self, panels):
        return self.pooling(panels).squeeze(1)