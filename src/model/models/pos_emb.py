import math
import os

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

# from .base import AVRModule


class PositionalEmbedding(pl.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        out_dim: int,
        nrows: int,
        ncols: int,
        ndim: int | None = None,
        row_wise: bool = True,
        feature_pooling: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.ndim = ndim or nrows + ncols
        self.nrows = nrows
        self.ncols = ncols
        self.row_wise = row_wise
        self.row_column_fc = nn.Linear(self.ndim, out_dim)
        self.feature_pooling = feature_pooling

    def forward(self, x):
        cells = [[0 for _ in range(self.ndim)] for _ in range(self.nrows * self.ncols)]

        it = 0
        if self.row_wise:
            for i in range(self.nrows):
                for j in range(self.ncols):
                    cell = cells[it]
                    cell[i] = 1
                    cell[self.nrows + j] = 1
                    cells[it] = cell
                    it += 1
        else:
            for j in range(self.ncols):
                for i in range(self.nrows):
                    cell = cells[it]
                    cell[i] = 1
                    cell[self.nrows + j] = 1
                    cells[it] = cell
                    it += 1

        cells = [torch.tensor(cell, device=self.device).repeat((x.shape[0], 1)).float() for cell in cells]
        if not self.feature_pooling:
            posemb = [
            self.row_column_fc(cell)
            .unsqueeze(1)
            .repeat((1, x.shape[2], 1))
            .unsqueeze(1)
            for cell in cells
            ]
            posemb_flatten = torch.flatten(torch.cat(posemb, dim=1), start_dim=1, end_dim=2)
        else:
            posemb = [
            self.row_column_fc(cell)
            .unsqueeze(1)
            for cell in cells
            ]
            posemb_flatten = torch.cat(posemb, dim=1)

        return posemb_flatten


if __name__ == "__main__":
    x = torch.randn(3, 5, 4)
    ndim = 3 + 3
    nrows, ncols = 3, 3
    row_wise = True
    # 1 0 0 1 0 0
    # 1 0 0 0 1 0
    # 1 0 0 0 0 1
    # 0 1 0
    #
    cells = [[0 for _ in range(ndim)] for _ in range(nrows * ncols)]
    if row_wise:
        it = 0
        for i in range(nrows):
            for j in range(ncols):
                cell = cells[it]
                cell[i] = 1
                cell[nrows + j] = 1
                cells[it] = cell
                it += 1
    else:
        it = 0
        for j in range(ncols):
            for i in range(nrows):
                cell = cells[it]
                cell[i] = 1
                cell[nrows + j] = 1
                cells[it] = cell
                it += 1

    cells = [torch.tensor(cell).repeat((x.shape[0], 1)).float() for cell in cells]
    posemb = [
        cell.unsqueeze(1).repeat((1, x.shape[2], 1)).unsqueeze(1)
        # self.row_column_fc(cell) # repeat answer.shape[2] ??? check
        for cell in cells
    ]
    posemb_flatten = torch.flatten(torch.cat(posemb, dim=1), start_dim=1, end_dim=2)
    print(posemb_flatten.shape)

    first = torch.tensor([1, 0, 0, 1, 0, 0]).repeat((x.shape[0], 1)).float()

    # given_panels_posencoded=[]
    first = torch.tensor([1, 0, 0, 1, 0, 0]).repeat((x.shape[0], 1)).float()
    second = torch.tensor([1, 0, 0, 0, 1, 0]).repeat((x.shape[0], 1)).float()
    third = torch.tensor([1, 0, 0, 0, 0, 1]).repeat((x.shape[0], 1)).float()
    fourth = torch.tensor([0, 1, 0, 1, 0, 0]).repeat((x.shape[0], 1)).float()
    fifth = torch.tensor([0, 1, 0, 0, 1, 0]).repeat((x.shape[0], 1)).float()
    sixth = torch.tensor([0, 1, 0, 0, 0, 1]).repeat((x.shape[0], 1)).float()
    seventh = torch.tensor([0, 0, 1, 1, 0, 0]).repeat((x.shape[0], 1)).float()
    eighth = torch.tensor([0, 0, 1, 0, 1, 0]).repeat((x.shape[0], 1)).float()
    nineth = torch.tensor([0, 0, 1, 0, 0, 1]).repeat((x.shape[0], 1)).float()

    first_posemb = first.unsqueeze(1).repeat((1, x.shape[2], 1))
    second_posemb = second.unsqueeze(1).repeat((1, x.shape[2], 1))
    third_posemb = third.unsqueeze(1).repeat((1, x.shape[2], 1))
    fourth_posemb = fourth.unsqueeze(1).repeat((1, x.shape[2], 1))
    fifth_posemb = fifth.unsqueeze(1).repeat((1, x.shape[2], 1))
    sixth_posemb = sixth.unsqueeze(1).repeat((1, x.shape[2], 1))
    seventh_posemb = seventh.unsqueeze(1).repeat((1, x.shape[2], 1))
    eighth_posemb = eighth.unsqueeze(1).repeat((1, x.shape[2], 1))
    nineth_posemb = nineth.unsqueeze(1).repeat((1, x.shape[2], 1))

    all_posemb_concat = torch.cat(
        (
            first_posemb.unsqueeze(1),
            second_posemb.unsqueeze(1),
            third_posemb.unsqueeze(1),
            fourth_posemb.unsqueeze(1),
            fifth_posemb.unsqueeze(1),
            sixth_posemb.unsqueeze(1),
            seventh_posemb.unsqueeze(1),
            eighth_posemb.unsqueeze(1),
            nineth_posemb.unsqueeze(1),
        ),
        dim=1,
    )
    all_posemb_concat_flatten = torch.flatten(all_posemb_concat, start_dim=1, end_dim=2)

    print(all_posemb_concat_flatten.shape)
    # print(cells[0].shape)
    # print(first.shape)
    # print(cells[0])
    print(first.shape)

    # for i in range(nrows * ncols):
    #     print(cells[i])
    # print(cells)
