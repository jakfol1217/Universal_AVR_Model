import pytorch_lightning as pl
import torch
import torch.nn as nn

from .RN_helper_funtions import *
NUM_CANDIDATE = 4

class VASR_model(pl.LightningModule):

    def __init__(self, object_size: int):

        super(VASR_model, self).__init__()

        self.object_size = object_size
        pair_embed_dim = 4 * self.object_size
        self.pairs_layer = nn.Sequential(
            nn.LayerNorm(pair_embed_dim),
            nn.Linear(pair_embed_dim, pair_embed_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(NUM_CANDIDATE * pair_embed_dim, 384),
            nn.ReLU(),
            nn.Linear(384, NUM_CANDIDATE)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, context: torch.Tensor, answers: torch.Tensor) -> torch.Tensor:

        pairs = []
        for i in range(answers.shape[1]):
            input_option_pair = torch.cat([context.flatten(1), answers[:,i]], dim=1)
            input_option_pair = self.pairs_layer(input_option_pair)
            pairs.append(input_option_pair)
        x = torch.cat(pairs, dim=1)

        x = self.classifier(x)
        return x
    

