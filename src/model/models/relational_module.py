import pytorch_lightning as pl
import torch
import torch.nn as nn


class RelationalModule(pl.LightningModule):
    def __init__(self,
                 object_size: int,
                 asymetrical: bool,
                 rel_activation_func: str = "softmax",
                 context_norm: bool = False,
                 hierarchical: bool = False):
        super(RelationalModule, self).__init__()
        if asymetrical:
            self.k_trans = nn.Linear(object_size, object_size)
            self.q_trans = nn.Linear(object_size, object_size)
        else:
            self.k_trans = nn.Identity()
            self.q_trans = nn.Identity()
        
        
        if rel_activation_func == "softmax":
            self.rel_activation_func = nn.Softmax(dim=2)
        elif rel_activation_func == "tanh":
            self.rel_activation_func = nn.Tanh(dim=2)
        else:
            self.rel_activation_func = nn.Identity() # todo: more activation functions?

        self.context_norm = context_norm # todo where to put this? probably at the start also what if multiple problems
        
        if hierarchical:
            self.create_relational = self.relational_bottleneck_hierarchical
        else:
            self.create_relational = self.relational_bottleneck

        self.hierarchical_aggregation = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1) # creates weighted sum with learnable parameters
        


    def forward(self, context: torch.Tensor, answers: torch.Tensor) -> torch.Tensor:
        k_context = self.k_trans(context)
        q_context = self.q_trans(context)

        k_answers = self.k_trans(answers)
        q_answers = self.q_trans(answers)

        relational_matrices = []
        for ans_i in range(answers.shape[1]):
            keys = torch.cat([k_context, k_answers[:, ans_i, :].unsqueeze(1)], dim=1)
            queries = torch.cat([q_context, q_answers[:, ans_i, :].unsqueeze(1)], dim=1)
            rel_matrix_1, rel_matrix_2 = self.create_relational(keys, queries)
            rel_matrix = torch.cat([rel_matrix_1.unsqueeze(1), rel_matrix_2.unsqueeze(1)], dim=1)
            rel_matrix = self.hierarchical_aggregation(rel_matrix)
            relational_matrices.append(rel_matrix)

        return torch.cat(relational_matrices, dim=1)


# todo: potentially other module for abstract shapes? utilizing slots etc

    def relational_bottleneck(self, keys, queries):
        rel_matrix = torch.matmul(keys, queries.transpose(1,2))
        return self.rel_activation_func(rel_matrix), torch.zeros(*rel_matrix.shape)
    
    def relational_bottleneck_hierarchical(self, keys, queries):
        rel_matrix_1 = torch.matmul(keys, queries.transpose(1,2)) # use activation on previous if hierarchical? probably not
        rel_matrix_2 = torch.matmul(rel_matrix_1, rel_matrix_1.transpose(1,2))
        return self.rel_activation_func(rel_matrix_1), self.rel_activation_func(rel_matrix_2)





class RelationalScoringModule(pl.LightningModule):
    def __init__(self,
                 in_dim:int,
                 hidden_dim:int=256,
                 pooling: str = "max"):
        super(RelationalScoringModule, self).__init__()
        self.scoring_mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
        )
        if pooling == "max":
            self.pooling = nn.AdaptiveMaxPool2d((1, in_dim))
        elif pooling == "avg":
            self.pooling = nn.AdaptiveAvgPool2d((1, in_dim))
        else:
            raise ValueError(f"Pooling type {pooling} not supported")
        
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, rel_matrix: torch.Tensor) -> torch.Tensor:
        rel_matrix = rel_matrix.flatten(-2).unsqueeze(-2)
        rel_matrix = self.pooling(rel_matrix).squeeze()
        answer_scores = []
        for ans_i in range(rel_matrix.shape[1]):
            answer_scores.append(self.scoring_mlp(rel_matrix[:, ans_i, :]))
        answer_scores = torch.cat(answer_scores, dim=1)
        return self.softmax(answer_scores)