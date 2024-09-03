import pytorch_lightning as pl
import torch
import torch.nn as nn


class RelationalModule(pl.LightningModule):
    def __init__(self,
                 object_size: int,
                 asymetrical: bool,
                 rel_activation_func: str = "softmax",
                 context_norm: bool = True,
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
            self.rel_activation_func = nn.Tanh()
        else:
            self.rel_activation_func = nn.Identity() # todo: more activation functions?

        self.context_norm = context_norm # todo where to put this? probably at the start also what if multiple problems

        if hierarchical:
            self.create_relational = self.relational_bottleneck_hierarchical
            self.hierarchical_aggregation = nn.Parameter(data=torch.rand(2, requires_grad=True))  # creates weighted sum with learnable parameters
        else:
            self.create_relational = self.relational_bottleneck
            self.hierarchical_aggregation = nn.Parameter(data=torch.tensor([1,0], dtype=torch.float32, requires_grad=False))


        self.gamma = nn.Parameter(torch.ones(object_size))
        self.beta = nn.Parameter(torch.zeros(object_size))

    def apply_context_norm(self, z_seq):
        eps = 1e-6
        z_mu = z_seq.mean(1)
        z_sigma = (z_seq.var(1) + eps).sqrt()
        z_seq = (z_seq - z_mu.unsqueeze(1)) / z_sigma.unsqueeze(1)
        z_seq = (z_seq * self.gamma.unsqueeze(0).unsqueeze(0)) + self.beta.unsqueeze(
            0
        ).unsqueeze(0)
        return z_seq


    def forward(self, context: torch.Tensor, answers: torch.Tensor) -> torch.Tensor:

        relational_matrices = []
        for ans_i in range(answers.shape[1]):
            context_choice = torch.cat([context, answers[:, ans_i, :].unsqueeze(1)], dim=1)
            if self.context_norm:
                context_choice = self.apply_context_norm(context_choice)

            keys = self.k_trans(context_choice)

            queries = self.q_trans(context_choice)

            rel_matrix_1, rel_matrix_2 = self.create_relational(keys, queries)
            rel_matrix = torch.cat([rel_matrix_1.unsqueeze(1), rel_matrix_2.unsqueeze(1)], dim=1)
            rel_matrix = torch.einsum('btch,m->bch', rel_matrix, self.hierarchical_aggregation)
            relational_matrices.append(rel_matrix.unsqueeze(1))

        return torch.cat(relational_matrices, dim=1)


# todo: potentially other module for abstract shapes? utilizing slots etc

    def relational_bottleneck(self, keys, queries):

        rel_matrix = torch.matmul(keys, queries.transpose(1,2))
        diag = torch.zeros(rel_matrix.shape[1], rel_matrix.shape[2], device=rel_matrix.device)
        diag.fill_diagonal_(torch.inf)
        rel_matrix = rel_matrix - diag
        return self.rel_activation_func(rel_matrix), torch.zeros(*rel_matrix.shape, device=rel_matrix.device)

    def relational_bottleneck_hierarchical(self, keys, queries):

        rel_matrix_1 = torch.matmul(keys, queries.transpose(1,2))
        diag = torch.zeros(rel_matrix_1.shape[1], rel_matrix_1.shape[2], device=rel_matrix_1.device)
        diag.fill_diagonal_(torch.inf)
        rel_matrix_1 = rel_matrix_1 - diag
        rel_matrix_1 = self.rel_activation_func(rel_matrix_1)  # use activation on previous if hierarchical? probably not
        rel_matrix_2 = torch.matmul(rel_matrix_1, rel_matrix_1.transpose(1,2))
        return rel_matrix_1, self.rel_activation_func(rel_matrix_2)



class RelationalModuleAnswersOnly(RelationalModule):
    def __init__(self,
                 object_size: int,
                 asymetrical: bool,
                 rel_activation_func: str = "softmax",
                 context_norm: bool = True,
                 hierarchical: bool = False):
        super(RelationalModuleAnswersOnly, self).__init__(object_size,
                                                          asymetrical,
                                                          rel_activation_func,
                                                          context_norm,
                                                          hierarchical)

        self.asymetrical = asymetrical


    def forward(self, context: torch.Tensor, answers: torch.Tensor) -> torch.Tensor:

        relational_matrices = []
        for ans_i in range(answers.shape[1]):
            context_choice = torch.cat([context, answers[:, ans_i, :].unsqueeze(1)], dim=1)
            if self.context_norm:
                context_choice = self.apply_context_norm(context_choice)
            keys = self.k_trans(context_choice)

            queries = self.q_trans(context_choice)

            if torch.any(keys.isnan()) or torch.any(keys.isinf()) or torch.any(queries.isnan()) or torch.any(queries.isinf()):
                print("after normalization")

            rel_matrix_1, rel_matrix_2 = self.create_relational(keys[:,:-1], queries[:,:-1], keys[:,-1].unsqueeze(1), queries[:,-1].unsqueeze(1))
            if not self.asymetrical:
                rel_matrix_1 = rel_matrix_1[:, 0].unsqueeze(1)
                rel_matrix_2 = rel_matrix_2[:, 0].unsqueeze(1)

            rel_matrix = torch.cat([rel_matrix_1.unsqueeze(1), rel_matrix_2.unsqueeze(1)], dim=1)
            rel_matrix = torch.einsum('btch,m->bch', rel_matrix, self.hierarchical_aggregation)
            relational_matrices.append(rel_matrix.unsqueeze(1))

        return torch.cat(relational_matrices, dim=1)


    def relational_bottleneck(self, context_keyes, context_queries, answers_keys, answers_queries):

        rel_answers_queries = torch.matmul(context_keyes, answers_queries.transpose(1,2)).squeeze()
        rel_answers_keys = torch.matmul(context_queries, answers_keys.transpose(1,2)).squeeze()
        rel_answers = torch.cat([rel_answers_queries.unsqueeze(1), rel_answers_keys.unsqueeze(1)], dim=1)
        return self.rel_activation_func(rel_answers), torch.zeros(*rel_answers.shape, device=rel_answers.device)
    

    def relational_bottleneck_hierarchical(self, context_keyes, context_queries, answers_keys, answers_queries):

        rel_answers_1, _ = self.relational_bottleneck(context_keyes, context_queries, answers_keys, answers_queries)  # use activation on previous if hierarchical?
        rel_context_1 = self.rel_activation_func(torch.matmul(context_keyes, context_queries.transpose(1,2)))
        rel_answers_2 = torch.matmul(rel_context_1, rel_answers_1.transpose(1,2))
        return rel_answers_1, self.rel_activation_func(rel_answers_2.reshape(*rel_answers_1.shape))


def relationalModelConstructor(use_answers_only,
                 object_size: int,
                 asymetrical: bool,
                 rel_activation_func: str = "softmax",
                 context_norm: bool = True,
                 hierarchical: bool = False)-> RelationalModule:
    if use_answers_only:
        return RelationalModuleAnswersOnly(object_size,
                                            asymetrical,
                                            rel_activation_func,
                                            context_norm,
                                            hierarchical)
    else:
        return RelationalModule(object_size,
                                    asymetrical,
                                    rel_activation_func,
                                    context_norm,
                                    hierarchical)



class RelationalScoringModule(pl.LightningModule):
    def __init__(self,
                 in_dim:int,
                 hidden_dim: int = 256,
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
        rel_matrix = self.pooling(rel_matrix).squeeze(-2)
        answer_scores = []
        for ans_i in range(rel_matrix.shape[1]):
            answer_scores.append(self.scoring_mlp(rel_matrix[:, ans_i]))
        answer_scores = torch.cat(answer_scores, dim=1)
        return self.softmax(answer_scores)
