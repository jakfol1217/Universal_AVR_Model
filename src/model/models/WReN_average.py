import pytorch_lightning as pl
import torch
import torch.nn as nn

from .RN_helper_funtions import *

class WReN_average(pl.LightningModule):
    """
    Wild Relation Network (WReN) [1] for solving Raven's Progressive Matrices.
    The originally proposed model uses a Relation Network (RN) [2] which works on object pairs.
    This version compares panels that have been previously averaged.

    [1] Santoro, Adam, et al. "Measuring abstract reasoning in neural networks." ICML 2018
    [2] Santoro, Adam, et al. "A simple neural network module for relational reasoning." NeurIPS 2017
    """

    def __init__(self, object_size: int, use_layer_norm: bool = False, g_depth=3, f_depth=2):
        """
         Initializes the WReN model.
         :param object_size: size of the object (panel) vector
         :param use_layer_norm: flag indicating whether layer normalization should be applied after
         the G submodule of RN.
        """
        super(WReN_average, self).__init__()

        self.group_objects = GroupImagesIntoPairs()
        self.group_objects_with = GroupImagesIntoPairsWith()

        self.object_size = object_size
        self.object_tuple_size = (2) * (self.object_size)
        self.g = G(
            depth=g_depth,
            in_size=self.object_tuple_size,
            out_size=self.object_tuple_size,
            use_layer_norm=False
        )
        self.norm = nn.LayerNorm(self.object_tuple_size) if use_layer_norm else Identity()
        self.f = F(
            depth=f_depth,
            object_size=self.object_tuple_size,
            out_size=1
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, context: torch.Tensor, answers: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the WReN model.
        :param context: a tensor with shape (batch_size, num_context_images, size_context). num_context_images vary between tasks,
        e.g. in case of the Bongard task it is 12.
        :param answers: a tensor with shape (batch_size, num_answer_images, size_answers). num_answer_images vary between tasks,
        e.g. in case of the Bongard task it is 2. size_answers and batch_size is the same as in the case of context parameters.
        :return: a tensor with shape (batch_size, num_answers). num_answers is dependent on the task given,
        e.g. for Bongard problems it is 2.
        """
        batch_context_size, num_context_images, size_context = context.size()
        batch_answers_size, num_answer_images, size_answers = answers.size()
        pair = self.group_objects(context, context)
        context_g_out = self.g(pair)
        del pair
        f_out = torch.zeros(batch_answers_size, num_answer_images, device=context.device).type_as(context)
        for i in range(num_answer_images):
            context_choice_pairs = self.group_objects_with(context, answers[:, i, :])
            context_choice_g_out = self.g(context_choice_pairs)
            relations = context_g_out + context_choice_g_out
            relations = self.norm(relations)
            f_out[:, i] = self.f(relations).squeeze()
        return f_out


