import pytorch_lightning as pl
import torch
import torch.nn as nn

from .RN_helper_funtions import *

class WReN_in_order(pl.LightningModule):
    """
    Wild Relation Network (WReN) [1] for solving Raven's Progressive Matrices.
    The originally proposed model uses a Relation Network (RN) [2] which works on object pairs.
    This version compares panels in order (1 vs 1, 2 vs 2 etc.)

    [1] Santoro, Adam, et al. "Measuring abstract reasoning in neural networks." ICML 2018
    [2] Santoro, Adam, et al. "A simple neural network module for relational reasoning." NeurIPS 2017
    """

    def __init__(self, object_size: int, use_layer_norm: bool = False):
        """
        Initializes the WReN model.
        :param object_size: size of the object (panel) vector
        :param use_layer_norm: flag indicating whether layer normalization should be applied after
        the G submodule of RN.
        """
        super(WReN_in_order, self).__init__()

        self.group_objects = GroupImagesIntoPairsWithPanels()
        self.group_objects_with = GroupImagesIntoPairsWithWithPanels()

        self.object_size = object_size
        self.object_tuple_size = (2) * (self.object_size)
        self.g = G(
            depth=3,
            in_size=self.object_tuple_size,
            out_size=self.object_tuple_size,
            use_layer_norm=False
        )
        self.norm = nn.LayerNorm(self.object_tuple_size) if use_layer_norm else Identity()
        self.f = F(
            depth=2,
            object_size=self.object_tuple_size,
            out_size=1
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, context: torch.Tensor, answers: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the WReN model.
        :param x: a tensor with shape (batch_size, num_panels, height, width). num_panels is assumed
        to be 16, as the model was designed to solve RPMs from the PGM dataset.
        :return: a tensor with shape (batch_size, num_answers). num_answers is always equal to 8,
        which is the number of answers for each RPM in PGM.
        """
        batch_context_size, num_context_images, num_context_panels, height_context, width_context = context.size()
        batch_answers_size, num_answer_images, num_answers_panels, height_answers, width_answers = answers.size()
        context = context.flatten(3, -1)
        answers = answers.flatten(3, -1)
        pair = self.group_objects(context, context)
        context_g_out = self.g(pair)

        del pair
        f_out = torch.zeros(batch_answers_size, num_answer_images, device=context.device).type_as(context)
        for i in range(num_answer_images):
            context_choice_pairs = self.group_objects_with(context, answers[:, i, :, :])
            context_choice_g_out = self.g(context_choice_pairs)
            del context_choice_pairs
            relations = context_g_out + context_choice_g_out
            relations = relations / num_context_panels
            relations = self.norm(relations)
            f_out[:, i] = self.f(relations).squeeze()
            del relations
        return self.softmax(f_out)



