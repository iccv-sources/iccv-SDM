#! /usr/bin/env python3

from .base_metric_loss_function import BaseMetricLossFunction
from ..utils import loss_and_miner_utils as lmu, common_functions as c_f
import torch

class NCALoss(BaseMetricLossFunction):
    def __init__(self, softmax_scale=1, **kwargs):
        super().__init__(**kwargs)
        self.softmax_scale = softmax_scale

    # https://www.cs.toronto.edu/~hinton/absps/nca.pdf
    def compute_loss(self, embeddings, labels, indices_tuple):
        if len(embeddings) <= 1:
            return self.zero_losses()
        return self.nca_computation(embeddings, embeddings, labels, labels, indices_tuple)

    def nca_computation(self, query, reference, query_labels, reference_labels, indices_tuple):
        miner_weights = lmu.convert_to_weights(indices_tuple, query_labels)
        x_list = []
        for q, r in zip(query, reference):
            x = -lmu.dist_mat(q, r, squared=True)
            if q is r:
                diag_idx = torch.arange(q.size(0))
                x[diag_idx, diag_idx] = float('-inf')
            x_list.append(x)

        x_list = torch.stack(x_list, dim=0)
        alpha = 32
        sfmax = torch.softmax(alpha * x_list, dim=0)
        sfmin = torch.softmax(-alpha * x_list, dim=0)
        x = (x_list * sfmin).sum(0)

        same_labels = (query_labels.unsqueeze(1) == reference_labels.unsqueeze(0)).float()
        exp = torch.nn.functional.softmax(self.softmax_scale*x, dim=1)
        exp = torch.sum(exp * same_labels, dim=1)
        non_zero = exp!=0
        loss = -torch.log(exp[non_zero])*miner_weights[non_zero]
        return {"loss": {"losses": loss, "indices": c_f.torch_arange_from_size(query)[non_zero], "reduction_type": "element"}}