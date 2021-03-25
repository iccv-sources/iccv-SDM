#! /usr/bin/env python3

from .base_metric_loss_function import BaseMetricLossFunction
import torch
import torch.nn.functional as F
from ..utils import loss_and_miner_utils as lmu
from ..reducers import AvgNonZeroReducer


class TripletMarginLoss(BaseMetricLossFunction):
    """
    Args:
        margin: The desired difference between the anchor-positive distance and the
                anchor-negative distance.
        distance_norm: The norm used when calculating distance between embeddings
        power: Each pair's loss will be raised to this power.
        swap: Use the positive-negative distance instead of anchor-negative distance,
              if it violates the margin more.
        smooth_loss: Use the log-exp version of the triplet loss
    """
    def __init__(
        self,
        margin=0.05,
        distance_norm=2,
        power=1,
        swap=False,
        smooth_loss=False,
        triplets_per_anchor="all",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.margin = margin
        self.distance_norm = distance_norm
        self.power = power
        self.swap = swap
        self.smooth_loss = smooth_loss
        self.triplets_per_anchor = triplets_per_anchor
        
    def compute_loss(self, embeddings, labels, indices_tuple):
        indices_tuple = lmu.convert_to_triplets(indices_tuple, labels, t_per_anchor=self.triplets_per_anchor)
        anchor_idx, positive_idx, negative_idx = indices_tuple
        if len(anchor_idx) == 0:
            return self.zero_losses()
        
        anchors_list = []
        positives_list = []
        negatives_list = []
        a_p_dist_list = []
        a_n_dist_list = []

        for i in range(len(embeddings)):
            anchors, positives, negatives = embeddings[i][anchor_idx], embeddings[i][positive_idx], embeddings[i][negative_idx]
            anchors_list.append(anchors)
            positives_list.append(positives)
            negatives_list.append(negatives)

            a_p_dist = F.pairwise_distance(anchors, positives, self.distance_norm)
            a_n_dist = F.pairwise_distance(anchors, negatives, self.distance_norm)
            if self.swap:
                p_n_dist = F.pairwise_distance(positives, negatives, self.distance_norm)
                a_n_dist = torch.min(a_n_dist, p_n_dist)
            a_p_dist_list.append(a_p_dist)
            a_n_dist_list.append(a_n_dist)           

        # for anch in anchors_list:
        #     for posi in positives_list:
        #         for neg in negatives_list:
        #             a_p_dist = F.pairwise_distance(anch, posi, self.distance_norm)
        #             a_n_dist = F.pairwise_distance(anch, neg, self.distance_norm)
        #             if self.swap:
        #                 p_n_dist = F.pairwise_distance(posi, neg, self.distance_norm)
        #                 a_n_dist = torch.min(a_n_dist, p_n_dist)
        #             a_p_dist_list.append(a_p_dist)
        #             a_n_dist_list.append(a_n_dist)



        a_p_dist_list = torch.cat([a_p_dist.unsqueeze(0) for a_p_dist in a_p_dist_list], 0)
        a_n_dist_list = torch.cat([a_n_dist.unsqueeze(0) for a_n_dist in a_n_dist_list], 0)
        
        alpha = 32
        softmax = torch.softmax(alpha * a_p_dist_list, 0)
        softmin = torch.softmax(-alpha * a_n_dist_list, 0)

        a_p_dist = (a_p_dist_list * softmax).sum(0)
        a_n_dist = (a_n_dist_list * softmin).sum(0)

        a_p_dist = a_p_dist ** self.power
        a_n_dist = a_n_dist ** self.power

        #std
        a_p_std = a_p_dist_list.std(0)
        a_n_std = a_n_dist_list.std(0)
        

        if self.smooth_loss:
            inside_exp = a_p_dist - a_n_dist
            inside_exp = self.maybe_modify_loss(inside_exp)
            loss = torch.log(1 + torch.exp(inside_exp))
        else:
            dist = a_p_dist - a_n_dist
            loss_modified = self.maybe_modify_loss(dist + self.margin)
            loss = torch.nn.functional.relu(loss_modified)

        return {"loss": {"losses": loss, "indices": indices_tuple, "reduction_type": "triplet"}}

    def maybe_modify_loss(self, x):
        return x

    def get_default_reducer(self):
        return AvgNonZeroReducer()