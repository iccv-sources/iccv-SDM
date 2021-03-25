#! /usr/bin/env python3

import numpy as np
from .base_trainer import BaseTrainer
import torch


class MetricLossOnly(BaseTrainer):
    def calculate_loss(self, curr_batch):
        data, labels = curr_batch
        embeddings_list = []
        for j in range(8):
            embeddings = self.compute_embeddings(data)
            indices_tuple = self.maybe_mine_embeddings(embeddings, labels)
            # embeddings_list.append(embeddings)
            for i in range(8):
                embeddings_ = torch.nn.Dropout(0.05)(embeddings)
                embeddings_list.append(embeddings_)
        embeddings = embeddings_list
        self.losses["metric_loss"] = self.maybe_get_metric_loss(embeddings, labels, indices_tuple)
        
    def maybe_get_metric_loss(self, embeddings, labels, indices_tuple):
        if self.loss_weights.get("metric_loss", 0) > 0:
            return self.loss_funcs["metric_loss"](embeddings, labels, indices_tuple)
        return 0

