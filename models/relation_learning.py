'''
@Author: your name
@Date: 2019-11-02 20:02:23
@LastEditTime : 2020-01-04 20:44:20
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: \models\Rescal.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RelationLearningModel(nn.Module):
    """ 关系学习模型的基类，提供一些共有的初始化和方法 """

    def __init__(self, config):
        super(RelationLearningModel, self).__init__()
        self.config = config

        self.nentity = self.config['entity_num']
        self.nrelation = self.config['relation_num']
        self.hidden_dim = self.config['entity_embedding_dim']

    def forward(self, sample):
        head = torch.index_select(
            self.entity_embedding,
            dim=0,
            index=sample[:, 0]
        )
        relation = torch.index_select(
            self.relation_embedding,
            dim=0,
            index=sample[:, 1]
        )
        tail = torch.index_select(
            self.entity_embedding,
            dim=0,
            index=sample[:, 2]
        )
        return self.score(head, relation, tail)

    def loss(self, positve_sample, negative_sample):
        positive_score = self.forward(positve_sample)
        positive_sample_loss = -F.logsigmoid(positive_score).mean(dim=-1)

        negative_score = self.forward(negative_sample)
        negative_sample_loss = -F.logsigmoid(-negative_score).mean(dim=-1)

        loss = (positive_sample_loss + negative_sample_loss)/2
        return loss


class TransE(RelationLearningModel):
    def __init__(self, config):
        super(TransE, self).__init__(config)

        self.entity_embedding = nn.Parameter(
            torch.zeros(self.nentity, self.hidden_dim))
        self.relation_embedding = nn.Parameter(
            torch.zeros(self.nrelation, self.hidden_dim))

        nn.init.xavier_uniform_(self.entity_embedding)
        nn.init.xavier_uniform_(self.relation_embedding)

        self.gamma = 12.0

    def score(self, h, r, t):
        h = F.normalize(h, 2, -1)
        r = F.normalize(r, 2, -1)
        t = F.normalize(t, 2, -1)
        score = (h + r) - t
        score = self.gamma - torch.norm(score, 1, -1).flatten()
        return score