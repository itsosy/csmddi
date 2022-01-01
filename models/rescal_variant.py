
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
from .utils import *


class Rescal(nn.Module):
    """ 最基本的rescal，用torch实现 """

    def __init__(self, config, adj_list):
        super(Rescal, self).__init__()

        n = config['entity_num']
        d = config['entity_embedding_dim']
        K = config['relation_num']

        self.E = nn.Parameter(torch.zeros(n, d))
        self.M = nn.Parameter(torch.zeros(K, d, d))

        nn.init.xavier_uniform_(self.E)
        nn.init.xavier_uniform_(self.M)

        self.A = torch.tensor(adj_list).to(device=config['device'])

    def forward(self):
        E = F.normalize(self.E)
        A_predict = E.matmul(self.M).matmul(E.transpose(0, 1))

        loss = torch.norm(self.A - A_predict) ** 2
        return loss


class RescalFeatureEmbedding(nn.Module):
    """ 蛋白特征嵌入，累加得到药物嵌入 """

    def __init__(self, config, adj_list):
        super(RescalFeatureEmbedding, self).__init__()

        self.entity_feature = torch.tensor(config['entity_feature']).float()
        feature_num = self.entity_feature.shape[1]
        feature_dim = config['feature_embedding_dim']
        relation_num = config['relation_num']

        self.feature_embedding = nn.Parameter(
            torch.zeros(feature_num, feature_dim))
        self.M = nn.Parameter(torch.zeros(
            relation_num, feature_dim, feature_dim))
        self.feature_importance = nn.Parameter(torch.zeros(
            relation_num, feature_num, 1))

        nn.init.xavier_uniform_(self.feature_embedding)
        nn.init.xavier_uniform_(self.M)
        nn.init.xavier_uniform_(self.feature_importance)

        self.A = torch.tensor(adj_list).to(device=config['device'])
        self.entity_feature = self.entity_feature.to(device=config['device'])

    def forward(self):
        E = self.get_entity_emb()
        A_predict = E.matmul(self.M).matmul(E.transpose(0, 1))

        loss = torch.norm(self.A - A_predict) ** 2
        return loss

    def get_entity_emb(self,):
        feature_importance = self.feature_importance.expand(
            -1, -1, self.feature_embedding.shape[1])
        feature_emb = torch.mul(self.feature_embedding, feature_importance)
        return torch.matmul(self.entity_feature, feature_emb)