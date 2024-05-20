
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


class RescalSymmetricTriu(Rescal):
    """ rescal改进  关系矩阵对称，上三角矩阵    """

    def __init__(self, config, adj_list):
        super(RescalSymmetricTriu, self).__init__(config, adj_list)

    def forward(self):
        E = F.normalize(self.E)
        M_symmetric = symmetric_matrix(self.M)

        A_predict = E.matmul(M_symmetric).matmul(E.transpose(0, 1))

        loss = torch.norm(self.A - A_predict) ** 2
        return loss


class RescalSymmetric(Rescal):
    """ rescal改进  关系矩阵对称，(矩阵+转置)/2   """

    def __init__(self, config, adj_list):
        super(RescalSymmetric, self).__init__(config, adj_list)

    def forward(self):
        E = F.normalize(self.E)
        M_symmetric = (self.M + self.M.permute(0, 2, 1))/2

        A_predict = E.matmul(M_symmetric).matmul(E.transpose(0, 1))

        loss = torch.norm(self.A - A_predict) ** 2
        return loss


class RescalRelationSimilarity(Rescal):
    """ rescal改进  关系矩阵之间相似    """

    def __init__(self, config, adj_list, relation_similarity):
        super(RescalRelationSimilarity, self).__init__(config, adj_list)

        self.relation_similarity = torch.tensor(
            relation_similarity).to(device=config['device'])

    def forward(self):
        E = F.normalize(self.E)
        A_predict = E.matmul(self.M).matmul(E.transpose(0, 1))
        loss1 = torch.norm(self.A - A_predict) ** 2

        # 添加关系矩阵之间相似性
        M_flatten = torch.flatten(self.M, start_dim=1)
        M_similarity = cosine_similarity(M_flatten)
        loss2 = torch.norm(self.relation_similarity - M_similarity) ** 2

        return loss1+loss2

    def more_similarity(self,):
        M_flatten = torch.flatten(self.M, start_dim=1)
        M_similarity = cosine_similarity(M_flatten)
        loss = torch.norm(self.relation_similarity - M_similarity) ** 2
        return loss


class RescalTypeSpecificiEmbedding(Rescal):
    """ rescal改进  药物在每个类型下都有一个嵌入表示    """

    def __init__(self, config, adj_list):
        super(RescalTypeSpecificiEmbedding, self).__init__(config, adj_list)

        n = config['entity_num']
        d = config['entity_embedding_dim']
        K = config['relation_num']

        self.E_maps = nn.Parameter(torch.zeros(K, d, d))
        nn.init.xavier_uniform_(self.E_maps)

    def forward(self):
        E = F.normalize(self.E)
        E_multi = E.matmul(self.E_maps)

        A_predict = E_multi.matmul(self.M).matmul(E_multi.transpose(1, 2))

        loss = torch.norm(self.A - A_predict) ** 2
        return loss


class RescalFeatureSimilarity(Rescal):
    """ rescal改进 
            属性特征相似性约束
    """

    def __init__(self, config, adj_list, feature_similarity):
        super(RescalFeatureSimilarity,
              self).__init__(config, adj_list)

        self.feature_similarity = torch.tensor(
            feature_similarity).to(device=config['device'])

    def forward(self):
        E = F.normalize(self.E)

        A_predict = E.matmul(self.M).matmul(E.transpose(0, 1))

        loss = torch.norm(self.A - A_predict) ** 2

        # 药物原始嵌入的内积作为相似性约束
        loss += torch.norm(self.feature_similarity -
                           E.matmul(E.transpose(0, 1))) ** 2
        return loss

    def more_similarity(self,):
        E = F.normalize(self.E)
        loss = torch.norm(self.feature_similarity -
                          E.matmul(E.transpose(0, 1))) ** 2
        return loss


class RescalFeatureSymmetricFeatureSimilarity(Rescal):
    """ rescal改进 
            关系矩阵对称
            属性特征相似性约束
    """

    def __init__(self, config, adj_list, feature_similarity):
        super(RescalFeatureSymmetricFeatureSimilarity,
              self).__init__(config, adj_list)

        self.feature_similarity = torch.tensor(
            feature_similarity).to(device=config['device'])

    def forward(self):
        E = F.normalize(self.E)
        M_symmetric = symmetric_matrix(self.M)
        A_predict = E.matmul(M_symmetric).matmul(E.transpose(0, 1))
        loss = torch.norm(self.A - A_predict) ** 2

        # 药物原始嵌入的内积作为相似性约束
        loss += torch.norm(self.feature_similarity -
                           E.matmul(E.transpose(0, 1))) ** 2
        return loss

    def more_similarity(self,):
        E = F.normalize(self.E)
        loss = torch.norm(self.feature_similarity -
                          E.matmul(E.transpose(0, 1))) ** 2
        return loss


class RescalSymmetricRelationSimilarityFeatureSimilarity(Rescal):
    """ rescal改进 
            关系矩阵对称
            关系矩阵相似
            属性特征相似性约束
    """

    def __init__(self, config, adj_list, relation_similarity, feature_similarity):
        super(RescalSymmetricRelationSimilarityFeatureSimilarity,
              self).__init__(config, adj_list)

        self.relation_similarity = torch.tensor(
            relation_similarity).to(device=config['device'])

        self.feature_similarity = torch.tensor(
            feature_similarity).to(device=config['device'])

    def forward(self):
        E = F.normalize(self.E)
        M_symmetric = symmetric_matrix(self.M)
        A_predict = E.matmul(M_symmetric).matmul(E.transpose(0, 1))
        loss1 = torch.norm(self.A - A_predict) ** 2

        # 关系矩阵之间相似性
        M_flatten = torch.flatten(M_symmetric, start_dim=1)
        M_similarity = cosine_similarity(M_flatten)
        loss2 = torch.norm(self.relation_similarity - M_similarity) ** 2

        # 药物原始嵌入的内积作为相似性约束
        loss3 = torch.norm(self.feature_similarity -
                           E.matmul(E.transpose(0, 1))) ** 2
        return loss1+loss2+loss3

    def more_similarity(self,):
        # 关系矩阵之间相似性
        M_symmetric = symmetric_matrix(self.M)
        M_flatten = torch.flatten(M_symmetric, start_dim=1)
        M_similarity = cosine_similarity(M_flatten)
        loss2 = torch.norm(self.relation_similarity - M_similarity) ** 2

        # 药物原始嵌入的内积作为相似性约束
        E = F.normalize(self.E)
        loss3 = torch.norm(self.feature_similarity -
                           E.matmul(E.transpose(0, 1))) ** 2
        return loss2+loss3


class RescalSymmetricTypeSpecificiEmbeddingFeatureSimilarity(RescalTypeSpecificiEmbedding):
    """ rescal改进 
            关系矩阵对称
            药物在每个类型下都有一个嵌入表示
            属性特征相似性约束
    """

    def __init__(self, config, adj_list, feature_similarity):
        super(RescalSymmetricTypeSpecificiEmbeddingFeatureSimilarity,
              self).__init__(config, adj_list)

        self.feature_similarity = torch.tensor(
            feature_similarity).to(device=config['device'])

    def forward(self):
        E = F.normalize(self.E)
        E_multi = E.matmul(self.E_maps)
        M_symmetric = symmetric_matrix(self.M)

        A_predict = E_multi.matmul(M_symmetric).matmul(E_multi.transpose(1, 2))

        loss = torch.norm(self.A - A_predict) ** 2

        # 药物原始嵌入的内积作为相似性约束
        loss += torch.norm(self.feature_similarity -
                           E.matmul(E.transpose(0, 1))) ** 2
        return loss

    def more_similarity(self,):
        E = F.normalize(self.E)
        loss = torch.norm(self.feature_similarity -
                          E.matmul(E.transpose(0, 1))) ** 2
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


class RescalAnalogy(Rescal):
    """ rescal改进  MM^T = M^TM    """

    def __init__(self, config, adj_list):
        super(RescalAnalogy, self).__init__(config, adj_list)

    def forward(self):
        E = F.normalize(self.E)
        A_predict = E.matmul(self.M).matmul(E.transpose(0, 1))

        loss = torch.norm(self.A - A_predict) ** 2

        # Analogy loss
        loss_ananoly = 0
        for k in range(self.M.shape[0]):
            M_transpos = self.M[k].transpose(0, 1)
            loss_ananoly += torch.norm(self.M[k].matmul(
                M_transpos) - M_transpos.matmul(self.M[k])) ** 2

        return loss+loss_ananoly
