'''
@Author: your name
@Date: 2019-11-02 20:02:23
@LastEditTime : 2020-01-04 20:44:20
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: \ddi_classification\models\Rescal.py
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


class RLMLP(RelationLearningModel):
    def __init__(self, config):
        super(RLMLP, self).__init__(config)

        self.entity_embedding = nn.Parameter(
            torch.zeros(self.nentity, self.hidden_dim))
        self.relation_embedding = nn.Parameter(
            torch.zeros(self.nrelation, self.hidden_dim))

        self.M1 = nn.Parameter(
            torch.zeros(self.hidden_dim, self.hidden_dim))
        self.M2 = nn.Parameter(
            torch.zeros(self.hidden_dim, self.hidden_dim))
        self.M3 = nn.Parameter(
            torch.zeros(self.hidden_dim, self.hidden_dim))
        self.w = nn.Parameter(
            torch.zeros(self.hidden_dim))

        nn.init.xavier_uniform_(self.entity_embedding)
        nn.init.xavier_uniform_(self.relation_embedding)
        nn.init.xavier_uniform_(self.M1)
        nn.init.xavier_uniform_(self.M2)
        nn.init.xavier_uniform_(self.M3)
        nn.init.uniform_(self.w)

    def score(self, h, r, t):
        h = h.unsqueeze(1)
        r = r.unsqueeze(1)
        t = t.unsqueeze(1)
        score = torch.matmul(h, self.M1) + \
            torch.matmul(r, self.M2) + torch.matmul(t, self.M3)
        score = torch.tanh(score)
        score = torch.matmul(score, self.w)
        return score.flatten()


class RLRescal(RelationLearningModel):
    def __init__(self, config):
        super(RLRescal, self).__init__(config)

        self.entity_embedding = nn.Parameter(
            torch.zeros(self.nentity, self.hidden_dim))
        self.relation_embedding = nn.Parameter(
            torch.zeros(self.nrelation, self.hidden_dim, self.hidden_dim))

        nn.init.xavier_uniform_(self.entity_embedding)
        nn.init.xavier_uniform_(self.relation_embedding)

    def score(self, h, r, t):
        score = h.matmul(r).matmul(t.transpose(0, 1))
        return score.flatten()


class Analogy(RelationLearningModel):

    def __init__(self, config):
        super(Analogy, self).__init__(config)

        n = self.nentity
        d = self.hidden_dim
        K = self.nrelation

        self.E_re = nn.Parameter(torch.zeros(n, d))
        self.E_im = nn.Parameter(torch.zeros(n, d))

        self.R_re = nn.Parameter(torch.zeros(K, d))
        self.R_im = nn.Parameter(torch.zeros(K, d))

        self.E = nn.Parameter(torch.zeros(n, d*2))
        self.R = nn.Parameter(torch.zeros(n, d*2))

        nn.init.xavier_uniform_(self.E_re)
        nn.init.xavier_uniform_(self.E_im)
        nn.init.xavier_uniform_(self.R_re)
        nn.init.xavier_uniform_(self.R_im)
        nn.init.xavier_uniform_(self.E)
        nn.init.xavier_uniform_(self.R)

    def score(self, h_re, h_im, h, t_re, t_im, t, r_re, r_im, r):
        return (-torch.sum(r_re * h_re * t_re +
                           r_re * h_im * t_im +
                           r_im * h_re * t_im -
                           r_im * h_im * t_re, -1)
                - torch.sum(h * t * r, -1))

    def forward(self, sample):
        h_re = torch.index_select(
            self.E_re,
            dim=0,
            index=sample[:, 0]
        )
        h_im = torch.index_select(
            self.E_im,
            dim=0,
            index=sample[:, 0]
        )
        h = torch.index_select(
            self.E,
            dim=0,
            index=sample[:, 0]
        )

        t_re = torch.index_select(
            self.E_re,
            dim=0,
            index=sample[:, 2]
        )
        t_im = torch.index_select(
            self.E_im,
            dim=0,
            index=sample[:, 2]
        )
        t = torch.index_select(
            self.E,
            dim=0,
            index=sample[:, 2]
        )

        r_re = torch.index_select(
            self.R_re,
            dim=0,
            index=sample[:, 2]
        )
        r_im = torch.index_select(
            self.R_im,
            dim=0,
            index=sample[:, 2]
        )
        r = torch.index_select(
            self.R,
            dim=0,
            index=sample[:, 2]
        )
        
        return self.score(h_re, h_im, h, t_re, t_im, t, r_re, r_im, r)



class RotatE(nn.Module):
    def __init__(self, config):
        super(RotatE, self).__init__()
        self.config = config

        self.nentity = self.config['entity_num']
        self.nrelation = self.config['relation_num']
        self.hidden_dim = self.config['entity_embedding_dim']
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([12.0]),
            requires_grad=False
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor(
                [(self.gamma.item() + self.epsilon) / self.hidden_dim]),
            requires_grad=False
        )
        self.entity_dim = self.hidden_dim
        self.relation_dim = self.hidden_dim//2

        self.entity_embedding = nn.Parameter(
            torch.zeros(self.nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(
            torch.zeros(self.nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

    def score(self, head, relation, tail):
        head = head.unsqueeze(1)
        relation = relation.unsqueeze(1)
        tail = tail.unsqueeze(1)

        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_score = re_relation * re_tail + im_relation * im_tail
        im_score = re_relation * im_tail - im_relation * re_tail
        re_score = re_score - re_head
        im_score = im_score - im_head

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        # score = self.gamma.item() - score.sum(dim=2)
        return score.sum(dim=2)

    def forward(self, sample, mode):
        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(
                0), head_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(
                0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
        return self.score(head, relation, tail)

    def loss(self, positve_sample, negative_sample):
        negative_score = self.forward(negative_sample)
        negative_score = F.logsigmoid(
            negative_score-self.gamma.item()).mean(dim=1)

        positive_score = self.forward(positve_sample)
        positive_score = F.logsigmoid(
            self.gamma.item()-positive_score).squeeze(dim=1)

        positive_sample_loss = - positive_score.mean()
        negative_sample_loss = - negative_score.mean()

        loss = (positive_sample_loss + negative_sample_loss)/2
        return loss

    def get_ent_embeddings(self,):
        return self.entity_embedding.data

    def predict(self, h, t):
        r_positive = self.relation_embedding[1]
        return -self.score(h, r_positive, t).cpu().data.numpy()

    def predict_multi_class(self, h, t):
        score_list = []
        for interaction_i in range(self.config['relation_num']):
            r = self.relation_embedding(interaction_i)
            score_list.append(-self.score(h, r, t).cpu().data.numpy())
        return score_list


class TransH(nn.Module):
    def __init__(self, config):
        super(TransH, self).__init__()
        self.config = config
        self.ent_embeddings = nn.Embedding(
            self.config['entity_num'], self.config['hidden_embedding_dim'])
        self.rel_embeddings = nn.Embedding(
            self.config['relation_num'], self.config['hidden_embedding_dim'])
        self.norm_vector = nn.Embedding(
            self.config['relation_num'], self.config['hidden_embedding_dim'])
        self.criterion = nn.MarginRankingLoss(1.0, False)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        nn.init.xavier_uniform_(self.norm_vector.weight.data)

    def _calc(self, h, r, t):
        return torch.norm(h + r - t, 2, -1)

    def _transfer(self, e, norm):
        norm = F.normalize(norm, p=2, dim=-1)
        return e - torch.sum(e * norm, -1, True) * norm

    def forward(self, positve_sparse_tensor, negative_sparse_tensor):
        h = self.ent_embeddings(positve_sparse_tensor._indices()[0])
        r = self.rel_embeddings(positve_sparse_tensor._values().long())
        t = self.ent_embeddings(positve_sparse_tensor._indices()[1])
        r_norm = self.norm_vector(positve_sparse_tensor._values().long())
        h = self._transfer(h, r_norm)
        t = self._transfer(t, r_norm)
        p_score = self._calc(h, r, t)

        h = self.ent_embeddings(negative_sparse_tensor._indices()[0])
        r = self.rel_embeddings(negative_sparse_tensor._values().long())
        t = self.ent_embeddings(negative_sparse_tensor._indices()[1])
        r_norm = self.norm_vector(negative_sparse_tensor._values().long())
        h = self._transfer(h, r_norm)
        t = self._transfer(t, r_norm)
        n_score = self._calc(h, r, t)

        y = torch.tensor([-1.0], requires_grad=True)
        return self.criterion(p_score, n_score, y)

    def predict(self, h, t):
        r_positive = self.rel_embeddings(torch.tensor(1))
        return -self._calc(h, r_positive, t).data

    def get_ent_embeddings(self,):
        return self.ent_embeddings(torch.tensor(list(range(self.config['entity_num'])))).data


class TransR(nn.Module):
    def __init__(self, config):
        super(TransR, self).__init__()
        self.config = config
        self.ent_embeddings = nn.Embedding(
            self.config['entity_num'], self.config['entity_embedding_dim'])
        self.rel_embeddings = nn.Embedding(
            self.config['relation_num'], self.config['interaction_hidden_embedding_dim'])
        self.transfer_matrix = nn.Embedding(
            self.config['relation_num'], self.config['entity_embedding_dim'] * self.config['interaction_hidden_embedding_dim'])
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        identity = torch.zeros(
            self.config['interaction_hidden_embedding_dim'], self.config['entity_embedding_dim'])
        for i in range(self.config['interaction_hidden_embedding_dim']):
            identity[i][i] = 1
            if i == self.config['entity_embedding_dim'] - 1:
                break
        identity = identity.view(
            self.config['entity_embedding_dim'] * self.config['interaction_hidden_embedding_dim'])
        for i in range(self.config['relation_num']):
            self.transfer_matrix.weight.data[i] = identity

    def set_ent_embeddings(self, feature):
        self.ent_embeddings.weight.data = feature

    def get_feature_transformed(self, feature):
        r_transfer = self.transfer_matrix(torch.tensor(1))
        return self._transfer(feature, r_transfer).data

    def get_ent_embeddings(self,):
        return self.ent_embeddings(torch.tensor(list(range(self.config['entity_num'])))).data

    def _calc(self, h, r, t):
        return torch.norm(h + r - t, 2, -1)

    def _transfer(self, e, r_transfer):
        e = e.view(-1, self.config['entity_embedding_dim'], 1)
        r_transfer = r_transfer.view(
            -1, self.config['interaction_hidden_embedding_dim'], self.config['entity_embedding_dim'])
        e = torch.matmul(r_transfer, e)
        e = e.view(-1, self.config['interaction_hidden_embedding_dim'])
        return e

    def forward(self, sparse_tensor):
        h = self.ent_embeddings(sparse_tensor._indices()[0])
        r = self.rel_embeddings(sparse_tensor._values().long())
        t = self.ent_embeddings(sparse_tensor._indices()[1])
        r_transfer = self.transfer_matrix(
            sparse_tensor._values().long())
        h = self._transfer(h, r_transfer)
        t = self._transfer(t, r_transfer)
        score = self._calc(h, r, t)
        return score

    def regularization(self, sparse_tensor):
        h = self.ent_embeddings(sparse_tensor._indices()[0])
        r = self.rel_embeddings(sparse_tensor._values().long())
        t = self.ent_embeddings(sparse_tensor._indices()[1])
        r_transfer = self.transfer_matrix(
            sparse_tensor._values().long())
        h_transfer = self._transfer(h, r_transfer)
        t_transfer = self._transfer(t, r_transfer)
        regul = torch.cat(
            [F.relu(torch.norm(h, p=2, dim=1)-1),
             F.relu(torch.norm(t, p=2, dim=1)-1),
             F.relu(torch.norm(r, p=2, dim=1)-1),
             F.relu(torch.norm(h_transfer, p=2, dim=1)-1),
             F.relu(torch.norm(t_transfer, p=2, dim=1)-1)],
            -1
        )
        return regul

    def predict(self, h, t):
        r_positive = self.rel_embeddings(torch.tensor(1))
        r_transfer = self.transfer_matrix(torch.tensor(1))
        h = self._transfer(h, r_transfer)
        t = self._transfer(t, r_transfer)
        return -self._calc(h, r_positive, t).cpu().data.numpy()
