'''
@Author: your name
@Date: 2019-10-30 21:03:58
@LastEditTime : 2019-12-27 16:10:29
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: models\gae\model.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from models.gae.layers import GraphConvolution


class GCNModelAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelAE, self).__init__()
        self.gc1 = GraphConvolution(
            input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(
            hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden = self.gc1(x, adj)
        hidden = self.gc2(hidden, adj)
        return hidden

    def forward(self, x, adj):
        self.node_emb = self.encode(x, adj)
        return self.dc(self.node_emb)


class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelVAE, self).__init__()
        self.gc1 = GraphConvolution(
            input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(
            hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(
            hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar


class GCNModelAEResidual(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelAEResidual, self).__init__()
        self.gc1 = GraphConvolution(
            input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(
            hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        # self.dc = InnerProductDecoder(dropout, act=lambda x: x)
        self.adj_recovered_weight = nn.Parameter(
            torch.rand((hidden_dim2, input_feat_dim)))
        nn.init.kaiming_normal_(self.adj_recovered_weight)

    # def encode(self, x, adj):  # 残差拼接
    #     hidden = self.gc1(x, adj)
    #     hidden = torch.cat((self.gc2(hidden, adj), hidden), 1)
    #     return hidden

    def encode(self, x, adj):  # 残差相加
        hidden = self.gc1(x, adj)
        hidden = self.gc2(hidden, adj) + hidden
        return hidden

    def forward(self, x, adj):
        hidden = self.encode(x, adj)
        adj_recovered = torch.mm(hidden, self.adj_recovered_weight)
        return adj_recovered, hidden


class GCNModifiedAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModifiedAE, self).__init__()
        self.gc1 = GraphConvolution(
            input_feat_dim, hidden_dim1, dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden = self.gc1(x, adj)
        return hidden

    def forward(self, x, adj):
        hidden = self.encode(x, adj)
        self.adj_recovered_weight = self.gc1.weight.t()
        adj_recovered = torch.matmul(hidden, self.adj_recovered_weight)
        return adj_recovered, [hidden]


class GCNModifiedAE2(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout, feature=None):
        super(GCNModifiedAE2, self).__init__()
        self.gc1 = GraphConvolution(
            input_feat_dim, hidden_dim1, dropout, act=F.relu)
        # self.gc2 = GraphConvolution(
        #     hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.node_map = nn.Linear(hidden_dim1, hidden_dim1)
        self.neighbor_map = nn.Linear(hidden_dim1, hidden_dim1)

    def forward(self, x, adj_norm):
        H1 = self.gc1(x, adj_norm)
        self.node_emb = self.node_map(H1)
        self.neighbor_emb = self.neighbor_map(H1)
        adj_recovered = torch.mm(self.node_emb, self.neighbor_emb.t())
        return adj_recovered, [self.node_emb, self.neighbor_emb]


class GCNModifiedAE2ASum(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModifiedAE2ASum, self).__init__()
        self.gc1 = GraphConvolution(
            input_feat_dim, hidden_dim1, dropout, act=lambda x: x)
        self.gc2 = GraphConvolution(
            hidden_dim1, hidden_dim2, dropout, act=lambda x: x)

    def forward(self, x, adj):
        hidden_1 = self.gc1(x, adj)
        hidden_2 = self.gc2(hidden_1, adj)
        self.adj_recovered_weight_1 = self.gc1.weight.t()
        self.adj_recovered_weight_2 = torch.matmul(
            self.gc2.weight.t(), self.gc1.weight.t())
        adj_recovered_1 = torch.matmul(hidden_1, self.adj_recovered_weight_1)
        adj_recovered_2 = torch.matmul(hidden_2, self.adj_recovered_weight_2)
        return (adj_recovered_1, adj_recovered_2), (hidden_1, hidden_2)


class GCNModifiedAEBoostingFirstLayer(nn.Module):
    """ 〖A ̂=H〗^((1))∗(H^((2) ) W^(1)T )^T """

    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModifiedAEBoostingFirstLayer, self).__init__()
        self.gc1 = GraphConvolution(
            input_feat_dim, hidden_dim1, dropout, act=lambda x: x)
        self.gc2 = GraphConvolution(
            hidden_dim1, hidden_dim2, dropout, act=lambda x: x)

        self.adj_two_hop = None

    def forward(self, x, adj):
        if self.adj_two_hop is None:
            self.adj_two_hop = torch.mm(adj.to_dense(), adj.to_dense())
            self.adj_two_hop = self.adj_two_hop - \
                torch.diagflat(torch.diagonal(self.adj_two_hop))
        embeddings = []
        embeddings.append(self.gc1(x, adj))
        # emb_2 = torch.mm(self.adj_two_hop, self.gc1.weight)
        # emb_2 = torch.mm(emb_2, self.gc2.weight)
        # embeddings.append(emb_2)
        embeddings.append(self.gc2(embeddings[-1], adj))
        # self.adj_recovered_weight = torch.matmul(
        #     embeddings[1], self.gc2.weight.t()).t()
        adj_recovered = torch.matmul(
            embeddings[0], embeddings[1].t())
        return adj_recovered, embeddings


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj
