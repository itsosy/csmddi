'''
@Author: your name
@Date: 2019-12-26 09:40:33
@LastEditTime : 2020-01-14 09:24:13
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: \models\models_train_and_test.py
'''


import importlib
import torch
from torch import optim
import numpy as np
import scipy.sparse as sp
from math import ceil
from scipy.stats import rankdata
import sklearn
from sklearn.cross_decomposition import PLSRegression as sk_plsr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import NMF as sk_nmf
from sklearn.preprocessing import label_binarize, normalize
from sklearn.utils import shuffle
from sklearn.decomposition import TruncatedSVD
from torch._C import dtype
from .relation_learning import *
from .gae.utils import preprocess_graph
from .gae.model import *
from .gae.optimizer import loss_function as gae_loss
from .rescal_tensor_factorization import als as rescal_tensor_factorization
from .rescal_variant import *

from torch.utils.data import DataLoader
from .dataloader import TrainDataset
from .dataloader import BidirectionalOneShotIterator

from .utils import *


class BaseModelTrain():
    """ 模型训练的基本类，主要包括，根据药物训练集和测试集划分数据，计算预测指标等 """

    def __init__(self, config, data):
        super(BaseModelTrain, self).__init__()
        self.config = config
        self.data = data

    def init(self, train_drugs_indices, test_drugs_indices):
        self.train_drugs_indices, self.test_drugs_indices = sorted(
            train_drugs_indices), sorted(test_drugs_indices)

        self.train_drugs_num = len(self.train_drugs_indices)
        self.test_drugs_num = len(self.test_drugs_indices)

        # 训练集的邻接矩阵
        self.adj_train = np.zeros(
            (self.train_drugs_num, self.train_drugs_num), dtype=np.int)
        for i in range(self.adj_train.shape[0]):
            for j in range(i+1, self.adj_train.shape[1]):
                self.adj_train[i, j] = self.data.adj[train_drugs_indices[i],
                                                     train_drugs_indices[j]]
        self.adj_train = self.adj_train + self.adj_train.T

    def train(self,):
        pass

    def test(self,):
        if self.config['binary_or_multi'] == 'binary':
            if self.config['S1_or_S2'] == 'S1':
                label_true, score_predict = self.binary_test_S1()
            elif self.config['S1_or_S2'] == 'S2':
                label_true, score_predict = self.binary_test_S2()
            return binary_evaluation_result(label_true, score_predict)
        elif self.config['binary_or_multi'] == 'multi':
            if self.config['S1_or_S2'] == 'S1':
                label_true, score_predict = self.multi_test_S1()
            elif self.config['S1_or_S2'] == 'S2':
                label_true, score_predict = self.multi_test_S2()
            return multi_evaluation_result(label_true, score_predict)
        else:
            raise RuntimeError('not support {}'.format(
                self.config['binary_or_multi']))

    def binary_test_S1(self):
        score_predict_matrices = self.predict_test_S1()
        label_true, score_predict = [], []
        for i in range(self.test_drugs_num):
            for j in range(self.train_drugs_num):
                label_true.append(self.data.adj[self.test_drugs_indices[i],
                                                self.train_drugs_indices[j]])
                score_predict.append(score_predict_matrices[i, j])

        return label_true, score_predict

    def binary_test_S2(self,):
        score_predict_matrices = self.predict_test_S2()
        label_true, score_predict = [], []
        for i in range(self.test_drugs_num):
            for j in range(i+1, self.test_drugs_num):
                label_true.append(
                    self.data.adj[self.test_drugs_indices[i], self.test_drugs_indices[j]])
                score_predict.append(score_predict_matrices[i, j])

        return label_true, score_predict

    def multi_test_S1(self):
        score_predict_matrices = self.predict_test_S1()
        label_true, score_predict = [], []
        for i in range(self.test_drugs_num):
            for j in range(self.train_drugs_num):
                interaction_idx = self.data.adj[self.test_drugs_indices[i],
                                                self.train_drugs_indices[j]]
                if interaction_idx > 0:
                    label_true.append(interaction_idx)
                    score_predict.append(score_predict_matrices[:, i, j])
        score_predict = np.array(score_predict)
        label_true = label_binarize(
            label_true, classes=np.arange(1, self.data.interaction_num))

        return label_true, score_predict

    def multi_test_S2(self):
        score_predict_matrices = self.predict_test_S2()
        label_true, score_predict = [], []
        for i in range(self.test_drugs_num):
            for j in range(i+1, self.test_drugs_num):
                interaction_idx = self.data.adj[self.test_drugs_indices[i],
                                                self.test_drugs_indices[j]]
                if interaction_idx > 0:
                    label_true.append(interaction_idx)
                    score_predict.append(score_predict_matrices[:, i, j])
        score_predict = np.array(score_predict)
        label_true = label_binarize(
            label_true, classes=np.arange(1, self.data.interaction_num))

        return label_true, score_predict

    def feature_map_to_embedding(self,):
        module = importlib.import_module('models.feature_learning')
        map_model = getattr(module, self.config['map_model'])()

        # 回归训练集
        train_feature = self.data.feature[self.train_drugs_indices]
        map_model.fit(train_feature, self.train_embedding)

        # 回归测试集
        test_features = self.data.feature[self.test_drugs_indices]
        self.test_embedding = map_model.predict(test_features)

        return self.test_embedding


class ColdStartGAE(BaseModelTrain):
    def __init__(self, config, data):
        super(ColdStartGAE, self).__init__(config, data)

    def init(self, train_drugs_indices, test_drugs_indices):
        super(ColdStartGAE, self).init(train_drugs_indices, test_drugs_indices)

        self.adj_train = torch.tensor(self.adj_train).float()
        self.feature_train = torch.eye(self.train_drugs_num).float()

    def train(self):
        n_nodes, feat_dim = self.feature_train.shape
        adj_norm = preprocess_graph(self.adj_train)

        self.GAE = GCNModelAE(feat_dim, hidden_dim1=512,
                              hidden_dim2=256, dropout=0)
        optimizer = optim.Adam(self.GAE.parameters())

        def loss_func():
            self.GAE.train()
            optimizer.zero_grad()
            adj_recovered = self.GAE(self.feature_train, adj_norm)

            loss = gae_loss(preds=adj_recovered, labels=self.adj_train,
                            mu=None, logvar=None, n_nodes=n_nodes,
                            norm=None, pos_weight=None)
            loss.backward()
            optimizer.step()
            return loss

        fit_model(loss_func=loss_func, model_name='GAE',
                  max_iter=self.config['epoch_num'])

        self.train_embedding = self.GAE.node_emb.data.numpy()

    def predict_test_S1(self,):
        test_emb = self.feature_map_to_embedding()
        adj_recovered = np.dot(test_emb, self.train_embedding.T)
        return adj_recovered

    def predict_test_S2(self,):
        test_emb = self.feature_map_to_embedding()
        adj_recovered = np.dot(test_emb, test_emb.T)
        return adj_recovered


class ColdStartSVD(BaseModelTrain):
    def __init__(self, config, data):
        super(ColdStartSVD, self).__init__(config, data)

    def train(self):
        svd = TruncatedSVD(
            n_components=self.config['drug_hidden_embedding_dim'], n_iter=7, random_state=42)
        self.train_embedding = svd.fit_transform(self.adj_train)

    def predict_test_S1(self,):
        test_emb = self.feature_map_to_embedding()
        adj_recovered = np.dot(test_emb, self.train_embedding.T)
        return adj_recovered

    def predict_test_S2(self,):
        test_emb = self.feature_map_to_embedding()
        adj_recovered = np.dot(test_emb, test_emb.T)
        return adj_recovered


class ColdStartRescalTensorFactorization(BaseModelTrain):
    def __init__(self, config, data):
        super(ColdStartRescalTensorFactorization,
              self).__init__(config, data)

    def init(self, train_drugs_indices, test_drugs_indices):
        super(ColdStartRescalTensorFactorization, self).init(
            train_drugs_indices, test_drugs_indices)

        # 多个类型的邻接矩阵
        adj_list = []
        if self.config['adj'] == 'adj_multi':
            adj_list = [np.zeros(self.adj_train.shape)
                        for _ in range(1, self.data.interaction_num)]
            for i in range(self.adj_train.shape[0]):
                for j in range(i+1, self.adj_train.shape[1]):
                    interaction_idx = self.adj_train[i, j]
                    if interaction_idx > 0:
                        adj_list[interaction_idx-1][i, j] = 1
            adj_list = [adj+adj.T for adj in adj_list]
        else:
            adj_list.append(self.adj_train)

        self.adj_list = adj_list

    def train(self):
        for i in range(len(self.adj_list)):
            self.adj_list[i] = sp.csr_matrix(self.adj_list[i])
        self.train_embedding, self.relation_matrices, _, _, _ = rescal_tensor_factorization(
            X=self.adj_list,
            rank=self.config['drug_hidden_embedding_dim'],
            lambda_A=0.01,
            lambda_R=0.01,
        )

    def predict_test_S1(self,):
        test_embedding = self.feature_map_to_embedding()
        score_predict_matrices = np.array([test_embedding.dot(relation_matrix).dot(self.train_embedding.T)
                                           for relation_matrix in self.relation_matrices])
        if self.config['binary_or_multi'] == 'binary':
            return score_predict_matrices[0, :, :]
        else:
            return score_predict_matrices

    def predict_test_S2(self,):
        test_embedding = self.feature_map_to_embedding()
        score_predict_matrices = np.array([test_embedding.dot(relation_matrix).dot(test_embedding.T)
                                           for relation_matrix in self.relation_matrices])
        if self.config['binary_or_multi'] == 'binary':
            return score_predict_matrices[0, :, :]
        else:
            return score_predict_matrices


class ColdStartRescalTensorFactorizationTorch(ColdStartRescalTensorFactorization):

    def __init__(self, config, data):
        super(ColdStartRescalTensorFactorizationTorch,
              self).__init__(config, data)

    def train_(self):
        self.model.to(device=self.config['device'])

        optimizer = torch.optim.Adam(
            self.model.parameters(), weight_decay=0.001)

        def loss_func():
            optimizer.zero_grad()
            loss = self.model()
            loss.backward()
            optimizer.step()
            return loss

        fit_model(loss_func=loss_func, model_name='Rescal',
                  max_iter=self.config['epoch_num'])

    def train(self,):
        self.model = Rescal(
            config={
                'entity_num': self.train_drugs_num,
                'entity_embedding_dim': self.config['drug_hidden_embedding_dim'],
                'relation_num': len(self.adj_list),
                'device': self.config['device']
            },
            adj_list=self.adj_list,
        )
        self.train_()
        self.train_embedding = self.model.E.detach().cpu().numpy()
        self.relation_matrices = self.model.M.detach().cpu().numpy()

    def predict_test_S1(self,):
        test_embedding = self.feature_map_to_embedding()
        score_predict_matrices = np.matmul(
            np.matmul(test_embedding, self.relation_matrices), self.train_embedding.T)

        if self.config['binary_or_multi'] == 'binary':
            return score_predict_matrices[0, :, :]
        else:
            return score_predict_matrices

    def predict_test_S2(self,):
        test_embedding = self.feature_map_to_embedding()
        score_predict_matrices = np.matmul(
            np.matmul(test_embedding, self.relation_matrices), test_embedding.T)

        if self.config['binary_or_multi'] == 'binary':
            return score_predict_matrices[0, :, :]
        else:
            return score_predict_matrices


class ColdStartRescalFeatureEmbedding(ColdStartRescalTensorFactorizationTorch):
    """ 蛋白特征嵌入，将蛋白嵌入累加得到药物嵌入 """

    def __init__(self, config, data):
        super(ColdStartRescalFeatureEmbedding,
              self).__init__(config, data)

    def train(self,):
        self.train_feature = self.data.feature[self.train_drugs_indices]

        self.model = RescalFeatureEmbedding(
            config={
                'entity_feature': self.train_feature,
                'feature_embedding_dim': self.config['drug_hidden_embedding_dim'],
                'relation_num': len(self.adj_list),
                'device': self.config['device']
            },
            adj_list=self.adj_list,
        )
        self.train_()
        self.feature_embedding = self.model.feature_embedding.detach().cpu().numpy()
        self.relation_matrices = self.model.M.detach().cpu().numpy()

        self.train_embedding = drug_emb_from_feature_emb(
            self.train_feature, self.feature_embedding)

    def feature_map_to_embedding(self):
        self.test_feature = self.data.feature[self.test_drugs_indices]
        self.test_embedding = drug_emb_from_feature_emb(
            self.test_feature, self.feature_embedding)
        return self.test_embedding


class ColdStartRelationLearning(BaseModelTrain):
    def __init__(self, config, data):
        super(ColdStartRelationLearning, self).__init__(config, data)

    def train(self,):
        # 导入 transE，mlp，rescal 等关系学习模型
        module = importlib.import_module('models.relation_learning')
        self.model_relation_learning = getattr(
            module, self.config['relation_learning_model'])(config={
                'entity_num': self.adj_train.shape[0],
                'relation_num': self.data.interaction_num,
                'entity_embedding_dim': self.config['drug_hidden_embedding_dim'],
            })

        # 取出所有的 有反应的ddi
        self.positive_triples_all = []
        for row in range(self.adj_train.shape[0]):
            for col in range(row+1, self.adj_train.shape[1]):
                interaction_idx = self.adj_train[row, col].item()
                if interaction_idx > 0:
                    self.positive_triples_all.append(
                        (row, interaction_idx, col))

        # 正负样本采样，类构造
        self.dataloader = DataLoader(
            TrainDataset(self.positive_triples_all,
                         self.adj_train.shape[0], self.data.interaction_num, 1, 'tail-batch'),
            batch_size=self.config['batch_size'],
            shuffle=True,
            collate_fn=TrainDataset.collate_fn
        )

        optimizer = torch.optim.Adam(
            self.model_relation_learning.parameters(),
            weight_decay=0.001,
        )

        def loss_func():
            loss = 0.0
            for batch_sample in self.dataloader:
                positive_sample = batch_sample[0]
                negative_tail = batch_sample[1]
                negative_sample = positive_sample.clone()
                negative_sample[:, 2] = negative_tail[:, 0]

                loss_batch = self.model_relation_learning.loss(
                    positive_sample, negative_sample)
                loss_batch.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss += loss_batch
            return loss
        fit_model(loss_func=loss_func,
                  model_name=self.config['relation_learning_model'],
                  max_iter=self.config.get('epoch_num', 50))

        self.train_embedding = self.model_relation_learning.entity_embedding.detach()

        self.train_classifier()

    def train_classifier(self,):
        if self.config['binary_or_multi'] == 'binary':
            self.train_single_type_classifier()
        else:
            self.train_multi_type_classifier()

    def train_single_type_classifier(self,):
        train_emb = self.train_embedding

        # 训练分类器
        self.clf = RandomForestClassifier(max_depth=8, random_state=0)
        X, Y = [], []
        for batch_sample in self.dataloader:
            # 正样本
            positive_sample = batch_sample[0]
            head, relation, tail = positive_sample[:,
                                                   0], positive_sample[:, 1], positive_sample[:, 2]
            embedding = np.concatenate(
                [train_emb[head], train_emb[tail]], axis=1)

            if type(X) == list:
                X = embedding
                Y = relation
            else:
                X = np.vstack((X, np.concatenate(
                    [train_emb[head], train_emb[tail]], axis=1)))
                Y = np.hstack((Y, relation))

            # 负样本
            negative_tail = batch_sample[1][:, 0]
            X = np.vstack((X, np.concatenate(
                [train_emb[head], train_emb[negative_tail]], axis=1)))
            Y = np.hstack((Y, torch.zeros_like(relation, dtype=torch.int)))

        self.clf.fit(X, Y)

    def train_multi_type_classifier(self,):
        train_emb = self.train_embedding

        # 训练分类器
        self.clf = RandomForestClassifier(max_depth=8, random_state=0)
        X, Y = [], []
        dataset = TrainDataset(self.positive_triples_all,
                               self.adj_train.shape[0], self.data.interaction_num, 1, 'tail-batch')
        for samples in dataset:
            positive_triple = samples[0]
            head, relation, tail = positive_triple[0], positive_triple[1], positive_triple[2]
            # negative_tail = samples[1][0]
            X.append(np.concatenate([train_emb[head], train_emb[tail]]))
            Y.append(relation)

        Y = label_binarize(
            Y, classes=np.arange(1, self.data.interaction_num))

        self.clf.fit(X, Y)

    def predict_test_S1(self,):
        test_emb = self.feature_map_to_embedding()
        train_emb = self.train_embedding

        score_predict_matrices = np.zeros(
            (self.data.interaction_num-1, self.test_drugs_num, self.train_drugs_num))
        for i in range(score_predict_matrices.shape[1]):
            for j in range(score_predict_matrices.shape[2]):
                interaction_idx = self.data.adj[self.test_drugs_indices[i],
                                                self.train_drugs_indices[j]]

                # 多类型预测中，不预测标签是0的，只预测存在DDI连边的
                if self.config['binary_or_multi'] == 'multi' and interaction_idx == 0:
                    continue

                ddi_sample = np.concatenate([test_emb[i], train_emb[j]])
                predict_probas = self.clf.predict_proba([ddi_sample])

                if self.config['binary_or_multi'] == 'multi':
                    for k in range(len(predict_probas)):
                        if predict_probas[k].shape[1] < 2:
                            predict_probas[k] = np.append(
                                predict_probas[k], [[0.0]], axis=1)

                    score_predict_matrices[:, i, j] = np.array(predict_probas)[
                        :, 0, 1]
                else:
                    score_predict_matrices[0, i, j] = predict_probas[0, 1]

        if self.config['binary_or_multi'] == 'binary':
            return score_predict_matrices[0, :, :]
        else:
            return score_predict_matrices

    def predict_test_S2(self,):
        test_emb = self.feature_map_to_embedding()

        score_predict_matrices = np.zeros(
            (self.data.interaction_num-1, self.test_drugs_num, self.test_drugs_num))
        for i in range(score_predict_matrices.shape[1]):
            for j in range(i+1, score_predict_matrices.shape[2]):
                interaction_idx = self.data.adj[self.test_drugs_indices[i],
                                                self.test_drugs_indices[j]]

                # 多类型预测中，不预测标签是0的，只预测存在DDI连边的
                if self.config['binary_or_multi'] == 'multi' and interaction_idx == 0:
                    continue

                ddi_sample = np.concatenate([test_emb[i], test_emb[j]])
                predict_probas = self.clf.predict_proba([ddi_sample])

                if self.config['binary_or_multi'] == 'multi':
                    for k in range(len(predict_probas)):
                        if predict_probas[k].shape[1] < 2:
                            predict_probas[k] = np.append(
                                predict_probas[k], [[0.0]], axis=1)

                    score_predict_matrices[:, i, j] = np.array(predict_probas)[
                        :, 0, 1]
                else:
                    score_predict_matrices[0, i, j] = predict_probas[0, 1]

        if self.config['binary_or_multi'] == 'binary':
            return score_predict_matrices[0, :, :]
        else:
            return score_predict_matrices


class LabelPropogation():
    def __init__(self, config, data):
        super(LabelPropogation, self).__init__()
        self.config = config
        self.data = data

        self.W = torch.FloatTensor(
            np.load('data/label_propogation_similarity_matrix.npz', allow_pickle=True)['data'])
        D = torch.diagflat(self.W.sum(1))
        self.W = torch.pinverse(D).matmul(self.W)

    def init(self, train_drugs_indices, test_drugs_indices):
        self.train_drugs_indices, self.test_drugs_indices = sorted(
            train_drugs_indices), sorted(test_drugs_indices)

        self.Y = torch.FloatTensor(self.data.adj.copy())
        self.Y[self.test_drugs_indices] = torch.zeros(self.data.drug_num)

    def train(self):
        """ F=(1-mu)(I-uW)^-1 Y """
        mu = 0.5
        I = torch.eye(self.W.shape[0])
        self.F = (1-mu)*torch.mm(torch.pinverse(I-mu*self.W), self.Y)

    def test(self,):
        if self.config['binary_or_multi'] == 'binary':
            if self.config['S1_or_S2'] == 'S1':
                return self.binary_test_S1()
            elif self.config['S1_or_S2'] == 'S2':
                raise RuntimeError('not support S2')
        else:
            raise RuntimeError('not support multi')

    def binary_test_S1(self):
        label_true, score_predict = [], []

        for i in range(len(self.test_drugs_indices)):
            for j in range(len(self.train_drugs_indices)):
                test_idx = self.test_drugs_indices[i]
                train_idx = self.train_drugs_indices[j]

                label_true.append(self.data.adj[test_idx, train_idx])
                score_predict.append(self.F[test_idx, train_idx])

        return binary_evaluation_result(label_true, score_predict)


class ColdStartDeepDDI(BaseModelTrain):
    def __init__(self, config, data):
        super(ColdStartDeepDDI, self).__init__(config, data)

        # self.data.adj = torch.LongTensor(self.data.adj)

        self.data.feature = Jaccard(self.data.feature)  # 计算蛋白特征相似性
        self.data.feature = PCA(n_components=500).fit_transform(
            self.data.feature)  # 降维
        self.data.feature = torch.FloatTensor(self.data.feature)

    def init(self, train_drugs_indices, test_drugs_indices):
        super(ColdStartDeepDDI, self).init(
            train_drugs_indices, test_drugs_indices)

        self.feature_train = torch.FloatTensor(
            self.data.feature[self.train_drugs_indices])

        self.model = torch.nn.Sequential(
            torch.nn.Linear(500*2, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(),
            torch.nn.Linear(256, self.data.interaction_num),
            torch.nn.Softmax(dim=1),
        )
        # self.model.to(device=self.config['device'])

    def train(self):
        # 取出所有的 有反应的ddi
        self.positive_triples_all = []
        for row in range(self.adj_train.shape[0]):
            for col in range(row+1, self.adj_train.shape[1]):
                interaction_idx = self.adj_train[row, col].item()
                if interaction_idx > 0:
                    self.positive_triples_all.append(
                        (row, interaction_idx, col))

        self.dataloader = DataLoader(
            TrainDataset(self.positive_triples_all,
                         self.adj_train.shape[0], self.data.interaction_num, 1, 'tail-batch'),
            batch_size=self.config['batch_size'],
            shuffle=True,
            collate_fn=TrainDataset.collate_fn
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(), weight_decay=0.001)

        def loss_func():
            loss = 0.0

            # 单类型
            if self.config['binary_or_multi'] == 'binary':
                for batch_sample in self.dataloader:
                    # 正样本
                    positive_sample = batch_sample[0]
                    drug1_feature = self.feature_train[positive_sample[:, 0], :]
                    drug2_feature = self.feature_train[positive_sample[:, 2], :]
                    positive_feature = torch.cat(
                        [drug1_feature, drug2_feature], dim=1)
                    positive_label = torch.ones(
                        positive_feature.shape[0], dtype=torch.long)

                    # 负样本
                    negative_tail = batch_sample[1][:, 0]
                    drug2_feature = self.feature_train[negative_tail, :]
                    negagive_feature = torch.cat(
                        [drug1_feature, drug2_feature], dim=1)
                    negagive_label = torch.zeros(
                        negagive_feature.shape[0], dtype=torch.long)

                    # 正负样本和标签合并
                    feature = torch.cat(
                        [positive_feature, negagive_feature], dim=0)
                    label = torch.cat([positive_label, negagive_label])

                    # 更新
                    output = self.model(feature)
                    output = torch.clamp(output, min=1e-8, max=1.0)
                    loss_batch = torch.nn.functional.nll_loss(
                        torch.log(output), label)
                    loss_batch.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    loss += loss_batch
            else:  # 多类型
                for (batch_sample, _, _, _) in self.dataloader:
                    label = batch_sample[:, 1]
                    drug1_feature = self.feature_train[batch_sample[:, 0], :]
                    drug2_feature = self.feature_train[batch_sample[:, 2], :]
                    feature = torch.cat([drug1_feature, drug2_feature], dim=1)

                    output = self.model(feature)
                    output = torch.clamp(output, min=1e-8, max=1.0)
                    loss_batch = torch.nn.functional.nll_loss(
                        torch.log(output), label)
                    loss_batch.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    loss += loss_batch
            if torch.isnan(loss):
                return np.inf
            return loss
        fit_model(loss_func=loss_func,
                  model_name='DeepDDI',
                  max_iter=self.config.get('epoch_num', 50))

    def predict_test_S1(self,):
        self.model.eval()

        score_predict_matrices = np.zeros(
            (self.data.interaction_num-1, self.test_drugs_num, self.train_drugs_num))
        for i in range(score_predict_matrices.shape[1]):
            for j in range(score_predict_matrices.shape[2]):
                idx1 = self.test_drugs_indices[i]
                idx2 = self.train_drugs_indices[j]

                interaction_idx = self.data.adj[idx1, idx2].item()

                # 多类型预测中，不预测标签是0的，只预测存在DDI连边的
                if self.config['binary_or_multi'] == 'multi' and interaction_idx == 0:
                    continue

                feature_cat = torch.cat(
                    [self.data.feature[idx1].view(1, -1), self.data.feature[idx2].view(1, -1)], dim=1)
                score = self.model(feature_cat).detach().numpy()[0, 1:]
                score_predict_matrices[:, i, j] = score

        if self.config['binary_or_multi'] == 'binary':
            return score_predict_matrices[0, :, :]
        else:
            return score_predict_matrices

    def predict_test_S2(self,):
        self.model.eval()

        score_predict_matrices = np.zeros(
            (self.data.interaction_num-1, self.test_drugs_num, self.test_drugs_num))
        for i in range(score_predict_matrices.shape[1]):
            for j in range(i+1, score_predict_matrices.shape[2]):
                idx1 = self.test_drugs_indices[i]
                idx2 = self.test_drugs_indices[j]

                interaction_idx = self.data.adj[idx1, idx2].item()

                # 多类型预测中，不预测标签是0的，只预测存在DDI连边的
                if self.config['binary_or_multi'] == 'multi' and interaction_idx == 0:
                    continue

                feature_cat = torch.cat(
                    [self.data.feature[idx1].view(1, -1), self.data.feature[idx2].view(1, -1)], dim=1)
                score = self.model(feature_cat).detach().numpy()[0, 1:]
                score_predict_matrices[:, i, j] = score

        if self.config['binary_or_multi'] == 'binary':
            return score_predict_matrices[0, :, :]
        else:
            return score_predict_matrices


class ColdStartDDIMDL(BaseModelTrain):
    def __init__(self, config, data):
        super(ColdStartDDIMDL, self).__init__(config, data)

        self.features = []
        for feature in self.data.features:
            feature = Jaccard(feature)
            feature = PCA(n_components=500).fit_transform(
                feature)  # 降维
            self.features.append(torch.FloatTensor(feature))

    def init(self, train_drugs_indices, test_drugs_indices):
        super(ColdStartDDIMDL, self).init(
            train_drugs_indices, test_drugs_indices)

        self.features_train = [feature[self.train_drugs_indices]
                               for feature in self.features]

        self.models = []
        for _ in range(len(self.features)):
            model = torch.nn.Sequential(
                torch.nn.Linear(500*2, 512),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(512),
                torch.nn.Dropout(),
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(256),
                torch.nn.Dropout(),
                torch.nn.Linear(256, self.data.interaction_num),
                torch.nn.Softmax(dim=1),
            )
            self.models.append(model)

    def train(self):
        # 取出所有的 有反应的ddi
        self.positive_triples_all = []
        for row in range(self.adj_train.shape[0]):
            for col in range(row+1, self.adj_train.shape[1]):
                interaction_idx = self.adj_train[row, col].item()
                if interaction_idx > 0:
                    self.positive_triples_all.append(
                        (row, interaction_idx, col))


        # 三元组按批次加载
        self.dataloader = DataLoader(
            TrainDataset(self.positive_triples_all,
                         self.adj_train.shape[0], self.data.interaction_num, 1, 'tail-batch'),
            batch_size=self.config['batch_size'],
            shuffle=True,
            collate_fn=TrainDataset.collate_fn
        )

        for i in range(len(self.features)):
            self.models[i] = self.train_model(
                self.models[i], self.dataloader, self.features_train[i])

    # 针对一种特征训练一个模型
    def train_model(self, model, dataloader, feature_train):
        optimizer = torch.optim.Adam(
            model.parameters(), weight_decay=0.001)

        def loss_func():
            loss = 0.0

            # 单类型
            if self.config['binary_or_multi'] == 'binary':
                for batch_sample in dataloader:
                    # 正样本
                    positive_sample = batch_sample[0]
                    drug1_feature = feature_train[positive_sample[:, 0], :]
                    drug2_feature = feature_train[positive_sample[:, 2], :]
                    positive_feature = torch.cat(
                        [drug1_feature, drug2_feature], dim=1)
                    positive_label = torch.ones(
                        positive_feature.shape[0], dtype=torch.long)

                    # 负样本
                    negative_tail = batch_sample[1][:, 0]
                    drug2_feature = feature_train[negative_tail, :]
                    negagive_feature = torch.cat(
                        [drug1_feature, drug2_feature], dim=1)
                    negagive_label = torch.zeros(
                        negagive_feature.shape[0], dtype=torch.long)

                    # 正负样本和标签合并
                    feature = torch.cat(
                        [positive_feature, negagive_feature], dim=0)
                    label = torch.cat([positive_label, negagive_label])

                    # 更新
                    output = model(feature)
                    output = torch.clamp(output, min=1e-8, max=1.0)
                    loss_batch = torch.nn.functional.nll_loss(
                        torch.log(output), label)
                    loss_batch.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    loss += loss_batch
            else:  # 多类型
                for (batch_sample, _, _, _) in dataloader:
                    label = batch_sample[:, 1]
                    drug1_feature = feature_train[batch_sample[:, 0], :]
                    drug2_feature = feature_train[batch_sample[:, 2], :]
                    feature = torch.cat([drug1_feature, drug2_feature], dim=1)

                    output = model(feature)
                    output = torch.clamp(output, min=1e-8, max=1.0)
                    loss_batch = torch.nn.functional.nll_loss(
                        torch.log(output), label)
                    loss_batch.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    loss += loss_batch
                    break
            if torch.isnan(loss):
                return np.inf
            return loss
        fit_model(loss_func=loss_func,
                  model_name='DDIMDL',
                  max_iter=self.config.get('epoch_num', 50))

        return model

    def predict_test_S1(self,):
        for i in range(len(self.models)):
            self.models[i].eval()

        score_predict_matrices = np.zeros(
            (self.data.interaction_num-1, self.test_drugs_num, self.train_drugs_num))
        for i in range(score_predict_matrices.shape[1]):
            for j in range(score_predict_matrices.shape[2]):
                idx1 = self.test_drugs_indices[i]
                idx2 = self.train_drugs_indices[j]

                interaction_idx = self.data.adj[idx1, idx2].item()

                # 多类型预测中，不预测标签是0的，只预测存在DDI连边的
                if self.config['binary_or_multi'] == 'multi' and interaction_idx == 0:
                    continue

                for k in range(len(self.models)):
                    feature_cat = torch.cat(
                        [self.features[k][idx1].view(1, -1), self.features[k][idx2].view(1, -1)], dim=1)
                    score = self.models[k](feature_cat).detach().numpy()[0, 1:]
                    # 将所有特征的预测结果累加起来
                    score_predict_matrices[:, i, j] += score

        score_predict_matrices /= len(self.models)

        if self.config['binary_or_multi'] == 'binary':
            return score_predict_matrices[0, :, :]
        else:
            return score_predict_matrices

    def predict_test_S2(self,):
        for i in range(len(self.models)):
            self.models[i].eval()

        score_predict_matrices = np.zeros(
            (self.data.interaction_num-1, self.test_drugs_num, self.test_drugs_num))
        for i in range(score_predict_matrices.shape[1]):
            for j in range(i+1, score_predict_matrices.shape[2]):
                idx1 = self.test_drugs_indices[i]
                idx2 = self.test_drugs_indices[j]

                interaction_idx = self.data.adj[idx1, idx2].item()

                # 多类型预测中，不预测标签是0的，只预测存在DDI连边的
                if self.config['binary_or_multi'] == 'multi' and interaction_idx == 0:
                    continue

                for k in range(len(self.models)):
                    feature_cat = torch.cat(
                        [self.features[k][idx1].view(1, -1), self.features[k][idx2].view(1, -1)], dim=1)
                    score = self.models[k](feature_cat).detach().numpy()[0, 1:]
                    # 将所有特征的预测结果累加起来
                    score_predict_matrices[:, i, j] += score
                    
        score_predict_matrices /= len(self.models)

        if self.config['binary_or_multi'] == 'binary':
            return score_predict_matrices[0, :, :]
        else:
            return score_predict_matrices
