
import numpy as np
import torch
import sklearn.metrics as metrics


def Jaccard(matrix):
    matrix = np.mat(matrix)
    numerator = matrix * matrix.T
    denominator = np.ones(np.shape(matrix)) * matrix.T + \
        matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T
    return numerator / denominator


def symmetric_matrix(matrix):
    m_triu = torch.triu(matrix)
    return m_triu + m_triu.transpose(-2, -1) - torch.diag_embed(torch.diagonal(matrix, dim1=-2, dim2=-1))


def cosine_similarity(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def binary_evaluation_result(label_true, score_predict):
    auc = metrics.roc_auc_score(label_true, score_predict)
    precision, recall, _ = metrics.precision_recall_curve(
        label_true, score_predict)
    aupr = metrics.auc(recall, precision)

    score_sigmoid = 1/(1 + np.exp(-np.array(score_predict)))
    label_predict = [1 if score > 0.5 else 0 for score in score_sigmoid]

    acc = metrics.accuracy_score(label_true, label_predict)
    precision = metrics.precision_score(label_true, label_predict)
    recall = metrics.recall_score(label_true, label_predict)
    f1 = metrics.f1_score(label_true, label_predict)

    return {
        'acc': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'aupr': aupr,
    }


def multi_evaluation_result(label_true, score_predict):
    auc_list = []
    for relation_idx in range(label_true.shape[1]):
        fpr, tpr, _ = metrics.roc_curve(
            label_true[:, relation_idx], score_predict[:, relation_idx])
        auc = metrics.auc(fpr, tpr)
        if np.isnan(auc):
            auc = 0.0
        auc_list.append(auc)
    auc = np.mean(auc_list)

    # auc2 = metrics.roc_auc_score(
    #     label_true, score_predict, average='macro')
    aupr_list = []
    for relation_idx in range(label_true.shape[1]):
        precision, recall, _ = metrics.precision_recall_curve(
            label_true[:, relation_idx], score_predict[:, relation_idx])
        aupr = metrics.auc(recall, precision)
        if np.isnan(aupr):
            aupr = 0.0
        aupr_list.append(aupr)
    aupr = np.mean(aupr_list)

    label_true = [np.argmax(v) for v in label_true]
    label_predict = [np.argmax(v) for v in score_predict]
    acc = metrics.accuracy_score(label_true, label_predict)
    precision_micro = metrics.precision_score(label_true, label_predict,average='micro')
    precision_macro = metrics.precision_score(label_true, label_predict,average='macro')
    recall_micro = metrics.recall_score(label_true, label_predict,average='micro')
    recall_macro = metrics.recall_score(label_true, label_predict,average='macro')
    f1_micro = metrics.f1_score(label_true, label_predict,average='micro')
    f1_macro = metrics.f1_score(label_true, label_predict,average='macro')

    print(123)
    return {
        'acc': acc,
        'precision_micro': precision_micro,
        'precision_macro': precision_macro,
        'recall_micro': recall_micro,
        'recall_macro': recall_macro,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'auc': auc,
        'aupr': aupr,
        'auc_per_type': auc_list,
        'aupr_per_type': aupr_list,
    }


def fit_model(loss_func, model_name='model', max_iter=1000):
    """ 训练模型， 统计损失值大于最优损失值的次数（bad_counter），超过设定次数后停止"""
    loss_values = []
    best = np.inf
    best_epoch = 0

    for epoch in range(max_iter):
        loss = loss_func()
        loss_values.append(loss)

        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == 3:
            print('epoch num: {}'.format(epoch))
            break

        if epoch % 10 == 0:
            print('epoch: {}, loss: {}'.format(epoch, loss))

    print('epoch num: {}'.format(max_iter))


class UnionFind(object):
    """ 并查集，实现查找图节点的群组 """

    id = []
    count = 0
    sz = []

    def __init__(self, n):
        self.count = n
        i = 0
        while i < n:
            self.id.append(i)
            self.sz.append(1)  # inital size of each tree is 1
            i += 1

    def connected(self, p, q):
        if self.find(p) == self.find(q):
            return True
        else:
            return False

    def find(self, p):
        while (p != self.id[p]):
            p = self.id[p]
        return p

    def union(self, p, q):
        idp = self.find(p)
        idq = self.find(q)
        if not self.connected(p, q):
            if (self.sz[idp] < self.sz[idq]):
                self.id[idp] = idq

                self.sz[idq] += self.sz[idp]
            else:
                self.id[idq] = idp
                self.sz[idp] += self.sz[idq]

            self.count -= 1


def drug_emb_from_feature_emb(drug_feature, feature_embedding):
    """ 从蛋白特征嵌入获得药物的嵌入向量 """

    if type(drug_feature) == np.ndarray:
        return np.matmul(drug_feature, feature_embedding)
    else:
        return drug_feature.matmul(feature_embedding)