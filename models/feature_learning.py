'''
@Author: your name
@Date: 2019-11-03 20:55:11
@LastEditTime : 2019-12-29 11:56:17
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: /ddi_classification/models/feature_learning.py
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from .models_train_and_test import fit_model


class DNN():
    """ 最基本的多层神经网络 """

    def fit(self, X, Y):
        input_dim = X.shape[1]
        output_dim = Y.shape[1]
        self.model = torch.nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(512),
            nn.LayerNorm(output_dim),
            nn.Dropout(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(256),
            nn.LayerNorm(output_dim),
            nn.Dropout(),
            nn.Linear(output_dim, output_dim),
        )

        X = torch.tensor(X, dtype=torch.float)
        Y = torch.tensor(Y, dtype=torch.float)

        optimizer = torch.optim.Adam(
            self.model.parameters(), weight_decay=0.001)

        def loss_func():
            optimizer.zero_grad()
            pred = self.model(X)
            loss = F.mse_loss(pred, Y)
            loss.backward()
            optimizer.step()
            return loss
        fit_model(loss_func=loss_func, model_name='feature_map_DNN')

    def predict(self, X):
        return self.model(torch.tensor(X, dtype=torch.float)).detach().numpy()


class PLSR():
    """ 偏最小二乘回归 """

    def fit(self, X, Y):
        self.model = PLSRegression(n_components=Y.shape[1])
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)


class GBR():
    """ 梯度提升回归，GBDT """

    def fit(self, X, Y):
        self.models = []
        for i in range(Y.shape[1]):
            gbr = GradientBoostingRegressor()
            gbr.fit(X, Y[:, i])
            self.models.append(gbr)

    def predict(self, X):
        Y = np.zeros((X.shape[0], len(self.models)))
        for i in range(Y.shape[1]):
            Y[:, i] = self.models[i].predict(X)
        return Y
