'''
@Author: your name
@Date: 2019-11-03 20:55:11
@LastEditTime : 2019-12-29 11:56:17
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: /models/feature_learning.py
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from .models_train_and_test import fit_model


class PLSR():
    """ 偏最小二乘回归 """

    def fit(self, X, Y):
        self.model = PLSRegression(n_components=Y.shape[1])
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)
