'''
@Author: your name
@Date: 2019-10-30 18:09:48
@LastEditTime : 2019-12-26 10:30:30
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: \models\BaseDataModel.py
'''

import numpy as np
from sklearn.decomposition import PCA
import csv
from sklearn.preprocessing import LabelEncoder


class DataModel():
    def __init__(self, data, adj_selector='adj_binary'):
        super(DataModel, self).__init__()
        self.load_data(data, adj_selector)

    def load_data(self, data, adj_selector):
        """ 导入数据

        args:
            data, dict, {'adj': array, ...}, 所有的数据

        得到: 
            self.adj, 邻接矩阵, 元素值为反应类别
            self.features, [ndarray(n,d_v),...] 多视图特征, 每个ndarray为一个视图下的特征
            self.view_num, 视图数量
            self.view_dims, 每个视图特征的维数
            self.drug_num, 药物数量
            self.interaction_num, 反应类型数量
         """

        self.adj = data[adj_selector]
        self.drug_num = self.adj.shape[0]
        self.drug_ids = data['drug_ids']

        features = [
            data["feature_dbp"],
            data["feature_structure"],
        ]
        self.load_features(features)
        self.interaction_num = self.adj.max().astype(np.int).item()+1


    def load_features(self, features):
        """ 导入特征, 并做降维预处理 """
        self.view_num = len(features)
        self.features = []
        self.view_dims = []
        pca = PCA(n_components=200)
        for feature in features:
            # feature = pca.fit_transform(feature)
            self.features.append(feature)
            self.view_dims.append(feature.shape[1])
        self.feature = self.features[0]

    def get_features(self, indices):
        """ 
        return:
            ret, [ndarray(n, 1), ndarray(n,2), ..., ndarray(n, V)], 所有视图下多涉及到药物的特征, 一个视图对应一个ndarray
         """

        ret = []
        for i in range(self.view_num):
            ret.append(self.features[i][indices, :])
        return ret
