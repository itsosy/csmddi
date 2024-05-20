'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-06-02 10:09:30
@LastEditTime : 2020-01-12 15:36:01
@LastEditors  : Please set LastEditors
'''

import sys
import os
import datetime
import math
import numpy as np
import importlib
import torch
import sklearn
from sklearn.utils import shuffle

from models.DataModel import DataModel
from models.feature_learning import PLSR, GBR


def cross_validation(config):

    # 导入数据
    data = np.load(config['data_path'],
                   allow_pickle=True)['data'].item()
    data = DataModel(data, adj_selector=config['adj'])

    # 导入模型
    module = importlib.import_module('models.models_train_and_test')
    model = getattr(
        module, config['model_name'])(config, data)

    # print('--------------- cv start ------------------')
    drug_indices = list(range(data.drug_num))
    if 'shuffle_drug' in config and config['shuffle_drug']:
        drug_indices = shuffle(drug_indices)
    cv_batch = math.ceil(data.drug_num / config['cv'])
    result = []
    for cv_i in range(config['cv']):
        print('\n--------------- cv NO.{}: ---------------'.format(cv_i))
        print('time: {}'.format(datetime.datetime.now()))

        # 测试集和训练集
        test_drugs_indices = drug_indices[cv_i *
                                          cv_batch: (cv_i + 1) * cv_batch]
        train_drugs_indices = list(
            set(drug_indices) - set(test_drugs_indices))

        model.init(train_drugs_indices, test_drugs_indices)
        model.train()
        res_test = model.test()
        result.append(res_test)

        print(res_test)
        print('time: {}'.format(datetime.datetime.now()))

    # 求10次交叉验证的平均
    ret = {}
    for key in result[0].keys():
        res_list = [res[key] for res in result]
        ret[key] = (np.mean(res_list, axis=0), np.var(res_list, axis=0))

    return ret


def compute_feature_mapping_loss(config):
    # 导入数据
    data = np.load(config['data_path'],
                   allow_pickle=True)['data'].item()
    data = DataModel(data, adj_selector=config['adj'])

    # 导入模型
    module = importlib.import_module('models.models_train_and_test')
    model = getattr(
        module, config['model_name'])(config, data)

    drug_indices = list(range(data.drug_num))
    model.init(drug_indices, [])
    model.train()

    np.savez_compressed('analysis/data_RESCAL_Symmetric_trained_embedding.npz',
                        data={
                            'adj_multi': model.adj_list,
                            'drug_embedding': model.train_embedding,
                            'relation_embedding': model.relation_matrices,
                        })

    # score_predict_matrices = np.matmul(
    #     np.matmul(model.train_embedding, model.relation_matrices), model.train_embedding.T)
    # np.savez_compressed('analysis/data_RESCAL_case_study.npz',
    #                     data=score_predict_matrices)

    # feature = data.feature
    # embedding = model.train_embedding

    # plsr = PLSR()
    # plsr.fit(feature, embedding)
    # embedding_pred = plsr.predict(feature)
    # print('plsr mapping 平方和: {}'.format(np.sum((embedding-embedding_pred)**2)))
    # loss = sklearn.metrics.mean_squared_error(embedding, embedding_pred)
    # print("plsr mapping loss: {}".format(loss))

    # gbr = GBR()
    # gbr.fit(feature, embedding)
    # embedding_pred = gbr.predict(feature)
    # print('gbr mapping 平方和: {}'.format(np.sum((embedding-embedding_pred)**2)))
    # loss = sklearn.metrics.mean_squared_error(embedding, embedding_pred)
    # print("gbr mapping loss: {}".format(loss))


if __name__ == "__main__":

    config = {
        'data_path': 'data/drugbank_v5_stanfordnlp.npz',
        'adj': 'adj_multi',
        'binary_or_multi': 'multi',
        'S1_or_S2': 'S1',

        'model_name': 'ColdStartRelationLearning',
        'relation_learning_model': 'TransE',

        'map_model': 'PLSR',

        'drug_hidden_embedding_dim': 200,
        'batch_size': 200,
        'learning_rate': 0.01,
        'epoch_num': 1000,

        'cv': 10,
        'shuffle_drug': True,

        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    for model_name in ['ColdStartDDIMDL']:
        for binary_or_multi in ['multi', 'binary']:
            for S1_or_S2 in ['S1', 'S2']:
                config.update({
                    'model_name': model_name,
                    'adj': 'adj_{}'.format(binary_or_multi),
                    'binary_or_multi': binary_or_multi,
                    'S1_or_S2': S1_or_S2
                })
                print(config)
                res = cross_validation(config)
                print("\n\n")
                for key in res:
                    if isinstance(res[key][0], float):
                        print('{}: {:.4f} +/- {:.4f}'.format(key,
                                                             res[key][0], res[key][1]))
                np.savez_compressed('analysis/result_{}_type_{}_{}_{}.npz'.format(
                    config['binary_or_multi'], config['S1_or_S2'], config['model_name'], config['map_model']), data=res)
