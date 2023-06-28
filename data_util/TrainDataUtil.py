import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
from data_util.DataUtil import *
from scipy.io import wavfile
from scipy import fft
import pprint
class TrainDataUtil:

    '''
        训练模型并保存
    '''
    def train_data(self, x, y):
        # ovr - > over-vs-rest 会拆成多个二分类进行计算 它比较适用于一种类型会存在哪几种分类里 比如一部电影可以有惊悚和悬疑同时存在的
        # multinomial softmax多分类 它比较适用于一种很明确的分类 比如一种生物是属于 动物 人 还是植物
        # solver = sag 随机梯度下降
        # max_iter 迭代次数
        logistic_obj = LogisticRegression(multi_class='multinomial', solver='sag', max_iter=10000)
        # 导入数据并训练模型
        logistic_obj.fit(x, y)
        print('start writing model ……')
        # 以写入二进制的模式打开文件
        output = open('model.pkl', 'wb')
        # 保存
        pickle.dump(logistic_obj, output)
        print('save writing model ……')
        output.close()

    '''
        以只读的方式打开模型文件
    '''
    def open_model_file(self):
        # 以只读的方式打开文件
        print('open model pkl')
        read_out_put = open('model.pkl', 'rb')
        model_result = pickle.load(read_out_put)
        print('load  model pkl finished')
        read_out_put.close()
        return model_result





