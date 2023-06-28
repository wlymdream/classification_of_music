# 做EDA 数据的探索和分析
# fft 快速傅里叶变换
import numpy as np
from scipy import fft
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import pandas as pd
import os


class DataUtil:

    '''
        利用傅里叶变换把音源用画图的形式保存下来，为了后期提取特征
    '''
    def create_fft_data_and_save(self):
        print('start read file……')
        os_files = []
        music_type = []
        train_file_data_dir_list = []
        path = '/Users/wly/AI_PyCharm_Project_Test/music_data'
        os_dir = os.listdir(path)
        data_key = {}
        i = 0
        for single_file in os_dir:
            # 如果不是文件夹的话 不遍历
            if os.path.isdir(path + '/' + single_file) == False:
                continue

            music_type.append(single_file)
            os_dir_files = os.listdir(path + '/' + single_file)
            data_key['music_type'] = single_file
            if os_files != os_dir_files:
                for single_file_dir in os_dir_files:
                    if single_file_dir.startswith('.') == True:
                        continue
                    elif single_file_dir.startswith('converted') == True:
                        files = os.listdir(path + '/' + single_file + '/' + single_file_dir)
                        for single_file1 in files:
                            final_file_path = path + '/' + single_file + '/' + single_file_dir + '/' + single_file1

                            simple_rate, data_x = wavfile.read(final_file_path)
                            # 对所有的点进行傅里叶变换 如果把采样率simpel_rate加进去的话，回头我们在对测试集数据进行测试的时候，也需要把采样率传进去
                            fft_features = abs(fft.fft(data_x)[:1000])
                            file_name = single_file1[0: len(single_file1) - 7]
                            save_dir = '/Users/wly/AI_PyCharm_Project_Test/train_fft_data/' + file_name + '.fft'
                            np.save(save_dir, fft_features)
                            train_file_data_dir_list.append(save_dir + '.npy')
            i += 1
        print('save  fft  file finished ……')
        return {'music_type': music_type, 'train_data_dir_list': train_file_data_dir_list}

    '''
        读取傅里叶变换后的数据集，将其转换为x和y
    '''
    def load_train_datas(self, music_data_dirs):
        X = []
        y= []
        music_type_list = music_data_dirs.get('music_type')
        train_data_dir_list = music_data_dirs.get('train_data_dir_list')

        for music_type in music_type_list:
            train_datas = [x for x in train_data_dir_list if x.find(music_type)]
            for train_data in train_datas:
                fft_features = np.load(train_data)
                X.append(fft_features)
                y.append(music_type_list.index(music_type))
        print('parse to x y ')
        return {'x': np.array(X), 'y': np.array(y)}












# 画图主要是感受一下不同曲风的时间频率图 将音乐文件用傅里叶变换展示出来
# data_util_obj = DataUtil()
# datas = data_util_obj.create_fft_data_and_save()
# f = data_util_obj.load_train_datas(datas)
# bules_data_result = data_util_obj.read_file('blues', '00000')
# plt.figure(num=None, figsize=(18, 9), dpi=80, facecolor='w', edgecolor='k')
# plt.subplot(6, 3, 1)
# data_util_obj.set_drawing_graph_config(bules_data_result, 'blues00000')
#
# jazz_data_result = data_util_obj.read_file('jazz', '00001')
# plt.subplot(6, 3, 2)
# data_util_obj.set_drawing_graph_config(jazz_data_result, 'jazz000001')
#
# classical_data_result = data_util_obj.read_file('classical', '00001')
# plt.subplot(6, 3, 3)
# data_util_obj.set_drawing_graph_config(classical_data_result, 'classical00001')
#
# classical_data_result_1 = data_util_obj.read_file('classical', '00005')
# plt.subplot(6, 3, 4)
# data_util_obj.set_drawing_graph_config(classical_data_result_1, 'classical00005')
#
# bules_data_result_2 = data_util_obj.read_file('blues', '00003')
# plt.subplot(6, 3, 5)
# data_util_obj.set_drawing_graph_config(bules_data_result_2, 'blues00003')
#
# pop_data_result = data_util_obj.read_file('pop', '00000')
# plt.subplot(6, 3, 6)
# data_util_obj.set_drawing_graph_config(pop_data_result, 'pop00000')
#
# pop_data_result_1 = data_util_obj.read_file('pop', '00009')
# plt.subplot(6, 3, 7)
# data_util_obj.set_drawing_graph_config(pop_data_result_1, 'pop00009')
#
# blues_data_result_6 = data_util_obj.read_file('blues', '00006')
# plt.subplot(6, 3, 8)
# data_util_obj.set_drawing_graph_config(blues_data_result_6, 'bules00006')
#
# classical_data_result_19 = data_util_obj.read_file('classical', '00009')
# plt.subplot(6, 3, 9)
# data_util_obj.set_drawing_graph_config(classical_data_result_19, 'classical00009')
#
# pad图之间的间隔大小w_pad宽的间隔大小 h_pad高的间隔大小
# plt.tight_layout(pad=0.4, w_pad=1, h_pad=1)
# plt.show()

# read_result = data_util_obj.read_file_by_dir('/Users/wly/AI_PyCharm_Project_Test/music_data/metal/converted/metal.00000.au.wav')
# sample_rate = read_result.get('simple_rate')
# x = read_result.get('x')
# plt.figure(num=None, figsize=(9, 6), dpi=80, facecolor='w', edgecolor='k')
# plt.subplot(2, 2, 1)
# plt.title('metal-')
# plt.xlabel('time')
# plt.ylabel('frequency')
# specgram(x, Fs=sample_rate, xextent=(0, 30))
#
# plt.subplot(2, 2, 2)
# plt.xlabel('frequency')
# plt.xlim(0,3000)
# plt.title('metal-')
# plt.ylabel('amplitude')
# # 对x进行傅里叶变换  abs 绝对值
# plt.plot(fft.fft(x, sample_rate))


# data_result = data_util_obj.read_file_by_dir('/Users/wly/AI_PyCharm_Project_Test/music_data/pop/converted/pop.00000.au.wav')
#
# sample_rate_1, x_1 = data_result.get('simple_rate'), data_result.get('x')
# plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
# plt.subplot(2, 1, 1)
# plt.xlabel('times')
# plt.ylabel('frequency')
# # 站在时域的角度
# specgram(x_1, Fs=sample_rate_1, xextent=(0, 30))
#
# plt.subplot(2, 1, 2)
# plt.xlabel('frequency')
# plt.ylabel('amplitude')
# plt.plot(fft.fft(x_1, sample_rate_1))
# plt.show()




