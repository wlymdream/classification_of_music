
from data_util.DataUtil import *
from data_util.TrainDataUtil import *
from scipy.io import wavfile
from scipy import fft

data_util_obj = DataUtil()
# 读取文件并使用傅里叶变换转换数据并保存
fft_data = data_util_obj.create_fft_data_and_save()
# 读取傅里叶变换后的数据并转成我们需要的x y
load_train_datas = data_util_obj.load_train_datas(fft_data)

train_data_util_obj = TrainDataUtil()
# 训练模型并写入文件中
train_data_util_obj.train_data(load_train_datas.get('x'), load_train_datas.get('y'))
# 从文件中加载模型
model_results = train_data_util_obj.open_model_file()

# 测试
tests_data_dir = '/Users/tests2.wav'
sample_rate, x = wavfile.read(tests_data_dir)
# 解决双通道音频…… 双通道表示的是，左耳是左声道，右耳是右声道 我们训练的模型的单声道的
X = np.reshape(x, (1, -1))[0]
test_data = abs(fft.fft(X, sample_rate)[:1000])

k = model_results.predict([test_data])
print(k)