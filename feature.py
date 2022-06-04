#提取特征
#注意两种特征的长度不一样
import python_speech_features
import numpy as np
from processing import data_process, data_process_1, data_process_2

#获取柔性传感器数据特征
def get_feature(data,rate=1000):
    # data_path是样本路径，rate是采样率，dim是特征维度
    mfcc = python_speech_features.mfcc(data, rate, winlen=0.025, winstep=0.01, nfft=1024) #检查数据维度和是否转置
    # deltas1 = python_speech_features.delta(mfcc, 1)
    # deltas2 = python_speech_features.delta(mfcc, 2)
    # features = np.hstack((mfcc, deltas1))
    # features = np.hstack((features, deltas2))
    #return mfcc.flatten()
    return np.resize(mfcc, (3,26,26))

#提取麦克风数据特征
def get_feature_1(data,rate):
    # data_path是样本路径，rate是采样率，dim是特征维度
    mfcc = python_speech_features.mfcc(data, rate, winlen=0.025, winstep=0.01, nfft=1024) #检查数据维度和是否转置
    # deltas1 = python_speech_features.delta(mfcc, 1)
    # deltas2 = python_speech_features.delta(mfcc, 2)
    # features = np.hstack((mfcc, deltas1))
    # features = np.hstack((features, deltas2))
    return np.resize(mfcc, (3,26,26))

#提取麦克风和传感器数据的mfcc特征，及其一二阶导
def get_feature_2(data_path,rate=1000):
    #data_path是样本路径，rate是采样率，dim是特征维度
    sensor_data, mic_data = data_process_2(data_path,rate)
    sensor_feature = get_feature(sensor_data, rate = 1000)
    mic_feature =  get_feature_1(mic_data, rate = 16000)
    return (sensor_feature, mic_feature)
    # sensor_data = data_process_2(data_path,rate)
    # return get_feature_1(sensor_data,rate)
