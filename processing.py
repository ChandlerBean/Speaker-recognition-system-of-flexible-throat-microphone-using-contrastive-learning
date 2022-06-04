#预处理，包括滤波、预加重、端点检测
#利用傅里叶变换来滤波，失去了相位的信息，可以尝试不要滤波，因为mfcc本身就带有一组滤波器，或者采用频带滤波的函数
#麦克风数据和传感器数据是不是用同一套预处理
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft,ifft
import librosa
import pandas as pd
import wave
import xlrd

#展示函数，检查波形
def Show(y,fft_y,ifft_y,DDJC_y):
    plt.subplot(221)
    plt.plot(y)
    plt.subplot(222)
    plt.plot(fft_y[60:400])
    plt.subplot(223)
    plt.plot(ifft_y)
    plt.subplot(224)
    plt.plot(DDJC_y)
    plt.show()

#计算短时能量的函数
def Energy(y,n):
    energy = []
    L = len(y)
    i = 0
    while i+n < L:
        temp = y[i:i+n]
        energy.append(np.sum(temp**2))
        i += n
    return energy

#双门限端点检测算法
def DDJC(FFT_Data,rate):
    # 获得短时能量和短时过零率
    FL = 16  # 帧长
    E = Energy(FFT_Data, FL)
    E /= np.max(E)  # 能量归一化
    #R /= np.max(R)  # 过零率归一化
    Z, T1, T2, Start, End, T = 0.01, 0.01, 0.8, 0, len(E) - 1, len(E) #短时过零率阈值,两个短时能量阈值，起点和终点
    # 第一步，根据T2找起终点
    while (Start < End) and (E[Start] < T2): Start += 1
    while (Start < End) and (E[End] < T2): End -= 1
    #print(Start*FL,End*FL)
    # 第二步，补齐太短的
    LL = rate//FL * 0.8 #表示截取的长度，等于采样帧数的60%
    while End-Start < LL:
        if Start == 0: End += 1
        else:
            if End == T: Start -= 1
            else:
                if E[Start] > E[End]: Start -= 1
                else: End += 1
    # 第三步，裁剪太长的
    while End-Start > LL:
        if Start == 0: End -= 1
        else:
            if End == T: Start += 1
            else:
                if E[Start] < E[End]: Start += 1
                else: End -= 1
    Y = np.array(FFT_Data[Start * FL:End * FL])  # 输出结果
    return Y

#这里改带通滤波50-500Hz试试，或者直接MFCC提取的时候滤波
def My_filter(y,rate, frequency_curve):
    # 傅里叶滤波
    fft_y = fft(y[:])
    fft_y[:50] = 0
    fft_y[-50:] = 0
    fft_y[500:-500] = 0
    # 预加重
    for i in range(50, 500):
        fft_y[i] /= frequency_curve[i - 50]
        fft_y[-i] /= frequency_curve[i - 50]
    ifft_y = ifft(fft_y).astype("double")  # 这里的结果是复数，带有虚部，应该是相位，需要转float
    #DDJC_y = DDJC(ifft_y[:],rate) #端点检测，去掉柔性传感器滤波后的头和尾
    #Show(y,fft_y,ifft_y,DDJC_y)
    return ifft_y
    #return ifft_y[50:-50]

#预处理柔性传感数据集
def data_process(data_path, rate=1000):
    # frequency_curve = np.array(pd.read_excel(r"F:\frequency.xlsx", header=None))  # 自己拟合的频率响应曲线，50到500Hz
    # frequency_curve = frequency_curve[:, 0]  # 曲线是从50Hz开始记录的
    # 预处理柔性传感器数据
    flag = data_path[-1]  # 读取增广标志
    a = data_path.find("*")
    #num = int(data_path[a + 1])
    b = data_path.find("-")
    num = int(data_path[a + 1:b])
    #xlrd打开表格，它与pandas不一样的地方在于，它会把末尾的空单元格读为''而不是0，而且它读出来的list的内容属于string类型
    data = xlrd.open_workbook(data_path[:a])
    table = data.sheets()[0]
    temp = table.col_values(num)
    t = len(temp) - 1
    while temp[t] == '': t -= 1  # 去掉结尾多余的空字符，这里也很费时间
    temp = temp[:t + 1]
    temp = [float(a) for a in temp]
    temp = np.array(temp)
    '''#pandas打开表格
    data = np.array(pd.read_excel(data_path[:a], header=None).fillna(0))
    temp = data[:, num]
    t = data.shape[0] - 1
    while temp[t] == 0: t -= 1  # 去掉结尾多余的0，这里也很费时间
    temp = temp[:t + 1]'''
    # 不做滤波只做增广
    if flag == "1": temp = librosa.effects.time_stretch(temp, rate=1.1)
    elif flag == "2": temp = librosa.effects.time_stretch(temp, rate=0.9)
    elif flag == "3": temp = librosa.effects.pitch_shift(temp, rate, n_steps=1)
    elif flag == "4": temp = librosa.effects.pitch_shift(temp, rate, n_steps=-1)
    elif flag == "5": temp = temp * np.random.uniform(0.98, 1.02, temp.size)
    elif flag == "6": temp = temp * np.random.uniform(0.99, 1.01, temp.size)
    elif flag == "7": temp = librosa.effects.time_stretch(temp, rate=0.85)
    elif flag == "8": temp = librosa.effects.time_stretch(temp, rate=1.15)
    elif flag == "9": temp = librosa.effects.pitch_shift(temp, rate, n_steps=2)
    elif flag == "a": temp = librosa.effects.pitch_shift(temp, rate, n_steps=-2)
    elif flag == "b": temp = librosa.effects.pitch_shift(temp, rate, n_steps=-2)
    elif flag == "c": temp = librosa.effects.pitch_shift(temp, rate, n_steps=2)
    elif flag == "d": temp = librosa.effects.pitch_shift(temp, rate, n_steps=-3)
    elif flag == "e": temp = librosa.effects.pitch_shift(temp, rate, n_steps=3)
    # return My_filter(temp, temp.shape[0], frequency_curve)
    return temp

#预处理麦克风数据集
def data_process_1(data_path, rate=16000):
    flag = data_path[-1]
    f = wave.open(data_path[:-2], 'rb')
    params = f.getparams()
    nframes = params[3]
    wave_Data = f.readframes(nframes)
    temp = np.frombuffer(wave_Data, dtype=np.int16) * 1.0  # 使得其出现浮点型，从而方便后面的变速变调
    # 不做滤波只做增广
    if flag == "1": temp = librosa.effects.time_stretch(temp, rate=1.1)
    elif flag == "2": temp = librosa.effects.time_stretch(temp, rate=0.9)
    elif flag == "3": temp = librosa.effects.pitch_shift(temp, rate, n_steps=1)
    elif flag == "4": temp = librosa.effects.pitch_shift(temp, rate, n_steps=-1)
    elif flag == "5": temp = temp * np.random.uniform(0.98, 1.02, temp.size)
    elif flag == "6": temp = temp * np.random.uniform(0.99, 1.01, temp.size)
    elif flag == "7": temp = librosa.effects.time_stretch(temp, rate=0.85)
    elif flag == "8": temp = librosa.effects.time_stretch(temp, rate=1.15)
    elif flag == "9": temp = librosa.effects.pitch_shift(temp, rate, n_steps=2)
    elif flag == "a": temp = librosa.effects.pitch_shift(temp, rate, n_steps=-2)
    elif flag == "b": temp = librosa.effects.pitch_shift(temp, rate, n_steps=-2)
    elif flag == "c": temp = librosa.effects.pitch_shift(temp, rate, n_steps=2)
    elif flag == "d": temp = librosa.effects.pitch_shift(temp, rate, n_steps=-3)
    elif flag == "e": temp = librosa.effects.pitch_shift(temp, rate, n_steps=3)
    return temp

# 预处理平行数据集
def data_process_2(data_path, rate):
    '''frequency_curve = np.array(pd.read_excel(r"F:\dataset\20211223new\frequency.xlsx", header=None))  # 自己拟合的频率响应曲线，50到500Hz
        frequency_curve = frequency_curve[50:350, 0]  # 从100Hz开始，到400Hz，因为曲线是从50Hz开始记录的'''
    # 做一个从传感器数据到麦克风数据的路径转换，例如F:\NewDataset20220309\data\zwl\sensor\eight.xlsx*6-1 到 F:\NewDataset20220309\data\zwl\mic\eight6.wav-1
    flag = data_path[-1]
    t1 = data_path.find("sensor")
    t2 = data_path.find("*")
    path = data_path[:t1] + "mic" + "\\" + data_path[t1 + 7:t2 - 5] + "\\" + data_path[t1 + 7:t2 - 5] + data_path[t2 + 1] + ".wav-" + flag
    return data_process(data_path, 1000), data_process_1(path, 16000) #这里的采样率不同
    #return data_process(data_path, 1000)
