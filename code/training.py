#模型训练
#蒸馏温度和阿尔法这两个超参数可以更新吗

from loading import mic_data_path, my_mic_data_path, sensor_data_path, parallel_dataset, MyDataset
from torch.utils.data import DataLoader
from feature import get_feature,get_feature_1,get_feature_2
import torch
import sys
import matplotlib.pyplot as plt
import time
import warnings
from CL2 import ContrastiveLoss
from model import StudentNet, LSTMNet, MyResnet, MyResnet18,LSTMNet18
import torch.nn.functional as F
from github_resnet import resnet18, resnet4, resnet50
from resnet import resnet20
#from losses import ContrastiveLoss, TripletLoss, OnlineContrastiveLoss
import numpy as np
from visualization import visualization
warnings.filterwarnings("ignore") #运行时不看警告

sys.path.append("..")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#评价教师模型的函数
def evaluate_accuracy_1(data_iter, net,device = torch.device('cuda' if torch.cuda.is_available()else 'cpu')):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        net.eval()
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                acc_sum += (net(X.to(device).float()).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            n += y.shape[0]
        net.train()
    return acc_sum / n

#训练教师模型的函数
def train_1(net, train_iter, test_iter, device, optimizer, num_epochs, save_path, save_name):
    net = net.to(device)
    print(save_name, "Model is training on ", device)
    loss = torch.nn.CrossEntropyLoss() #损失函数
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X= X.to(device).float()
            y = y.to(device).long() #这里要把标签转成tensor才能计算loss
            y_hat = net(X.float())
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy_1(test_iter, net)
        print('epoch %d, loss %.6f, train acc %.5f, test acc %.5f, time %.1f sec' % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
    torch.save(net.state_dict(), save_path+"\\" + save_name + "_model.pkl") #保存模型

#评价学生模型的函数
def evaluate_accuracy_2(data_iter, net1, net2, device = torch.device('cuda' if torch.cuda.is_available()else 'cpu')):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        #for X1, y in data_iter:
        net1.eval()
        for X1, X2, y in data_iter:
            if isinstance(net1, torch.nn.Module):
                y_hat = net1.forward_features(X1.to(device).float())
                y_hat = net2.fc(y_hat)
                acc_sum += (y_hat.argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            n += y.shape[0]
        net1.train()
    return acc_sum / n

#训练学生模型的函数
#def train_2(net1, net2, train_iter, optimizer,device, num_epochs, save_path, save_name, num=1, T=0.5):
def train_2(net1, train_iter, device, num_epochs, save_path, save_name, num=1, T=0.5, lr=0.001, margin=2.5):
    net1 = net1.to(device)
    #net2 = net2.to(device)
    print(save_name, "Model is training on ", device)

    # 冻结全连接层
    count = 0
    para_optim = []
    for k in net1.children():
        count += 1
        if count <= 6: #改层数
            for param in k.parameters():
                para_optim.append(param)
        else:
            for param in k.parameters():
                param.requires_grad = False
    optimizer = torch.optim.SGD(para_optim, lr=lr, momentum=0.9) #优化器

    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum ,train_l1_sum, train_l2_sum, train_acc_sum, n, start = 0.0, 0.0, 0.0, 0.0, 0, time.time()
        for X1, y in train_iter:
            bs = int(X1.size(0)/2)
            X1 = X1.to(device).float() #传感器数据
            #X2 = X2.to(device).float() #麦克风数据
            loss = ContrastiveLoss(batch_size=X1.size(0), temperature=T, num=num, margin=margin)
            y_hat = net1.forward_features(X1[:bs])
            soft_y = net1.forward_features(X1[bs:])
            l = loss(y_hat, soft_y, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            batch_count += 1
        print('epoch %d, loss %.6f, time %.1f sec' % (epoch + 1, train_l_sum / batch_count, time.time() - start))
    torch.save(net1.state_dict(), save_path + "\\" + save_name + "_model.pkl")  # 保存模型

#评价消融模型的函数
def evaluate_accuracy_3(data_iter, net3, device = torch.device('cuda' if torch.cuda.is_available()else 'cpu')):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        net3.eval()
        for X, y in data_iter:
            if isinstance(net3, torch.nn.Module):
                y_hat = net3(X.to(device).float())
                acc_sum += (y_hat.argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            n += y.shape[0]
        net3.train()
    return acc_sum / n

#训练消融模型的函数
def train_3(net3, train_iter, test_iter, device, num_epochs, save_path, save_name, lr=0.001,  weight_decay=0):
    net3 = net3.to(device)
    print(save_name, "Model is training on ", device)

    # 冻结前面层
    count = 0
    para_optim = []
    for k in net3.children():
        count += 1
        if count > 6: #改层数
            for param in k.parameters():
                para_optim.append(param)
        else:
            for param in k.parameters():
                param.requires_grad = False
    #optimizer = torch.optim.SGD(para_optim, lr=lr, momentum=0.9,  weight_decay=weight_decay) #优化器
    optimizer = torch.optim.Adam(para_optim, lr=lr, weight_decay=weight_decay)
    loss = torch.nn.CrossEntropyLoss() #损失函数
    batch_count = 0

    acc_train, acc_test, loss_train, acc_X = [],[],[],[]  #画图用

    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X= X.to(device).float()
            y = y.to(device).long() #这里要把标签转成tensor才能计算loss
            y_hat = net3(X.float())
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy_3(test_iter, net3)

        acc_X.append(epoch)
        acc_test.append(test_acc)
        acc_train.append(train_acc_sum / n)
        loss_train.append(train_l_sum / batch_count)

        print('epoch %d, loss %.6f, train acc %.5f, test acc %.5f, time %.1f sec' % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
    torch.save(net3.state_dict(), save_path+"\\" + save_name + "_model.pkl") #保存模型
    # 画图
    fig = plt.figure(1)
    ax1 = fig.add_subplot(111)
    ax1.plot(acc_X, acc_train, label="training accuracy")
    ax1.plot(acc_X, acc_test, label="testing accuracy")
    #ax1.legend()

    ax1.set_ylabel("accuracy")
    ax2 = ax1.twinx()
    ax2.plot(acc_X, loss_train, label="loss", c="red")
    ax2.set_ylabel("loss")
    ax2.set_xlabel("epoch")
    #ax2.legend()
    plt.savefig(save_path + "\\" + save_name + "_output_loss.jpg")
    plt.show()

if __name__ == "__main__":
    file_path = r"F:\NewDataset20220316\data"
    save_path = r"F:\NewDataset20220316\model\teacher"
    batch_size, lr, num_epochs, ratio, weight_decay = 32, 0.0001, 100, 0.8, 0 #bs必须是双数
    train_set, train_label, test_set, test_label = sensor_data_path(file_path, ratio)

    # #对比学习
    batch_size, lr, num_epochs,  = 160, 0.0001, 200
    train_data = MyDataset(data_set=train_set, data_label=train_label)  # 实例化
    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)  # 迭代器， num_workers=0表示单线程
    test_data = MyDataset(data_set=test_set, data_label=test_label)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)
    save_name = "test1"
    print(save_name)
    net1 = resnet20()
    train_2(net1, trainloader, device, num_epochs, save_path, save_name, lr=lr)

    #监督调优
    batch_size, lr, num_epochs = 32, 0.0005, 1000
    train_data = MyDataset(data_set=train_set, data_label=train_label)  # 实例化
    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)  # 迭代器， num_workers=0表示单线程
    test_data = MyDataset(data_set=test_set, data_label=test_label)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)
    net3 = resnet20()
    net3.load_state_dict(torch.load(r"F:\NewDataset20220316\model\teacher\test1_model.pkl"))
    save_name = "Test"
    print(save_name)
    train_3(net3, trainloader, testloader, device, num_epochs, save_path, save_name, lr=lr)

    #单独训练柔性传感器数据
    # save_name = "XiaoRong"
    # net2 = resnet20()
    # lr, num_epochs, ratio = 0.0005, 300, 0.8
    # optimizer = torch.optim.SGD(net2.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9) #优化器
    # train_1(net2, trainloader, testloader, device, optimizer, num_epochs, save_path, save_name)

    # 监督调优
    # file_path = r"F:\NewDataset20220316\data"
    # save_path = r"F:\NewDataset20220316\model\teacher"
    # batch_size, lr, num_epochs, ratio = 10, 0.000001, 3000, 0.8 #bs必须是双数
    # train_set, train_label, test_set, test_label = sensor_data_path(file_path, ratio)
    # train_data = MyDataset(data_set=train_set, data_label=train_label)  # 实例化
    # trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)  # 迭代器， num_workers=0表示单线程
    # test_data = MyDataset(data_set=test_set, data_label=test_label)
    # testloader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=0)
    # save_name = "TiaoYou_20220429_CL_A6" #"20220429_CL_A0", "20220429_CL_Ashift", "20220429_CL_Aspeed"
    # print(save_name)
    # net3 = resnet18(pretrained=False)
    # net3.load_state_dict(torch.load(r"F:\NewDataset20220316\model\teacher\20220429_CL_A6_model.pkl")) #"20220429_CL_A0", "20220429_CL_Ashift", "20220429_CL_Aspeed"
    # train_3(net3, trainloader, testloader, device, num_epochs, save_path, save_name, lr=lr)


    # 可视化
    # file_path = r"F:\NewDataset20220316\data"
    # ratio = 0.8
    # train_set, train_label, test_set, test_label = sensor_data_path(file_path, ratio)
    # train_data = MyDataset(data_set=train_set, data_label=train_label)  # 实例化
    # trainloader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=0)  # 迭代器， num_workers=0表示单线程
    # net5 = resnet18(pretrained=False)
    # net5.load_state_dict(torch.load(r"F:\NewDataset20220316\model\teacher\20220428_CL_A4_model.pkl"))
    # net5.eval()
    # temp = []
    # for X, y in trainloader:
    #     X = X.to(device).float()
    #     temp.append(net5.forward_features(X).detach().numpy())
    # temp = np.array(temp).reshape(-1,512)
    # visualization(temp, train_label)

    # M = [3.5]
    # for margin in M:
    #     net5 = resnet18(pretrained=False)
    #     net5.load_state_dict(torch.load(r"F:\NewDataset20220316\model\teacher\20220515_" + str(margin) + "_model.pkl"))
    #     net5.eval()
    #     temp = []
    #     for X, y in trainloader:
    #         X = X.to(device).float()
    #         temp.append(net5.forward_features(X).detach().numpy())
    #     temp = np.array(temp).reshape(-1,512)
    #     visualization(temp, train_label)

