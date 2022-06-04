from torch import nn
import numpy as np
import torch.nn.functional as F
from adaptive_avgmax_pool import AdaptiveAvgMaxPool2d
n_class = 20

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.drop_rate > 0.:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class LSTMNet(nn.Module):
    def __init__(self, block, layers, in_chs=3, drop_rate=0.0, block_drop_rate=0.0, global_pool='avg'):
        self.inplanes = 64
        self.drop_rate = drop_rate
        super(LSTMNet, self).__init__()
        self.conv1 = nn.Conv2d(in_chs, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.global_pool = AdaptiveAvgMaxPool2d(pool_type=global_pool)
        self.layer0 = self._make_layer(block, 64, layers[0], drop_rate=block_drop_rate) #64改16
        self.layer1 = nn.LSTM(input_size=64, hidden_size=512, batch_first=True, num_layers=1)
        self.layer2 = nn.Linear(512, 64)
        self.fc = nn.Linear(64, 20)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, drop_rate=0.):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, dilation, downsample, drop_rate)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward_features(self, x, pool=True):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.global_pool(x)
        x = self.layer0(x)
        x = x.view(x.size(0), 1, 64)
        x = self.layer1(x)
        x = x[0].view(x[0].size(0), -1)
        x = self.layer2(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.fc(x)
        return x

def LSTMNet18(pretrained=False, last_num=20, **kwargs):
    model = LSTMNet(BasicBlock, [2], **kwargs)
    return model


class TeacherNet(nn.Module):
    def __init__(self):
        super(TeacherNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size= 1)
        )
        self.Res_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size= 1)
        )
        self.Pool1 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size= 1)
        )
        self.Res_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size= 1)
        )
        self.Pool2 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        #self.lstm = nn.LSTM(input_size=512, dropout=0, hidden_size=128, num_layers=4, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(72, n_class),
            nn.Softmax(dim=1)
        )

    def forward(self, data):
        X = self.conv1(data)
        Y = self.Res_conv1(data)
        output = self.Pool1(X+Y)
        X = self.conv2(output)
        Y = self.Res_conv2(output)
        output = self.Pool2(X+Y)
        #output = self.lstm(output)
        output = output.view(-1,72) #批量为64，维度为72
        output = self.fc(output)
        return output

class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_class),
            nn.Softmax()
        )
    def forward(self, data):
        output = self.fc(data)
        return output


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.drop_rate > 0.:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class MyResnet(nn.Module):

    def __init__(self, block, layers, in_chs=3, num_classes=1000,
                 drop_rate=0.0, block_drop_rate=0.0, global_pool='avg',last_num=512):
        self.num_classes = num_classes
        self.inplanes = 64
        self.drop_rate = drop_rate
        super(MyResnet, self).__init__()
        self.conv1 = nn.Conv2d(in_chs, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.layer1 = self._make_layer(block, 64, layers[0], drop_rate=block_drop_rate) #64改16
        self.global_pool = AdaptiveAvgMaxPool2d(pool_type=global_pool)
        #self.fc = nn.Linear(64, 20) #self.num_features改64

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, drop_rate=0.):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, dilation, downsample, drop_rate)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward_features(self, x, pool=True):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        if pool:
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x

def MyResnet18( **kwargs):
    model = MyResnet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model