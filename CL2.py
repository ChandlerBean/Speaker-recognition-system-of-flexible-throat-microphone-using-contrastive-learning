import torch
from torch import nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size,  device='cpu', temperature=0.5, num=1, margin=2.5):
        super().__init__()
        self.batch_size = batch_size
        self.margin = margin
        self.num = num
        self.register_buffer("temperature", torch.tensor(temperature).to(device))  # 超参数 温度
        self.register_buffer("negatives_mask", (~torch.eye(batch_size, batch_size, dtype=bool).to(device)).float())  # 主对角线为0，其余位置全为1的mask矩阵
        self.register_buffer("postives_mask", (torch.zeros(size=(batch_size, batch_size), dtype=bool).to(device)).float())  # 主对角线为0，其余位置全为1的mask矩阵


    def forward(self, emb_i, emb_j, y):  # emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到

        batch_size = self.batch_size
        num_n, num_p = 0, 0 # 正负样本数量
        for i in range(batch_size):
            for j in range(batch_size):
                if (y[i] == y[j]) and (i!=j):
                    self.negatives_mask[i][j] = 0 #标记负样本
                    self.postives_mask[i][j] = 1 #标记正样本
                    num_p += 1
                elif i != j:
                    num_n += 1

        z_i = F.normalize(emb_i, dim=1)  # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)  # (bs, dim)  --->  (bs, dim)
        representations = torch.cat([z_i, z_j], dim=0) # repre: (2*bs, dim)
        y_1 = representations.unsqueeze(1)# 将representations分别转换为列向量和行向量
        y_0 = representations.unsqueeze(0)

        #另一条计算公式
        distance_matrix = F.pairwise_distance(y_1, y_0, p=2)
        postives = self.num * self.postives_mask * (distance_matrix / self.temperature)  # 分子，正样本，进行对应元素乘法
        negatives = self.negatives_mask * (distance_matrix / self.temperature)  # 分母，负样本，进行对应元素乘法
        sum_p = torch.sum(postives.pow(2), dim=1)

        #实现公式的第二部分
        Far = self.margin * self.negatives_mask - negatives
        Near = self.margin * self.postives_mask - postives
        num_F, num_N = 0, 0 # 阈值范围外的样本数，应该随着训练变大
        for i in range(batch_size):
            for j in range(batch_size):
                if (Far[i][j]<0) and (i!=j):
                    Far[i][j] = 0
                    num_F += 1
                if (Near[i][j]>1.5) and (postives[i][j]!=0) and (i!=j):
                    num_N += 1
        #print("Postives:", int(num_p/2), "Negtives:", int(num_n/2), "Near:", int(num_N/2), "Far:", int(num_F/2)) # 输出每个批量的情况

        sum_n = torch.sum(Far.pow(2) , dim=1)
        loss_partial = sum_p + sum_n   # 将分子分母分别累加，然后计算
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss
