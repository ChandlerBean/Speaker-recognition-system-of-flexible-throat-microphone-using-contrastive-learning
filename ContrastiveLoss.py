import torch
from torch import nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size,  device='cpu', temperature=0.5, num=1):
        super().__init__()
        self.batch_size = batch_size
        self.num = num
        self.register_buffer("temperature", torch.tensor(temperature).to(device))  # 超参数 温度
        self.register_buffer("negatives_mask", (~torch.eye(batch_size , batch_size , dtype=bool).to(device)).float())  # 主对角线为0，其余位置全为1的mask矩阵
        self.register_buffer("postives_mask", (torch.zeros(size=(batch_size , batch_size ), dtype=bool).to(device)).float())  # 主对角线为0，其余位置全为1的mask矩阵


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

        similarity_matrix = F.cosine_similarity(y_1, y_0, dim=2)  # simi_mat: (2*bs, 2*bs)
        nominator = self.postives_mask * torch.exp(similarity_matrix / self.temperature)  # 分子，正样本，进行矩阵乘法
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature) + nominator # 分母，正负样本，进行矩阵乘法

        num_F, num_N = 0, 0 # 阈值范围外的样本数，应该随着训练变大
        for i in range(batch_size):
            for j in range(batch_size):
                if (denominator[i][j]<1.5) and (denominator[i][j]>0) and (i!=j):
                    num_F += 1
                if (nominator[i][j]>5) and (i!=j):
                    num_N += 1
        print("Postives:", int(num_p/2), "Negtives:", int(num_n/2), "Near:", int(num_N/2), "Far:", int(num_F/2)) # 输出每个批量的情况

        # loss_partial = -torch.log( torch.sum(nominator, dim=1) / torch.sum(denominator, dim=1))  # 将分子分母分别累加，然后计算
        # loss = torch.sum(loss_partial) / (2 * self.batch_size)

        loss_partial = torch.log(torch.sum(denominator, dim=1) / torch.sum(nominator, dim=1))  # 将分子分母分别累加，然后计算
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss
