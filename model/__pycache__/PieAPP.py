import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
'''
        此版本就是前面用PieAPP的网络 无concat 后面直接用conv11的输出结果 再后面和Wa完全一样
'''


class PieAPP(nn.Module):
    def __init__(self, batch_size, num_patches, weighted_average=True):
        super(PieAPP, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)

        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv7 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv8 = nn.Conv2d(256, 256, 3, padding=1)

        self.conv9 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv10 = nn.Conv2d(256, 512, 3, padding=1)

        self.conv11 = nn.Conv2d(512, 512, 3, padding=1)

        self.ref_score_subtract = nn.Linear(1, 1)

        self.dropout = nn.Dropout()
        self.weighted_average = weighted_average

        self.batch_size = batch_size
        self.num_patches = num_patches  # 32 / 64

        # 参考Wa-FRNet
        self.fc1_q = nn.Linear(512*3, 512)
        self.fc2_q = nn.Linear(512, 1)
        self.fc1_w = nn.Linear(512*3, 512)
        self.fc2_w = nn.Linear(512, 1)

    def extract_features(self, x):
        # 以下shape省略32 32指的是patch数量 如[32, 64, 32, 32]
        # patch数量除了32还有可能是64 为啥？？？
        h = F.relu(self.conv1(x))  # 64,32,32
        h = F.relu(self.conv2(h))  # 64,32,32
        h = F.max_pool2d(h, 2)  # 64,16,16
        x3 = F.relu(self.conv3(h))  # 64,16,16

        h = F.relu(self.conv4(x3))  # 128,16,16
        h = F.max_pool2d(h, 2)  # 128,8,8

        x5 = F.relu(self.conv5(h))  # 128,8,8
        h = F.relu(self.conv6(x5))  # 128,8,8
        h = F.max_pool2d(h, 2)  # 128,4,4

        x7 = F.relu(self.conv7(h))  # 256,4,4
        h = F.relu(self.conv8(x7))  # 256,4,4
        h = F.max_pool2d(h, 2)  # 256,2,2

        x9 = F.relu(self.conv9(h))  # 256,2,2
        h = F.relu(self.conv10(x9))  # 512,2,2
        h = F.max_pool2d(h, 2)  # 512,1,1

        x11 = F.relu(self.conv11(h))  # 32,512,1,1
        res = x11.view(-1, 512)     # 32,512

        return res

    def forward(self, data):
        """
        :param data: distorted and reference patches of images
        :return: quality of images/patches
        """
        # 一个batch中的原图、参考图
        x, x_ref = data     # [batchsize, patch数量, 3, 32, 32]
        batch_size = x.size(0)
        n_patches = x.size(1)  # 32
        if self.weighted_average:
            q = torch.ones((batch_size, 1), device=x.device)
        else:
            q = torch.ones((batch_size * n_patches, 1), device=x.device)

        for i in range(batch_size):
            # x[i] - [patch数, 3,32,32]
            h = self.extract_features(x[i])  # [32， 512]
            h_ref = self.extract_features(x_ref[i])  # [32, 512]
            f_cat = torch.cat((h - h_ref, h, h_ref), 1)  # [32, 512*3]

            # f_inter = f_cat  # 临时保存cat变量

            f = F.relu(self.fc1_q(f_cat))     # f-[32, 512]
            f = self.dropout(f)
            f = self.fc2_q(f)   # f-[32, 1] 即32个patch每个一个分数

            if self.weighted_average:
                w = F.relu(self.fc1_w(f_cat))  # [32, 512*3] -> [32, 512]
                w = self.dropout(w)
                w = F.relu(self.fc2_w(w)) + 0.000001 # small constant # [32, 512] -> [32, 1]
                q[i] = torch.sum(f * w) / torch.sum(w)  # 加权平均 / 权重归一化
            else:
                q[i*n_patches:(i+1)*n_patches] = h

            # import pdb;pdb.set_trace();

        return q

