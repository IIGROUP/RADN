import torch
from torch import nn
import torch.nn.functional as F
from .BlurPooling import *
from .L2Pooling import *
from torch.nn import init


# WResNet
class WResNet(nn.Module):
    # pooling: 0 - downsampling with 2-stride conv instead of pooling layers (void)
    #          1 - downsampling with blur pooling layers
    #          2 - downsampling with l2 pooling layers
    def __init__(self, weighted_average=True, pooling=0):
        super(WResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)

        if pooling == 0:
            self.conv5 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        elif pooling == 1:
            self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
            self.bp1 = BlurPooling(128)
        elif pooling == 2:
            self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
            self.lp1 = L2pooling(channels=128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv7 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv8 = nn.Conv2d(128, 128, 3, padding=1)

        if pooling == 0:
            self.conv9 = nn.Conv2d(128, 256, 3, padding=1, stride=2)
        elif pooling == 1:
            self.conv9 = nn.Conv2d(128, 256, 3, padding=1)
            self.bp2 = BlurPooling(256)
        elif pooling == 2:
            self.conv9 = nn.Conv2d(128, 256, 3, padding=1)
            self.lp2 = L2pooling(channels=256)
        self.conv10 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv11 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv12 = nn.Conv2d(256, 256, 3, padding=1)

        if pooling == 0:
            self.conv13 = nn.Conv2d(256, 256, 3, padding=1, stride=2)
        elif pooling == 1:
            self.conv13 = nn.Conv2d(256, 256, 3, padding=1)
            self.bp3 = BlurPooling(256)
        elif pooling == 2:
            self.conv13 = nn.Conv2d(256, 256, 3, padding=1)
            self.lp3 = L2pooling(channels=256)
        self.conv14 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv15 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv16 = nn.Conv2d(256, 256, 3, padding=1)

        if pooling == 0:
            self.conv17 = nn.Conv2d(256, 512, 3, padding=1, stride=2)
        elif pooling == 1:
            self.conv17 = nn.Conv2d(256, 512, 3, padding=1)
            self.bp4 = BlurPooling(512)
        elif pooling == 2:
            self.conv17 = nn.Conv2d(256, 512, 3, padding=1)
            self.lp4 = L2pooling(channels=512)
        self.conv18 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv19 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv20 = nn.Conv2d(512, 512, 3, padding=1)

        # FC
        self.fc1_q = nn.Linear(512 * 3, 512)
        self.fc2_q = nn.Linear(512, 1)
        self.fc1_w = nn.Linear(512 * 3, 512)
        self.fc2_w = nn.Linear(512, 1)
        self.dropout = nn.Dropout()
        self.weighted_average = weighted_average

        self.pooling = pooling

    def extract_features(self, x):
        x11 = self.conv1(x)

        x12 = F.relu(self.conv2(x11))
        x13 = F.relu(self.conv3(x12))
        x14 = F.relu(self.conv4(x13)) + x11

        x21 = F.relu(self.conv5(x14))
        if self.pooling == 1:
            x21 = self.bp1(x21)
        elif self.pooling == 2:
            x21 = self.lp1(x21)
        x22 = F.relu(self.conv6(x21))
        x23 = F.relu(self.conv7(x22))
        x24 = F.relu(self.conv8(x23)) + x21

        x31 = F.relu(self.conv9(x24))
        if self.pooling == 1:
            x31 = self.bp2(x31)
        elif self.pooling == 2:
            x31 = self.lp2(x31)
        x32 = F.relu(self.conv10(x31))
        x33 = F.relu(self.conv11(x32))
        x34 = F.relu(self.conv12(x33)) + x31

        x41 = F.relu(self.conv13(x34))
        if self.pooling == 1:
            x41 = self.bp3(x41)
        elif self.pooling == 2:
            x41 = self.lp3(x41)
        x42 = F.relu(self.conv14(x41))
        x43 = F.relu(self.conv15(x42))
        x44 = F.relu(self.conv16(x43)) + x41

        x51 = F.relu(self.conv17(x44))
        if self.pooling == 1:
            x51 = self.bp4(x51)
        elif self.pooling == 2:
            x51 = self.lp4(x51)
        x52 = F.relu(self.conv18(x51))
        x53 = F.relu(self.conv19(x52))
        x54 = F.relu(self.conv20(x53)) + x51

        x6 = F.avg_pool2d(x54, 2)
        res = x6.view(-1, 512)

        return res

    def forward(self, data):
        """
        :param data: distorted and reference patches of images
        :return: quality of images/patches
        """
        x, x_ref = data     # [batch_size, patch_num, 3, 32, 32]
        batch_size = x.size(0)
        n_patches = x.size(1)
        if self.weighted_average:
            q = torch.ones((batch_size, 1), device=x.device)
        else:
            q = torch.ones((batch_size * n_patches, 1), device=x.device)

        for i in range(batch_size):
            # x[i] - [patch_num, 3,32,32]
            h = self.extract_features(x[i])  # [patch_num, 512]
            h_ref = self.extract_features(x_ref[i])  # [patch_num, 512]

            f_cat = torch.cat((h - h_ref, h, h_ref), 1)  # [patch_num, 512 * 3]

            f = F.relu(self.fc1_q(f_cat))     # [patch_num, 512*3] -> [patch_num, 512]
            f = self.dropout(f)
            f = self.fc2_q(f)   # [patch_num, 512] -> [patch_num, 1]

            if self.weighted_average:
                w = F.relu(self.fc1_w(f_cat))  # [patch_num, 512*3] -> [patch_num, 512]
                w = self.dropout(w)
                w = F.relu(self.fc2_w(w)) + 0.000001    # [patch_num, 512] -> [patch_num, 1]
                q[i] = torch.sum(f * w) / torch.sum(w)  # weighted averaging
            else:
                q[i*n_patches:(i+1)*n_patches] = h

        return q


