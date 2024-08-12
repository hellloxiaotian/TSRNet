from model import common
import torch
import torch.nn as nn
import torch.nn.functional as F
import model.ops as ops
from model.DynamicConv import *
from model.sharpened_cosine_similarity import SharpCosSim2d
from model.absolute_pooling import MaxAbsPool2d


#  二叉树+Adan优化器+余弦卷积？
# 不同卷积核用不同的padding就可以保证特征图尺寸的一致，方案1.不同的支路使用不同大小的卷积核；2,。使用不同类型的卷积
def make_model(args, parent=False):
    return Binary(args)


# backbone: 9Conv+ReLU
class Branch(nn.Module):
    def __init__(self, features):
        super(Branch, self).__init__()
        # features = 64  # channel number
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True))
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True))

        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True))
        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True))
        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x2 = self.conv2(x)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        return x10


# 5Conv+ReLU
class FeatureRefineBlock(nn.Module):
    def __init__(self, features):
        super(FeatureRefineBlock, self).__init__()
        # features = 64  # channel number
        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True))
        self.conv12 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True))
        self.conv13 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True))
        self.conv14 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True))
        self.conv15 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x11 = self.conv11(x)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        x15 = self.conv15(x14)
        return x15


# 16层+14层+14层+9层卷积，最后一层上采样
class Binary(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(Binary, self).__init__()
        self.scale = args.scale[0]
        multi_scale = False
        features = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.cosine = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            SharpCosSim2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1),
            nn.ReLU(inplace=True)
        )
        self.branch4 = Branch(features)
        self.branch3 = Branch(features)
        self.FRB3 = FeatureRefineBlock(features)
        self.branch2 = Branch(features)
        self.FRB2 = FeatureRefineBlock(features)
        self.branch1 = Branch(features)
        self.FRB1 = FeatureRefineBlock(features)

        self.upsample = ops.UpsampleBlock(64, scale=self.scale, multi_scale=multi_scale, group=1)
        self.channel3 = nn.Conv2d(in_channels=features, out_channels=3, kernel_size=3, padding=1, groups=1, bias=False)

    def forward(self, x):
        x1 = self.conv1(x)

        branch4 = self.branch4(x1)
        branch3 = self.branch3(x1)

        branch3_1 = self.cosine(branch3)  # 分支3、4、FRB3后各加一个普通、余弦卷积、ReLU
        branch4_1 = self.cosine(branch4)

        FRB3 = self.FRB3(branch3_1 + branch4_1)
        FRB3_1 = self.cosine(FRB3)

        branch2 = self.branch2(x1)

        FRB2 = self.FRB2(branch2 + FRB3_1)
        branch1 = self.branch1(x1)
        FRB1 = self.FRB1(branch1 + FRB2)

        # temp = self.upsample(FRB1, scale=self.scale)
        temp = self.upsample(FRB1, 4)
        out = self.channel3(temp)

        return out
