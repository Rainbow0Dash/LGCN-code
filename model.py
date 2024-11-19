from part import *
import torch.nn as nn
import torchvision

import torch
import torch.nn.functional as F



class model(nn.Module):
    def __init__(self, out_channel):
        super(model, self).__init__()
        self.left1 = former(in_channel=3, out_channel=out_channel)
        self.left2 = former(in_channel=out_channel, out_channel=out_channel * 2)
        self.left3 = former(in_channel=out_channel * 2, out_channel=out_channel * 4)
        self.left4 = former(in_channel=out_channel * 4, out_channel=out_channel * 8)
        self.leftpool = nn.MaxPool2d(kernel_size=2)

        self.right1 = sampleblock(in_channel=3, out_channel=out_channel)
        self.right2 = sampleblock(in_channel=out_channel, out_channel=out_channel * 2)
        self.right3 = sampleblock(in_channel=out_channel * 2, out_channel=out_channel * 4)
        self.right4 = sampleblock(in_channel=out_channel * 4, out_channel=out_channel * 8)
        self.rightpool = nn.MaxPool2d(kernel_size=2)

        self.side1 = nn.Sequential(dilationblock(in_channel=out_channel),
                                   channel_attention_block())
        self.side2 = nn.Sequential(dilationblock(in_channel=out_channel * 2),
                                   channel_attention_block())
        self.side3 = nn.Sequential(dilationblock(in_channel=out_channel * 4),
                                   channel_attention_block())
        self.side4 = nn.Sequential(dilationblock(in_channel=out_channel * 8),
                                   channel_attention_block())

        self.outblock = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1)
        self.sig = nn.Sigmoid()

    def get_weight(self):
        conv_weights = []
        bn_weights = []
        relu_weights = []
        for pname, p in self.named_parameters():  # 获取每一层模型的名字和参数
            if 'bn' in pname:
                bn_weights.append(p)
            elif 'relu' in pname:
                relu_weights.append(p)
            else:
                conv_weights.append(p)
        return conv_weights, bn_weights, relu_weights

    def forward(self, x):
        left = self.left1(x)
        right = self.right1(x)
        out1 = torch.add(left, right)

        left = self.leftpool(left)
        right = self.rightpool(right)

        left = self.left2(left)
        right = self.right2(right)
        out2 = torch.add(left, right)

        left = self.leftpool(left)
        right = self.rightpool(right)

        left = self.left3(left)
        right = self.right3(right)
        out3 = torch.add(left, right)

        left = self.leftpool(left)
        right = self.rightpool(right)

        left = self.left4(left)
        right = self.right4(right)
        out4 = torch.add(left, right)

        out1 = self.side1(out1)
        out2 = self.side2(out2)
        out3 = self.side3(out3)
        out4 = self.side4(out4)

        shape = x.size()
        out = [out1, out2, out3, out4]
        for i in range(len(out)):
            out[i] = F.interpolate(out[i], (shape[-2], shape[-1]), mode="bilinear", align_corners=False)
        concatout = torch.cat(out, dim=1)
        out5 = self.outblock(concatout)
        out.append(out5)
        out = [torch.sigmoid(r) for r in out]
        return out


def adjust_learning_rate(optimizer, epoch, lr):
    if epoch != 0:
        if epoch % 10 == 0:
            lr = lr * 0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


def cross_entropy_loss_RCF(prediction, labelf, beta):
    label = labelf.long()
    mask = labelf.clone()
    num_positive = torch.sum(label == 1).float()
    num_negative = torch.sum(label == 0).float()

    mask[label == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[label == 0] = beta * num_positive / (num_positive + num_negative)
    mask[label == 2] = 0
    labelf[label == 2] = 0
    cost = F.binary_cross_entropy(prediction, labelf, weight=mask, reduction='sum')

    return cost
