import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import DataLoader

import dataloader


class pdcconv(nn.Module):
    def __init__(self, type, inc=3, outc=3, stride=1):
        super(pdcconv, self).__init__()
        self.conv = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=3, padding=1)
        self.type = type
        self.stride = stride

    def forward(self, x):
        if self.type == 'ad':
            weights = self.conv.weight.data
            shape = weights.shape
            weights = weights.view(shape[0], shape[1], -1)
            weights_conv = (weights - weights[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape)  # clock-wise
            y = F.conv2d(x, weights_conv, padding=1, stride=self.stride)
            return y
        elif self.type == 'cd':
            weights = self.conv.weight.data
            weights_c = weights.sum(dim=[2, 3], keepdim=True)
            y = F.conv2d(x, weights - weights_c, padding=1, stride=self.stride)
            return y


class Deform2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=True):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(Deform2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        # conv则是实际进行的卷积操作，注意这里步长设置为卷积核大小，因为与该卷积核进行卷积操作的特征图是由输出特征图中每个点扩展为其对应卷积核那么多个点后生成的。
        # self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        self.pdc_ad = pdcconv('ad', inc=inc, outc=outc, stride=kernel_size)
        self.pdc_cd = pdcconv('cd', inc=inc, outc=outc, stride=kernel_size)
        self.conv = nn.Conv2d(in_channels=2 * outc, out_channels=outc, kernel_size=1, padding=0)
        # p_conv是生成offsets所使用的卷积，输出通道数为卷积核尺寸的平方的2倍，代表对应卷积核每个位置横纵坐标都有偏移量。
        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        # 初始化为0
        nn.init.constant_(self.p_conv.weight, 0)

        self.p_conv.register_full_backward_hook(self._set_lr)

        self.modulation = modulation  # modulation是可选参数,若设置为True,那么在进行卷积操作时,对应卷积核的每个位置都会分配一个权重。
        self.relu = nn.ReLU()
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        # 获取带偏移量的矩阵
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        # 维度重调
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)

        out1 = self.pdc_ad(x_offset)
        out2 = self.pdc_cd(x_offset)

        out = torch.cat([out1, out2], dim=1)

        out = self.conv(out)

        out = self.relu(out)
        return out

    def _get_p_n(self, N, dtype):
        # 由于卷积核中心点位置是其尺寸的一半，于是中心点向左（上）方向移动尺寸的一半就得到起始点，向右（下）方向移动另一半就得到终止点
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        # p0_y、p0_x就是输出特征图每点映射到输入特征图上的纵、横坐标值。
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride))

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    # 输出特征图上每点（对应卷积核中心）加上其对应卷积核每个位置的相对（横、纵）坐标后再加上自学习的（横、纵坐标）偏移量。
    # p0就是将输出特征图每点对应到卷积核中心，然后映射到输入特征图中的位置；
    # pn则是p0对应卷积核每个位置的相对坐标；
    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        # 可变形卷积公式
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        # 计算双线性插值点的4邻域点对应的权重
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):

        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset


class former(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(former, self).__init__()
        self.layer1 = Deform2d(inc=in_channel, outc=in_channel)
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel,
                               kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channel)  # BN处理
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.ic = in_channel
        self.oc = out_channel
        if in_channel != out_channel:
            self.conv1x1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1)
        # self.layer2 = Deform2d()

    def forward(self, x):
        identity = x
        x = self.layer1(x)
        x = self.relu(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.ic != self.oc:
            out = self.relu(out + self.conv1x1(identity))
        else:
            out = self.relu(out + identity)
        return out


class sampleblock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(sampleblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1)
        self.ic = in_channel
        self.oc = out_channel
        if in_channel != out_channel:
            self.conv1x1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.relu(y)
        y = self.conv3(y)
        if self.ic != self.oc:
            return y + self.conv1x1(x)
        return x + y


class dilationblock(nn.Module):
    def __init__(self, in_channel, out_channel=14):
        super(dilationblock, self).__init__()
        self.relu = nn.ReLU()
        self.conv_1x1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1)
        self.conv1 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=3, dilation=3)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=5, dilation=5)
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=9, dilation=9)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.bn3 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.relu(x)
        x = self.conv_1x1(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x1 = self.bn1(x1)
        x2 = self.bn2(x2)
        x3 = self.bn3(x3)
        x = self.relu(x1 + x2 + x3)
        return x


class channel_attention_block(nn.Module):
    def __init__(self, in_channel=14, out_channel=1):
        super(channel_attention_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv_1x1 = nn.Conv2d(in_channels=in_channel, out_channels=3, kernel_size=1)
        self.conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1)
        self.sig = nn.Sigmoid()
        self.conv_out = nn.Conv2d(in_channels=14, out_channels=out_channel, kernel_size=1)

    def forward(self, x):
        identity = x
        x = self.relu(x)
        x = self.conv_1x1(x)
        x = self.conv(x)
        x = self.sig(x)
        x = x * identity
        x = self.conv_out(x)
        return x


if __name__ == '__main__':
    data = dataloader.BSDS_Loader(split='test')
    test_data = DataLoader(dataset=data, batch_size=1,
                           num_workers=0, shuffle=True)
    model = former(3, 3)
    # model = channel_attention_block(3,3)
    for idx, (img, name, lb) in enumerate(test_data):
        output = model(img)
        print(00)
        print(output.size())
        print(img.size())
        output = torch.cat([output, img], dim=0)
        torchvision.utils.save_image(output,
                                     r'543s.png')
        torchvision.utils.save_image(img,
                                     r'54s.png')
        break
