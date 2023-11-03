import torch
import torch.nn as nn
import numpy as np
import scipy.stats as st
from torchvision import transforms
import torch.nn.functional as F
import torchfields
from scipy.io import savemat
from matplotlib import pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def REGconvdownblock(input_c, output_c):
    conv = nn.Sequential(
        nn.Conv2d(input_c, output_c, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(output_c),
        nn.LeakyReLU(0.2, True)
    )
    return conv
def REGconvblock(input_c, output_c):
    conv = nn.Sequential(
        nn.Conv2d(input_c, output_c, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(output_c),
        nn.LeakyReLU(0.2, True)
    )
    return conv

def REGconvupblock(input_c, output_c):
    conv = nn.Sequential(
        nn.ConvTranspose2d(input_c, output_c, kernel_size=2, stride=2),
        nn.BatchNorm2d(output_c),
        nn.LeakyReLU(0.2, True)
    )
    return conv

def Fconvdownblock(input_c, output_c):
    conv = nn.Sequential(
        nn.Conv2d(input_c, output_c, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(output_c),
        nn.ReLU()
    )
    return conv

def get_kernel(kernlen=5, nsig=3):     # nsig 标准差 ，kernlen=16核尺寸
    nsig = nsig.detach().numpy()
    interval = (2*nsig+1.)/kernlen      #计算间隔
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)   #在前两者之间均匀产生数据

                                          #高斯函数其实就是正态分布的密度函数
    kern1d = np.diff(st.norm.cdf(x))      #先积分在求导是为啥？得到一个维度上的高斯函数值
    '''st.norm.cdf(x):计算正态分布累计分布函数指定点的函数值
        累计分布函数：概率分布函数的积分'''
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))   #np.outer计算外积，再开平方，从1维高斯参数到2维高斯参数
    kernel = kernel_raw/kernel_raw.sum()             #确保均值为1
    return kernel

class R_PSF_down(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.nsig = nn.Parameter(torch.FloatTensor(torch.rand([1])), requires_grad=True)
        kernel = get_kernel(3, torch.max(self.nsig, 0)[0])
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = kernel.repeat(c,c,1,1)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
    def forward(self, input):
        out = torch.nn.functional.conv2d(input, self.weight, stride=2, padding=1)
        return out

"""
模拟B，下采样+高斯
"""
class B_Block(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(in_c),
            # nn.ReLU()
        )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(in_c, in_c, kernel_size=3, stride=2, padding=1),
        #     # nn.BatchNorm2d(in_c),
        #     # nn.ReLU()
        # )
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(in_c, in_c, kernel_size=3, stride=2, padding=1),
        #     # nn.BatchNorm2d(in_c),
        #     # nn.ReLU()
        # )

    def forward(self, x):
        # y = self.conv1(x)+self.conv2(x)+self.conv3(x)
        y = self.conv1(x)
        return y

"""
模拟B转置，上采样
"""
class B_T_Block(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_c, in_c, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(in_c),
            # nn.ReLU()
        )
        # self.conv2 = nn.Sequential(
        #     nn.ConvTranspose2d(in_c, in_c, kernel_size=4, stride=2, padding=1),
        #     # nn.BatchNorm2d(in_c),
        #     # nn.ReLU()
        # )
        # self.conv3 = nn.Sequential(
        #     nn.ConvTranspose2d(in_c, in_c, kernel_size=4, stride=2, padding=1),
        #     # nn.BatchNorm2d(in_c),
        #     # nn.ReLU()
        # )

    def forward(self, x):
        # y = self.conv1(x)+self.conv2(x)+self.conv3(x)
        y = self.conv1(x)
        return y


"""
模拟R，光谱下采
"""
class R_Block(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, 4, kernel_size=1, stride=1),
            # nn.BatchNorm2d(1)
        )

    def forward(self, x):
        y = self.conv(x)
        return y


"""
模拟R转置，光谱上采
"""
class R_T_Block(nn.Module):
    def __init__(self, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, out_c, kernel_size=1, stride=1),
            # nn.BatchNorm2d(out_c)
        )

    def forward(self, x):
        y = self.conv(x)
        return y

class Encoding_Block(torch.nn.Module):
    def __init__(self, c_in ):
        super(Encoding_Block, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=c_in, out_channels=128, kernel_size=3, padding=3 // 2)
        self.conv2 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=3 // 2)
        self.conv3 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=3 // 2)
        self.conv4 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=3 // 2)
        self.conv5 = torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride = 2, padding=3 // 2)


        self.act = torch.nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)


    def forward(self, input):

        out1 = self.act(self.conv1(input))
        out2 = self.act(self.conv2(out1))
        out3 = self.act(self.conv3(out2))
        f_e = self.conv4(out3)
        down = self.act(self.conv5(f_e))
        return f_e, down

class Encoding_Block_End(torch.nn.Module):
    def __init__(self, c_in = 64 ):
        super(Encoding_Block_End, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=c_in, out_channels=128, kernel_size=3, padding=3 // 2)
        self.conv2 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=3 // 2)
        self.conv3 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=3 // 2)
        self.conv4 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=3 // 2)
        self.act = torch.nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, input):
        out1 = self.act(self.conv1(input))
        out2 = self.act(self.conv2(out1))
        out3 = self.act(self.conv3(out2))
        f_e = self.conv4(out3)
        return f_e

class Decoding_Block(torch.nn.Module):
    def __init__(self,c_in ):
        super(Decoding_Block, self).__init__()
        self.conv0 = torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=3 // 2)
        self.conv1 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=3 // 2)
        self.conv2 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=3 // 2)

        self.conv3 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1)
        # self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=3 // 2)
        self.batch = 1
        #self.up = torch.nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.up = torch.nn.ConvTranspose2d(c_in, 256, kernel_size=3, stride=2,padding=3 // 2)

        self.act =  torch.nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def up_sampling(self, input,label, kernel_size=3, out_features=64):

        batch_size = self.batch

        label_h1 = int(label.shape[2])
        label_h2 = int(label.shape[3])

        in_features = int(input.shape[1])
        Deconv = self.up(input)

        return Deconv
    def forward(self, input, map):

        up = self.up(input, output_size = [input.shape[0],input.shape[1],map.shape[2],map.shape[3]] )
        cat = torch.cat((up, map), 1)
        cat = self.act(self.conv0(cat))
        out1 = self.act(self.conv1(cat))
        out2 = self.act(self.conv2(out1))

        out3 = self.conv3(out2)

        return out3

class Feature_Decoding_End(torch.nn.Module):
    def __init__(self, c_out):
        super(Feature_Decoding_End, self).__init__()
        self.conv0 = torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=3 // 2)
        self.conv1 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=3 // 2)
        self.conv2 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=3 // 2)

        self.conv3 = torch.nn.Conv2d(in_channels=256, out_channels=c_out, kernel_size=3, padding=3 // 2)
        self.batch = 1
        self.up = torch.nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=3 // 2)
        self.act = torch.nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
    def up_sampling(self, input,label, kernel_size=3, out_features=64):

        batch_size = self.batch

        label_h1 = int(label.shape[2])
        label_h2 = int(label.shape[3])

        in_features = int(input.shape[1])

        Deconv = self.up(input)

        return Deconv
    def forward(self, input,map):

        up = self.up(input,  output_size = [input.shape[0],input.shape[1],map.shape[2],map.shape[3]] )
        cat = torch.cat((up, map), 1)
        cat = self.act(self.conv0(cat))
        out1 = self.act(self.conv1(cat))
        out2 = self.act(self.conv2(out1))

        out3 = self.conv3(out2)

        return out3


"""
    配准模块
    用LeakyReLu能够表示反向偏移
"""
class Registration(nn.Module):
    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        """
        This function creates one contracting block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels,
                            padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(0.2),
        )
        return block

    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This function creates one expansive block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.LeakyReLU(0.2),
            torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2,
                                     padding=1, output_padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(0.2),
        )
        return block

    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This returns final block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(0.2),
        )
        return block

    def __init__(self, in_channel, out_channel):
        super(Registration, self).__init__()
        # Encode
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=32)
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(32, 64)
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(64, 128)
        self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
        # Bottleneck
        mid_channel = 128
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=3, in_channels=mid_channel, out_channels=mid_channel * 2, padding=1),
            torch.nn.BatchNorm2d(mid_channel * 2),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(kernel_size=3, in_channels=mid_channel * 2, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.LeakyReLU(0.2),
            torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=3, stride=2,
                                     padding=1, output_padding=1),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.LeakyReLU(0.2),
        )
        # Decode
        self.conv_decode3 = self.expansive_block(256, 128, 64)
        self.conv_decode2 = self.expansive_block(128, 64, 32)
        self.final_layer = self.final_block(64, 32, out_channel)

    def crop_and_concat(self, upsampled, bypass, crop=False):
        """
        This layer crop the layer from contraction block and concat it with expansive block vector
        """
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)
        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool3)
        # Decode
        decode_block3 = self.crop_and_concat(bottleneck1, encode_block3)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1)
        final_layer = self.final_layer(decode_block1)
        final_layer = final_layer.clamp(min=-5, max=5)
        return final_layer

"""
    重采样模块
"""
class SpatialTransformation(nn.Module):
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        super(SpatialTransformation, self).__init__()

    def meshgrid(self, height, width):
        x_t = torch.matmul(torch.ones([height, 1]), torch.transpose(torch.unsqueeze(torch.linspace(0.0, width -1.0, width), 1), 1, 0))
        y_t = torch.matmul(torch.unsqueeze(torch.linspace(0.0, height - 1.0, height), 1), torch.ones([1, width]))

        x_t = x_t.expand([height, width])
        y_t = y_t.expand([height, width])
        if self.use_gpu==True:
            x_t = x_t.cuda(0)
            y_t = y_t.cuda(0)

        return x_t, y_t

    def repeat(self, x, n_repeats):
        rep = torch.transpose(torch.unsqueeze(torch.ones(n_repeats), 1), 1, 0)
        rep = rep.long()
        x = torch.matmul(torch.reshape(x, (-1, 1)), rep)
        if self.use_gpu:
            x = x.cuda(0)
        return torch.squeeze(torch.reshape(x, (-1, 1)))


    def interpolate(self, im, x, y):

        im = F.pad(im, (0,0,1,1,1,1,0,0))

        batch_size, height, width, channels = im.shape

        batch_size, out_height, out_width = x.shape

        x = x.reshape(1, -1)
        y = y.reshape(1, -1)

        x = x + 1
        y = y + 1

        max_x = width - 1
        max_y = height - 1

        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1

        x0 = torch.clamp(x0, 0, max_x)
        x1 = torch.clamp(x1, 0, max_x)
        y0 = torch.clamp(y0, 0, max_y)
        y1 = torch.clamp(y1, 0, max_y)

        dim2 = width
        dim1 = width*height
        base = self.repeat(torch.arange(0, batch_size)*dim1, out_height*out_width)

        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2

        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = torch.reshape(im, [-1, channels])
        im_flat = im_flat.float()
        dim, _ = idx_a.transpose(1,0).shape
        Ia = torch.gather(im_flat, 0, idx_a.transpose(1,0).expand(dim, channels))
        Ib = torch.gather(im_flat, 0, idx_b.transpose(1,0).expand(dim, channels))
        Ic = torch.gather(im_flat, 0, idx_c.transpose(1,0).expand(dim, channels))
        Id = torch.gather(im_flat, 0, idx_d.transpose(1,0).expand(dim, channels))

        # and finally calculate interpolated values
        x1_f = x1.float()
        y1_f = y1.float()

        dx = x1_f - x
        dy = y1_f - y

        wa = (dx * dy).transpose(1,0)
        wb = (dx * (1-dy)).transpose(1,0)
        wc = ((1-dx) * dy).transpose(1,0)
        wd = ((1-dx) * (1-dy)).transpose(1,0)

        output = torch.sum(torch.squeeze(torch.stack([wa*Ia, wb*Ib, wc*Ic, wd*Id], dim=1)), 1)
        output = torch.reshape(output, [-1, out_height, out_width, channels])
        return output

    def forward(self, moving_image, deformation_matrix):
        dx = deformation_matrix[:, :, :, 0]
        dy = deformation_matrix[:, :, :, 1]

        batch_size, height, width = dx.shape

        x_mesh, y_mesh = self.meshgrid(height, width)

        x_mesh = x_mesh.expand([batch_size, height, width])
        y_mesh = y_mesh.expand([batch_size, height, width])
        x_new = dx + x_mesh
        y_new = dy + y_mesh

        return self.interpolate(moving_image, x_new, y_new)



"""
    模型驱动--重建模块
"""
class FusionM(nn.Module):
    def __init__(self, R, R_T, B, B_T, REG, STF, conv_p, size_E, bands, layer, args):
        super().__init__()
        self.R = R
        self.R_T = R_T
        self.B = B
        self.B_T = B_T
        self.REG = REG
        self.STF = STF
        self.layer = layer
        self.conv_p = conv_p
        self.args = args
        self.ones = self.Generate_ONES(size_E, bands, args.batch_size)
        self.eta = torch.nn.Parameter(torch.tensor(2.))

    def Generate_ONES(self, size_E, bands, batchsize):
        ones = torch.torch.eye(size_E, size_E).cuda(self.args.gpu)
        ones = torch.unsqueeze(torch.unsqueeze(ones, 0), 0)
        ones = ones.repeat([batchsize, bands, 1, 1])
        return ones


    # 对HS进行变形场偏移
    def UseDSP(self, HS, DSP, inv=False):
        # DSP = DSP.field()
        # if inv:
        #     DSP = -DSP
        # HS_R = DSP(HS)
        DSP = DSP.permute(0, 2, 3, 1)
        HS = HS.permute(0, 2, 3, 1)
        HS_R = self.STF(HS, DSP).permute(0, 3, 1, 2)
        return HS_R

    def forward(self, PAN, PAN_reg, HS, Xpk, DSP_T):
        # I = self.R(HS).detach()
        # I = HS[:, 59, :, :].unsqueeze(1)

        I = torch.nn.functional.conv2d(HS, self.conv_p, bias=None, stride=1, padding=0)
        I = torch.nn.functional.interpolate(I, scale_factor=2, mode='bilinear', align_corners=False)
        DSP = self.REG(torch.cat((I, PAN_reg), dim=1))
        # ## 调试用，DSP设为0
        # DSP = torch.zeros_like(DSP)

        # 初始化Xp0需要用到DSP_T
        if self.layer == 1:
            Xpk = self.UseDSP(Xpk,  DSP_T.detach())
            # savemat('./dataout/pavia_alpha1/Xpk.mat', {'Xpk': Xpk.detach().cpu().numpy()})

        E_as_D = self.UseDSP(self.ones, DSP.detach())
        HS_p = torch.matmul(self.B_T(HS), E_as_D)
        # HS_p = self.B_T(HS)

        PAN_p = self.R_T(PAN)

        X_p1 = self.UseDSP(Xpk, DSP.detach())
        # X_p1 = self.UseDSP(Xpk, DSP)
        X_p1 = self.B(self.B_T(X_p1))
        X_p1 = torch.matmul(X_p1, E_as_D)
        X_p2 = self.R_T(self.R(Xpk))
        X_p = Xpk - self.eta*X_p1 - self.eta*X_p2

        Uk = self.eta*HS_p + self.eta*PAN_p + X_p
        return Uk, DSP, I

"""
    模型驱动--去噪模块
"""
class Denoised(torch.nn.Module):
    def __init__(self, cin):
        super(Denoised, self).__init__()

        self.Encoding_block1 = Encoding_Block(128)
        self.Encoding_block2 = Encoding_Block(128)
        self.Encoding_block3 = Encoding_Block(128)
        self.Encoding_block4 = Encoding_Block(128)
        self.Encoding_block_end = Encoding_Block_End(128)

        self.Decoding_block1 = Decoding_Block(256)
        self.Decoding_block2 = Decoding_Block(512)
        self.Decoding_block3 = Decoding_Block(512)
        self.Decoding_block_End = Feature_Decoding_End(102)

        self.acti = torch.nn.ReLU()
        self.reset_parameters()

        self.fe_conv1 = torch.nn.Conv2d(in_channels=cin, out_channels=128, kernel_size=3, padding=3 // 2)
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = self.fe_conv1(x)

        encode0, down0 = self.Encoding_block1(x)
        encode1, down1 = self.Encoding_block2(down0)
        encode2, down2 = self.Encoding_block3(down1)
        encode3, down3 = self.Encoding_block4(down2)

        media_end = self.Encoding_block_end(down3)

        decode3 = self.Decoding_block1(media_end, encode3)
        decode2 = self.Decoding_block2(decode3, encode2)
        decode1 = self.Decoding_block3(decode2, encode1)
        decode0 = self.Decoding_block_End(decode1, encode0)

        return decode0

"""
    融合去噪迭代
"""
class FandD_M(nn.Module):
    def __init__(self, R, R_T, B, B_T, REG, STF, conv_p, Denoise, size_E, bands, args):
        super().__init__()

        self.F1 = FusionM(R, R_T, B, B_T, REG, STF, conv_p, size_E, bands, 1, args)
        self.F2 = FusionM(R, R_T, B, B_T, REG, STF, conv_p, size_E, bands, 2, args)
        self.F3 = FusionM(R, R_T, B, B_T, REG, STF, conv_p, size_E, bands, 3, args)
        self.F4 = FusionM(R, R_T, B, B_T, REG, STF, conv_p, size_E, bands, 4, args)
        self.D1 = Denoise
        self.D2 = Denoise
        self.D3 = Denoise
        self.D4 = Denoise

    def forward(self, PAN, PAN_reg, HS, DSP_T):
        X0 = torch.nn.functional.interpolate(HS, scale_factor=2, mode='bicubic', align_corners=False)
        X1, DSP1, I = self.F1(PAN, PAN_reg, HS, X0, DSP_T)
        X1 = self.D1(X1)
        X2, _, __ = self.F2(PAN, PAN_reg, HS, X1, DSP_T)
        X2 = self.D2(X2)
        X3, _, __ = self.F3(PAN, PAN_reg, HS, X2, DSP_T)
        X3 = self.D3(X3)
        # X4, _, __ = self.F4(PAN, PAN_reg, HS, X3, DSP_T)
        # X4 = self.D4(X4)
        return X3, DSP1, I
class FandD_H(nn.Module):
    def __init__(self, R, R_T, B, B_T, REG, STF, conv_p, Denoise, size_E, bands, args):
        super().__init__()

        self.F1 = FusionM(R, R_T, B, B_T, REG, STF, conv_p, size_E, bands, 1, args)
        self.F2 = FusionM(R, R_T, B, B_T, REG, STF, conv_p, size_E, bands, 2, args)
        self.F3 = FusionM(R, R_T, B, B_T, REG, STF, conv_p, size_E, bands, 3, args)
        self.F4 = FusionM(R, R_T, B, B_T, REG, STF, conv_p, size_E, bands, 4, args)
        self.D1 = Denoise
        self.D2 = Denoise
        self.D3 = Denoise
        self.D4 = Denoise

    def forward(self, PAN, PAN_reg, HS, DSP_T):
        X0 = torch.nn.functional.interpolate(HS, scale_factor=2, mode='bicubic', align_corners=False)
        X1, DSP1, I = self.F1(PAN, PAN_reg, HS, X0, DSP_T)
        X1 = self.D1(X1)
        X2, _, __ = self.F2(PAN, PAN_reg, HS, X1, DSP_T)
        X2 = self.D2(X2)
        X3, _, __ = self.F3(PAN, PAN_reg, HS, X2, DSP_T)
        X3 = self.D3(X3)
        # X4, _, __ = self.F4(PAN, PAN_reg, HS, X3, DSP_T)
        # X4 = self.D4(X4)
        return X3, DSP1, I


class Registration_T(nn.Module):
    def __init__(self, input, output, conv_p):
        super().__init__()
        self.REG_T = Registration(input, output)
        self.conv_p = conv_p
    def forward(self, PAN_reg, HS):
        I = torch.nn.functional.conv2d(HS, self.conv_p, bias=None, stride=1, padding=0)
        I = torch.nn.functional.interpolate(I, scale_factor=2, mode='bilinear', align_corners=False)
        DSP_T = self.REG_T(torch.cat((I, PAN_reg), dim=1))

        return DSP_T


"""
    主框架
"""
class Main(nn.Module):
    def __init__(self, size_E, bands, args):
        super().__init__()
        R_T = R_T_Block(bands)
        R = R_Block(bands)
        B = B_Block(bands)
        B_T = B_T_Block(bands)
        REG_M = Registration(2, 2)
        REG_H = Registration(2, 2)
        STF_M = SpatialTransformation(True)
        STF_H = SpatialTransformation(True)
        DM = Denoised(102)
        DH = Denoised(102)
        conv_p = nn.Parameter(torch.FloatTensor(torch.ones((1, 102, 1, 1)) / 102), requires_grad=True).cuda(
            args.gpu)
        self.R = R
        self.B_T = B_T
        self.transform = transforms.GaussianBlur(15, 1)
        self.REG_M_T = Registration_T(2, 2, conv_p)
        self.REG_H_T = Registration_T(2, 2, conv_p)
        self.STF_M = STF_M
        self.STF_H = STF_H

        self.FandD_M = FandD_M(R, R_T, B, B_T, REG_M, STF_M, conv_p, DM, size_E, bands, args)
        self.FandD_H = FandD_H(R, R_T, B, B_T, REG_H, STF_H, conv_p, DH, 2*size_E, bands, args)
    # def UseDSP(self, HS, DSP):
    #     DSP = DSP.field()
    #     HS_R = DSP(HS)
    #     return HS_R
    def forward(self, MS, HS):
        MS_H = MS
        MS_M = torch.nn.functional.interpolate(MS_H, scale_factor=1/2, mode='bilinear', align_corners=False)

        HS_L = HS

        MS_H_gauss = self.transform(MS_H)
        MS_L = torch.nn.functional.interpolate(MS_H_gauss, scale_factor=1/4, mode='bilinear', align_corners=False)
        MS_M_reg = torch.nn.functional.interpolate(MS_L, scale_factor=2, mode='bilinear', align_corners=False)
        MS_M_reg = torch.mean(MS_M_reg, dim=1).unsqueeze(1)

        MS_H_reg = torch.nn.functional.interpolate(MS_H_gauss, scale_factor=1/2, mode='bilinear', align_corners=False)
        MS_H_reg = torch.nn.functional.interpolate(MS_H_reg, scale_factor=2, mode='bilinear', align_corners=False)
        MS_H_reg = torch.mean(MS_H_reg, dim=1).unsqueeze(1)

        # PAN_M_reg = PAN_M
        # PAN_H_reg = PAN_H

        DSPM_T = self.REG_M_T(MS_M_reg, HS_L)
        HS_M, DSPM, I_M = self.FandD_M(MS_M, MS_M_reg, HS_L, DSPM_T)
        DSPH_T = self.REG_M_T(MS_H_reg, HS_M)
        HS_H, DSPH, I_H = self.FandD_H(MS_H, MS_H_reg, HS_M, DSPH_T)


        # savemat('./dataout/pavia/PAN.mat', {'PAN': PAN_M.detach().cpu().numpy()})
        # savemat('./dataout/pavia/I_M.mat', {'I_M': I_M.detach().cpu().numpy()})

        # savemat('./dataout/pavia_alpha1/HS_M.mat', {'HS_M': HS_M.detach().cpu().numpy()})
        # savemat('./dataout/pavia_alpha1/I_Mtest.mat', {'I_M': self.UseDSP(I_H, DSPH).detach().cpu().numpy()})

        return HS_H, HS_M, I_M, I_H, MS_M_reg, MS_H_reg, DSPM, DSPH, DSPM_T, DSPH_T, self.STF_M, self.STF_H





if __name__ == "__main__":
    # I = torch.rand([1, 1, 40, 40])
    # PAN = torch.rand([1, 1, 40, 40])
    # net = Registration()
    # out = net(I, PAN)
    # print(out.size())

    # HS = torch.rand([1, 102, 40, 40])
    # PAN = torch.rand([1, 1, 80, 80])
    # net = FusionM()
    # out = net(PAN, HS)
    # print(out.size())

    # HS = torch.rand([1, 102, 40, 40])
    # net = HS2I()
    # out = net(HS)
    # print(out.size())

    HS = torch.rand([8, 102, 40, 40])
    PAN = torch.rand([8, 1, 160, 160])
    net = Main(80, 102, 8)
    out = net(PAN, HS)
    print(out.size())

    # I = torch.rand([8, 1, 40, 40])
    # Xp = torch.rand([8, 102, 40, 40])
    # HS = torch.rand([8, 102, 20, 20])
    # R_T = R_T_Block(102)
    # R = R_Block(102)
    # B = B_Block(102)
    # B_T = B_T_Block(102)
    # net = FandD(R, R_T, B, B_T, 40, 102, 8)
    # out = net(I, HS)
    # print(out.size())