import torch
import torch.nn as nn
# import SVASTIN.modules.module_util as mutil
import SVASTIN.config as c
# from torchsummary import summary

# Dense connection
class ResidualDenseBlock_out(nn.Module):
    def __init__(self, input, output, nf=c.nf, gc=c.gc, bias=True, use_snorm=False):
        super(ResidualDenseBlock_out, self).__init__()
        if use_snorm:
            self.conv1 = nn.utils.spectral_norm(nn.Conv2d(nf, gc, 3, 1, 1, bias=bias))
            self.conv_1 = nn.utils.spectral_norm(nn.Conv3d(nf, gc, 3, 1, 1, bias=bias))
            self.conv2 = nn.utils.spectral_norm(nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias))
            self.conv_2 = nn.utils.spectral_norm(nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias))
            self.conv3 = nn.utils.spectral_norm(nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias))
            self.conv_3 = nn.utils.spectral_norm(nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias))
            self.conv4 = nn.utils.spectral_norm(nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias))
            self.conv_4 = nn.utils.spectral_norm(nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias))
            self.conv5 = nn.utils.spectral_norm(nn.Conv2d(nf + 4 * gc, output, 3, 1, 1, bias=bias))
            self.conv_5 = nn.utils.spectral_norm(nn.Conv2d(nf + 4 * gc, output, 3, 1, 1, bias=bias))
        else:
            self.conv_1 = nn.Conv3d(input, 48, kernel_size=(3,3,3), padding=(1,1,1), bias=bias)
            self.conv_2 = nn.Conv3d(input + 48, 48, kernel_size=(3,3,3), padding=(1,1,1), bias=bias)
            self.conv_3 = nn.Conv3d(input + 2 * 48, 48, kernel_size=(3,3,3), padding=(1,1,1), bias=bias)
            self.conv_4 = nn.Conv3d(input + 3 * 48, 48, kernel_size=(3,3,3), padding=(1,1,1), bias=bias)
            self.conv_5 = nn.Conv3d(input + 4 * 48, output, kernel_size=(3,3,3), padding=(1,1,1), bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv_1(x))
        x2 = self.lrelu(self.conv_2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv_3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv_4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv_5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5

if __name__ == '__main__':
    net = ResidualDenseBlock_out(12, 12, nf=c.nf, gc=c.gc).cuda()
    print(net)