import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

'''-------------一、SE模块-----------------------------'''

# 全局平均池化+1*1卷积核+ReLu+1*1卷积核+Sigmoid
class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        # 读取批数据图片数量及通道数
        b, c, h, w = x.size()
        # Fsq操作：经池化后输出b*c的矩阵
        y = self.gap(x).view(b, c)
        # Fex操作：经全连接层输出（b，c，1，1）矩阵
        y = self.fc(y).view(b, c, 1, 1)
        # Fscale操作：将得到的权重乘以原来的特征图x
        return x * y.expand_as(x)
    
    
    
    
# 深度可分离卷积    (1, 3, 30, 30) -> (1, 3, 30, 30)
class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch, kernel_size=3, padding=1, bias=True):
        super(DepthwiseSeparable, self).__init__()

        # 第一步    depthwise convolution
        # 要求输入通道和输出通道相同，并且有几个输入通道，就有几个groups
        self.depthwise = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=in_ch,
            bias=bias
        )

    def forward(self, input):
        out = self.depthwise(input)
        return out
    
#dw卷积
class dwconv(nn.Module):
    def __init__(self, inp_channels):
        super(dwconv, self).__init__()
        self.conv1 = nn.Conv2d(inp_channels, inp_channels, kernel_size=5, stride=1, padding=2, groups=inp_channels)
        
        # self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.gl = nn.GELU()
        
        self.hidden_features = inp_channels
    def forward(self, x):
        b, c, _, _ = x.size()   # (1,3,240,240) -> (1, 240, 3, 240)
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, b, c).contiguous()  # b Ph*Pw c    1, 3, 1, 3
        x = self.gl(self.conv1(x))
        # x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x



class LCBlock(nn.Module):
    def __init__(self,in_chanel,out_chanel):
        super(LCBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_chanel,out_chanel,kernel_size=1,stride=1)
        self.dw = DepthwiseSeparable(in_chanel)
        self.se = SE_Block(in_chanel)
        
    def forward(self,x):
        input = x
        x = self.conv1(x)   # [1, 3, 240, 240]
        x = self.dw(x)  # [1, 3, 240, 240]
        x = self.se(x)  # [1, 3, 240, 240]
        x = self.conv1(x)
        x = input+x
        return x

if __name__ == '__main__':
    # [1, 60, 264, 184]
    inp = torch.randn((1, 60, 264, 184))
    model = LCBlock(60, 60)
    oup = model(inp)
    print(oup.shape)    # torch.Size([1, 3, 240, 240])