import os
import torch.nn as nn
import torch.optim as optim
from archs.base_networks import *
from torchvision.transforms import *
from archs.hebing import *

class IUDTNet(nn.Module):
    def __init__(self, embed_dim, img_size, num_heads, window_size=8, scale_factor=2):
        super(IUDTNet, self).__init__()
        
        if scale_factor == 2:
        	kernel = 6
        	stride = 2
        	padding = 2
        elif scale_factor == 4:
        	kernel = 8
        	stride = 4
        	padding = 2
        elif scale_factor == 8:
        	kernel = 12
        	stride = 8
        	padding = 2


        #Global and local feature extraction
        h, w = img_size[0], img_size[1]
        self.gpsab = GPSAB(dim=embed_dim, input_resolution=(h, w), num_heads=num_heads, window_size=window_size)
        # self.lcb = LCBlock(embed_dim, embed_dim)


        #Back-projection stages
        # self.up1 = UpBlock(embed_dim, kernel, stride, padding)
        # self.down1 = DownBlock(embed_dim, kernel, stride, padding)
        # self.up2 = UpBlock(embed_dim, kernel, stride, padding)
        # self.down2 = D_DownBlock(embed_dim, kernel, stride, padding, 2)
        # self.up3 = D_UpBlock(embed_dim, kernel, stride, padding, 2)
        # self.down3 = D_DownBlock(embed_dim, kernel, stride, padding, 3)
        # self.up4 = D_UpBlock(embed_dim, kernel, stride, padding, 3)
        # self.down4 = D_DownBlock(embed_dim, kernel, stride, padding, 4)
        # self.up5 = D_UpBlock(embed_dim, kernel, stride, padding, 4)
        # self.down5 = D_DownBlock(embed_dim, kernel, stride, padding, 5)
        # self.up6 = D_UpBlock(embed_dim, kernel, stride, padding, 5)
        # self.down6 = D_DownBlock(embed_dim, kernel, stride, padding, 6)
        # self.up7 = D_UpBlock(embed_dim, kernel, stride, padding, 6)
        # self.down7 = D_DownBlock(embed_dim, kernel, stride, padding, 7)
        # self.up8 = D_UpBlock(embed_dim, kernel, stride, padding, 7)
        # self.down8 = D_DownBlock(embed_dim, kernel, stride, padding, 8)
        # self.up9 = D_UpBlock(embed_dim, kernel, stride, padding, 8)
        # self.down9 = D_DownBlock(embed_dim, kernel, stride, padding, 9)
        # self.up10 = D_UpBlock(embed_dim, kernel, stride, padding, 9)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            
    def forward(self, x, x_size):
        H, W = x_size
        B, _, C = x.shape       
        
        x = self.gpsab(x, (H, W))    # [1, 48576, 60]
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        # x = self.lcb(x) # [1, 60, 264, 184]
        
        

        # h1 = self.up1(x)
        # l1 = self.down1(h1)
        # h2 = self.up2(l1)
        
        # concat_h = torch.cat((h2, h1),1)
        # l = self.down2(concat_h)
        
        # concat_l = torch.cat((l, l1),1)
        # h = self.up3(concat_l)
        
        # concat_h = torch.cat((h, concat_h),1)
        # x = self.down3(concat_h)
        
        
        # concat_l = torch.cat((l, concat_l),1)
        # h = self.up4(concat_l)
        
        # concat_h = torch.cat((h, concat_h),1)
        # l = self.down4(concat_h)
        
        # concat_l = torch.cat((l, concat_l),1)
        # h = self.up5(concat_l)
        
        # concat_h = torch.cat((h, concat_h),1)
        # l = self.down5(concat_h)
        
        # concat_l = torch.cat((l, concat_l),1)
        # h = self.up6(concat_l)
        
        # concat_h = torch.cat((h, concat_h),1)
        # l = self.down6(concat_h)
        
        # concat_l = torch.cat((l, concat_l),1)
        # h = self.up7(concat_l)
        
        # concat_h = torch.cat((h, concat_h),1)
        # x = self.down7(concat_h)
        
        # concat_l = torch.cat((l, concat_l),1)
        # h = self.up8(concat_l)
        
        # concat_h = torch.cat((h, concat_h),1)
        # l = self.down8(concat_h)
        
        # concat_l = torch.cat((l, concat_l),1)
        # h = self.up9(concat_l)
        
        # concat_h = torch.cat((h, concat_h),1)
        # x = self.down9(concat_h)
        
        # concat_l = torch.cat((l, concat_l),1)
        # x = self.up10(concat_l)
        

        x = x.reshape(B, C, H*W).permute(0, 2, 1)

        return x


    

if __name__ == '__main__':
    # window_size = 8       [1, 60, 264, 184]
    b, h, w, c = 1, 264, 184, 60
    inp = torch.randn((1, 48576, 60))
    # model1 = GPSAB(dim=c, input_resolution=(h, w), num_heads=2, window_size=8)
    # oup1 = model1(inp, (h, w))
    # rev = oup1.reshape(b, h, w, c).permute(0, 3, 1, 2)
    
    # model2 = LCBlock(c, c)
    # oup2 = model2(rev)
    # print(oup2.shape)   # torch.Size([3, 60, 264, 184])
    
    model = IUDTNet(c, (h,w), (6,), 8, 2)
    oup = model(inp, (h,w))    # [1, 60, 528, 368]
    print(oup.shape)
	
    
    