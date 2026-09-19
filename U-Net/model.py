from torch import nn
import torch
from matplotlib import pyplot as plt
import math
import random

class U_Net(nn.Module):
    def __init__(self,size,classes:int,color:bool=True):
        super().__init__()
        if color:
            self.channels=3
        else:
            self.channels=1
        # Size initially at n
        self.block1_contract=double_conv(in_channels=self.channels,out_channels=64)
        self.block2_contract=double_conv(in_channels=64,out_channels=128)
        self.block3_contract=double_conv(in_channels=128,out_channels=256)
        self.block4_contract=double_conv(in_channels=256,out_channels=512)
        self.block5_contract=double_conv(in_channels=512,out_channels=1024)
        self.maxpool=nn.MaxPool2d(kernel_size=2,stride=2)

        # Going back up
        self.upconv1=nn.ConvTranspose2d(in_channels=1024,out_channels=512,kernel_size=2,stride=2)
        self.block1_expand=double_conv(in_channels=1024,out_channels=512)
        self.upconv2=nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=2,stride=2)
        self.block2_expand=double_conv(in_channels=512,out_channels=256)
        self.upconv3=nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=2,stride=2)
        self.block3_expand=double_conv(in_channels=256,out_channels=128)
        self.upconv4=nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=2,stride=2)
        self.block4_expand=double_conv(in_channels=128,out_channels=64)

        # Classification channels
        self.classification=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=classes,kernel_size=1,stride=1),
            nn.Sigmoid()
        )
        

    def forward(self,x):

        # Saving layers for skip connections (clayer is contracting layer)
        clayer1=self.block1_contract(x)
        clayer2=self.block2_contract(self.maxpool(clayer1))
        clayer3=self.block3_contract(self.maxpool(clayer2))
        clayer4=self.block4_contract(self.maxpool(clayer3))

        # Going back up (expanding layers)
        x=self.upconv1(self.block5_contract(self.maxpool(clayer4)))
        x=torch.cat([x,self.crop(clayer4,x)],dim=1)
        x=self.block1_expand(x)
        x=self.upconv2(x)
        x=torch.cat([x,self.crop(clayer3,x)],dim=1)
        x=self.block2_expand(x)
        x=self.upconv3(x)
        x=torch.cat([x,self.crop(clayer2,x)],dim=1)
        x=self.block3_expand(x)
        x=self.upconv4(x)
        x=torch.cat([x,self.crop(clayer1,x)],dim=1)
        x=self.block4_expand(x)
        return self.classification(x)
    
    # Claude gave this fn:
    def crop(self, encoder_feat, decoder_feat):
        _, _, H, W = decoder_feat.shape  # target size
        _, _, h, w = encoder_feat.shape  # source size (bigger)
        start_h = (h - H) // 2
        start_w = (w - W) // 2
        return encoder_feat[:, :, start_h:start_h+H, start_w:start_w+W]

        
def double_conv(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3),
            nn.ReLU(),
        )