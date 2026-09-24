from torch import nn
import torch
from matplotlib import pyplot as plt
import math
import random
import torch.nn.functional as F

class U_Net(nn.Module):
    def __init__(self,size,classes:int,color:bool=True,kernel_size=3):
        super().__init__()
        if color:
            self.channels=3
        else:
            self.channels=1
        # Size initially at n
        self.block1_contract=double_conv(in_channels=self.channels,out_channels=64,kernel_size=kernel_size)
        self.block2_contract=double_conv(in_channels=64,out_channels=128,kernel_size=kernel_size)
        self.block3_contract=double_conv(in_channels=128,out_channels=256,kernel_size=kernel_size)
        self.block4_contract=double_conv(in_channels=256,out_channels=512,kernel_size=kernel_size)
        self.block5_contract=double_conv(in_channels=512,out_channels=1024,kernel_size=kernel_size)
        self.maxpool=nn.MaxPool3d(kernel_size=2,stride=2)
        self.dropout = nn.Dropout3d(p=0.5)

        # Going back up
        self.upconv1=nn.ConvTranspose3d(in_channels=1024,out_channels=512,kernel_size=2,stride=2)
        self.block1_expand=double_conv(in_channels=1024,out_channels=512,kernel_size=kernel_size)
        self.upconv2=nn.ConvTranspose3d(in_channels=512,out_channels=256,kernel_size=2,stride=2)
        self.block2_expand=double_conv(in_channels=512,out_channels=256,kernel_size=kernel_size)
        self.upconv3=nn.ConvTranspose3d(in_channels=256,out_channels=128,kernel_size=2,stride=2)
        self.block3_expand=double_conv(in_channels=256,out_channels=128,kernel_size=kernel_size)
        self.upconv4=nn.ConvTranspose3d(in_channels=128,out_channels=64,kernel_size=2,stride=2)
        self.block4_expand=double_conv(in_channels=128,out_channels=64,kernel_size=kernel_size)

        # Classification channels
        self.classification=nn.Sequential(
            nn.Conv3d(in_channels=64,out_channels=classes,kernel_size=1,stride=1),
            # nn.Sigmoid()
        )
        

    def forward(self,x):

        # Saving layers for skip connections (clayer is contracting layer)
        clayer1=self.block1_contract(x)
        clayer2=self.block2_contract(self.maxpool(clayer1))
        clayer3=self.block3_contract(self.maxpool(clayer2))
        clayer4=self.block4_contract(self.maxpool(clayer3))
        clayer4=self.dropout(clayer4)

        # Going back up (expanding layers)
        x=self.block5_contract(self.maxpool(clayer4))
        x=self.dropout(x)
        x=self.upconv1(x)
        x=torch.cat([x,clayer4],dim=1)
        x=self.block1_expand(x)
        x=self.upconv2(x)
        x=torch.cat([x,clayer3],dim=1)
        x=self.block2_expand(x)
        x=self.upconv3(x)
        x=torch.cat([x,clayer2],dim=1)
        x=self.block3_expand(x)
        x=self.upconv4(x)
        x=torch.cat([x,clayer1],dim=1)
        x=self.block4_expand(x)
        return self.classification(x)
    


        
def double_conv(in_channels,out_channels,kernel_size=3):
    return nn.Sequential(
            nn.Conv3d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size, padding="same"),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels=out_channels,out_channels=out_channels,kernel_size=kernel_size, padding="same"),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(),
        )

class U_Netx2(nn.Module):
    def __init__(self, size, classes, color = True, kernel_size=3):
        super().__init__()
        self.Unet1=U_Net(size=size,classes=classes,color=color,kernel_size=kernel_size)
        self.Unet2=U_Net(size=size,classes=classes,color=color,kernel_size=kernel_size)
        self.comblayer=nn.Conv3d(in_channels=classes*2,out_channels=classes,kernel_size=1,stride=1)

    def forward(self, x):
        output1=self.Unet1(x)
        output2=self.Unet2(x)
        catoutput=torch.cat([output1,output2],dim=1)
        return self.comblayer(catoutput)