from torch import nn
import torch
from matplotlib import pyplot as plt
import math

class Generator(nn.Module):
    def __init__(self,z_size,height,width,hidden_size:int=128,color:bool=True):
        super().__init__()
        self.z_size=z_size
        self.height=height # Output img height
        self.width=width # Output img width
        self.hidden_size=hidden_size # Neurons per hidden layer
        if color:
            self.channels=3
        else:
            self.channels=1

        self.MLP=nn.Sequential(
        nn.Linear(in_features=z_size,out_features=hidden_size),
        nn.ReLU(),
        nn.Linear(in_features=hidden_size,out_features=hidden_size*2),
        nn.ReLU(),
        nn.Linear(in_features=hidden_size*2,out_features=height*width*self.channels),
        nn.Sigmoid() # Probs between 0 and 1
        )
    def forward(self,z_vector):
        return self.MLP(z_vector)

class Descriminator(nn.Module):
    """
    Should guess 1 if real img
    """

    def __init__(self,height,width,hidden_size:int=128,color:bool=True):
        super().__init__()

        if color:
            self.channels=3
        else:
            self.channels=1

        self.MLP=nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=height*width*self.channels,out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size,out_features=hidden_size*2),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size*2,out_features=1),
            nn.Sigmoid() # Probs between 0 and 1
        )
    def forward(self,X):
        return self.MLP(X)

def GAN_loss(ml_guess,real_guess):
    """
    Returns Generator_loss, Descriminator_loss
    Generator loss is log(D(z)) Descriminator goal is to get this to 1
    Descriminator loss is log(D(x))+log(1-D(z)) Generator goal is to get this to 0
    """
    return torch.log(ml_guess),-1*(torch.log(real_guess)+torch.log(1-ml_guess))
