from torch import nn
import torch
from matplotlib import pyplot as plt
import math
import random

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
        nn.LeakyReLU(0.2),
        nn.Linear(in_features=hidden_size,out_features=hidden_size*2),
        nn.LeakyReLU(0.2),
        nn.Linear(in_features=hidden_size*2,out_features=hidden_size*2),
        nn.LeakyReLU(0.2),
        nn.Linear(in_features=hidden_size*2,out_features=height*width*self.channels),
        nn.Sigmoid() # Probs between 0 and 1
        )
    def forward(self,z_vector):
        return self.MLP(z_vector)
    
    def get_info(self):
        return int(self.z_size),int(self.channels),int(self.height),int(self.width)

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
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=hidden_size,out_features=hidden_size*2),
            nn.LeakyReLU(0.2),
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
    # Clamp so I dont get -inf in the computation
    ml_guess=torch.clamp(ml_guess, min=1e-7, max=1-1e-7)
    real_guess=torch.clamp(real_guess, min=1e-7, max=1-1e-7)
    # Random number between the two so its target is always moving and thus doesnt stabilize and gives the generator a proper gradient
    return -1*torch.log(ml_guess).mean(),-1*(torch.log(real_guess)+torch.log(random.uniform(0.8, 1.0)-ml_guess)).mean()


# Taken from VAE code and adapted
def generate_img(model:Generator,device):
    model.eval()
    z_size,channels,height,width=model.get_info()
    z_vector=torch.randn(z_size)
    z_vector=z_vector.to(device)
    output=model.forward(z_vector)
    output=output.to('cpu')
    image_tensor = output.view(height, width, channels)
    image_numpy = image_tensor.detach().numpy()
    plt.imshow(image_numpy)
    plt.axis("off")
    plt.show()
