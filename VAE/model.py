from torch import nn
import torch

class VAE(nn.Module):
    def __init__(self,z_size,height,width,color:bool=True):
        super().__init__()
        if color:
            channels=3
        else:
            channels=2
        self.block1_encoder=nn.Sequential(
            nn.Linear(in_features=width*height*channels,out_features=36),
            nn.ReLU(),
            nn.Linear(in_features=36,out_features=256),
            nn.ReLU
        )
        self.mu=nn.Linear(in_features=256,out_features=z_size)  # Seperating mu and sigma to 
        self.log_var=nn.Linear(in_features=256,out_features=z_size)
        self.block_decoder=nn.Sequential(
            nn.Linear(in_features=z_size*2,out_features=36),
            nn.ReLU(),
            nn.Linear(in_features=36,out_features=258),
            nn.ReLU(),
            nn.Linear(in_features=258,out_features=height*width)
        )

    # This function was Claude
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)  # convert log_var to std
        eps = torch.randn_like(std)      # sample random noise
        z = mu + std * eps               # reparameterization trick!
        return z
    
    def forward(self,x):
        hidden=self.encoder(x)
        mu=self.mu(hidden)
        log_var = self.log_var(hidden)
        encoder_output=self.reparameterize(mu,log_var)
        return self.block_decoder(encoder_output)


