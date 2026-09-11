from torch import nn
import torch
from matplotlib import pyplot as plt

class VAE(nn.Module):
    def __init__(self,z_size:int,height,width,color:bool=True):
        super().__init__()
        self.z_size=z_size
        self.height=height
        self.width=width
        if color:
            self.channels=3
        else:
            self.channels=1
        self.encoder=nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=width*height*self.channels,out_features=784), # Input layer goes to 784 for first hidden layer
            nn.ReLU(),
            nn.Linear(in_features=784,out_features=256),
            nn.ReLU()
        )
        self.mu=nn.Linear(in_features=256,out_features=z_size)  # Seperating mu and sigma
        self.log_var=nn.Linear(in_features=256,out_features=z_size)
        self.block_decoder=nn.Sequential(
            nn.Linear(in_features=z_size,out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features=784),
            nn.ReLU(),
            nn.Linear(in_features=784,out_features=height*width*self.channels), # Output to same size as input
            nn.Sigmoid() #Since we want probabilities for each pixel between 0 and 1
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
        return self.block_decoder(encoder_output), mu, log_var

    def get_info(self):
        return int(self.z_size),int(self.channels),int(self.height),int(self.width)

# This function was Claude as well
def vae_loss(x, x_reconstructed, mu, log_var):
    # Flatten x to match x_reconstructed shape
    x_flat = x.view(x.shape[0], -1)
    
    # Reconstruction loss (BCE)
    reconstruction_loss = nn.functional.binary_cross_entropy(
        x_reconstructed, x_flat, reduction='sum'
    )
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + log_var - mu**2 - torch.exp(log_var))
    
    return reconstruction_loss + kl_loss

def generate_image(model:VAE,device):
    model.eval()
    z_size,channels,height,width=model.get_info()
    z_vector=torch.randn(z_size)
    z_vector=z_vector.to(device)
    output=model.block_decoder(z_vector)
    output=output.to('cpu')
    image_tensor = output.view(height, width, channels)
    image_numpy = image_tensor.detach().numpy()
    plt.imshow(image_numpy)
    plt.axis("off")
    plt.show()


