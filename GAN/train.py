# Compied up to dataloader from VAE since it isnt anything different
import torch
from model import GAN_loss,Descriminator,Generator
import torchvision
from torchvision.transforms import ToTensor
from tqdm import tqdm
from torch import nn
from pathlib import Path

def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    
    save_path = target_dir_path / model_name
    torch.save(model.state_dict(), save_path)


height=28
width=28
epochs=50
z_size=25

# Copied from some of my previously written code
if __name__=="__main__":
    # Setup random seed
    RANDOM_SEED = 42

    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    resize=torchvision.transforms.Compose([
        torchvision.transforms.Resize((height,width)),
        torchvision.transforms.ToTensor()
    ])

    train_data=torchvision.datasets.MNIST(root="data",train=True,transform=resize,target_transform=None,download=True)
    test_data=torchvision.datasets.MNIST(root="data",train=False,transform=resize,target_transform=None,download=True)

    batch=512
    train_dataloader =torch.utils.data.DataLoader(train_data,batch_size=batch)
    test_dataloader=torch.utils.data.DataLoader(test_data,batch_size=batch)
    descriminator=Descriminator(height=height,width=width,color=False)
    descriminator.to(device)
    generator=Generator(z_size=z_size,height=height,width=width,color=False)
    generator.to(device)

    de_optimizer=torch.optim.Adamax(params=descriminator.parameters(),lr=0.01)
    ge_optimizer=torch.optim.Adamax(params=generator.parameters(),lr=0.01)

    for i in tqdm(range(epochs)):
        generator.train()
        descriminator.train()
        Des_loss_sum=0
        for batch, (X,_) in enumerate(train_dataloader):
            X=X.to(device)
            current_batch_size = X.shape[0]
            z_tensor = torch.normal(mean=0.0, std=1.0, size=(current_batch_size, z_size))
            z_tensor=z_tensor.to(device)
            ml_guess=descriminator(generator(z_tensor))
            real_guess=descriminator(X)
            Gen_loss,Des_loss=GAN_loss(ml_guess=ml_guess,real_guess=real_guess)
            Des_loss_sum+=Des_loss.item()
            de_optimizer.zero_grad()
            Des_loss.backward()
            de_optimizer.step()

        ge_optimizer.zero_grad()
        Gen_loss.backward()
        ge_optimizer.step()
        
        if i%5==0:
            print(Des_loss_sum/len(train_dataloader))
            print(Gen_loss.item()/len(train_dataloader))