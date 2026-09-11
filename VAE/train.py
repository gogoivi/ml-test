import torch
from model import VAE, vae_loss,generate_image
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
epochs=30

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
    vae=VAE(z_size=25,height=height,width=width,color=False)
    vae.to(device)

    optimizer=torch.optim.Adamax(params=vae.parameters(),lr=0.01)

    for i in tqdm(range(epochs)):
        vae.train()
        loss_sum=0
        for batch, (X,_) in enumerate(train_dataloader):
            X=X.to(device)
            x_pred,mu,log_var=vae(X)
            loss=vae_loss(x=X,x_reconstructed=x_pred,mu=mu,log_var=log_var)
            loss_sum+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if i%5==0:
            print(loss_sum/len(train_dataloader))

    save_model(model=vae,target_dir="VAE_models",model_name="MNIST_Z25_E30")

    for i in range(3):
        generate_image(vae,device)
    