import torch
from model import U_Net
import torchvision
from torchvision.transforms import ToTensor
from tqdm import tqdm
from torch import nn
from pathlib import Path
from torch.utils.data import Dataset


def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    
    save_path = target_dir_path / model_name
    torch.save(model.state_dict(), save_path)


size=32
epochs=20
batch_size=20

# Copied from some of my previously written code
if __name__=="__main__":
    # Setup random seed
    RANDOM_SEED = 42

    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    resize=torchvision.transforms.Compose([
        torchvision.transforms.Resize((size,size)),
        torchvision.transforms.ToTensor()
    ])

    # Put in Dataloader
    class DummyMRIDataset(Dataset):
        def __init__(self, num_samples, volume_size, num_classes):
            self.num_samples = num_samples
            self.volume_size = volume_size  # e.g., (64, 64, 64)
            self.num_classes = num_classes
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Random input volume (1 channel, grayscale MRI)
            x = torch.randn(1, *self.volume_size)
            
            # Random segmentation mask - integer labels
            # Shape should be ??? (no channel dim!)
            # Values should be integers from 0 to ???
            y = torch.randint(low=0, high=self.num_classes, size=[*self.volume_size])
            
            return x, y

    # Create datasets
    train_dataset = DummyMRIDataset(num_samples=100, volume_size=(size,size,size), num_classes=3)
    test_dataset = DummyMRIDataset(num_samples=20, volume_size=(size,size,size), num_classes=3)

    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    u_net=U_Net(size=size,classes=3,color=False)
    u_net.to(device)

    optimizer=torch.optim.Adamax(params=u_net.parameters(),lr=0.01)
    loss_fn=nn.CrossEntropyLoss(weight=torch.tensor([0.04, 0.32,0.64]))  # [background, nerve,vessel]
    u_net.train()
    (X,_)=next(iter(train_dataloader))
    X=X.to(device)
    x_pred=u_net(X)
    print(x_pred)
    # for i in tqdm(range(epochs)):
    #     u_net.train()
    #     loss_sum=0
    #     for batch, (X,_) in enumerate(train_dataloader):
    #         X=X.to(device)
    #         x_pred=u_net(X)
    #         loss=loss_fn(x_pred)
    #         loss_sum+=loss.item()
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     if i%5==0:
    #         print(loss_sum/len(train_dataloader))

    # save_model(model=vae,target_dir="VAE_models",model_name="MNIST_Z25_E50")