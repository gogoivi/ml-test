import torch
from model import U_Net
import torchvision
from torchvision.transforms import ToTensor
from tqdm import tqdm
from torch import nn
from pathlib import Path
from torch.utils.data import Dataset
from load_data import compute_class_weights,get_dataloaders
from torch.utils.tensorboard import SummaryWriter
import os


def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    
    save_path = target_dir_path / model_name
    torch.save(model.state_dict(), save_path)


size=48
epochs=100
batch_size=20

# Patience counter idea was claude
best_val_loss = float('inf')
patience = 5  # Stop if no improvement for 5 eval cycles (15 epochs)
patience_counter = 0
best_model_name=""
target_dir = "U-Net_models"

# Copied from some of my previously written code
if __name__=="__main__":
    # Create writer (logs to ./runs folder)
    writer = SummaryWriter()

    ROOT_DIR = r"C:\Users\gogoi\Desktop\ml-test\2023 Patients"
    # Setup random seed
    RANDOM_SEED = 42

    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize((size,size)),
        torchvision.transforms.ToTensor()
    ])

    weights = compute_class_weights(ROOT_DIR, num_classes=3)

    # Create dataloaders
    train_dataloader, test_dataloader = get_dataloaders(ROOT_DIR, batch_size=2)

    u_net=U_Net(size=size,classes=3,color=False)
    u_net.to(device)

    optimizer=torch.optim.Adamax(params=u_net.parameters(),lr=0.001)
    loss_fn=nn.CrossEntropyLoss(weight=weights.to(device),ignore_index=3)  # [background, nerve,vessel]

    for i in tqdm(range(epochs)):
        u_net.train()
        train_loss_sum=0
        train_sample_count=0
        for batch, (X,y) in enumerate(train_dataloader):
            X=X.to(device)
            y=y.to(device)
            x_pred=u_net(X)
            loss=loss_fn(x_pred,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_size = X.shape[0]
            train_loss_sum += loss.item() * batch_size
            train_sample_count += batch_size
            

            

        if i%3==0:
            u_net.eval()
            with torch.no_grad():
                sample_count = 0
                loss_sum=0
                for (X,y) in test_dataloader:
                    X=X.to(device)
                    y=y.to(device)
                    x_pred=u_net(X)
                    loss=loss_fn(x_pred,y)
                    batch_size = X.shape[0]
                    loss_sum += loss.item() * batch_size
                    sample_count += batch_size
            train_loss=train_loss_sum/train_sample_count
            val_loss = loss_sum / sample_count
            writer.add_scalar('Loss/train', train_loss, i)
            writer.add_scalar('Loss/val', val_loss, i)

            if val_loss<best_val_loss:
                best_val_loss=val_loss
                patience_counter=0
                save_model(model=u_net,target_dir="U-Net_models",model_name=f"MRI_E{i}")
                best_model_name=f"MRI_E{i}"
            else:
                patience_counter+=1
                if patience_counter>patience:
                    print(f"Stopping at Epoch {i}")
                    break

    file_path = os.path.join(target_dir, best_model_name)
    # Using best model
    u_net.load_state_dict(torch.load(file_path))

    #  Claude Function
    from load_data import evaluate_and_visualize

    evaluate_and_visualize(u_net, test_dataloader, device, num_samples=3)
