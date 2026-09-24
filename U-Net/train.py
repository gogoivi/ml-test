import torch
from model import U_Net,U_Netx2
import torchvision
from torchvision.transforms import ToTensor
from tqdm import tqdm
from torch import nn
from pathlib import Path
from torch.utils.data import Dataset
from load_data import compute_class_weights,get_dataloaders
from torch.utils.tensorboard import SummaryWriter
import os
from monai.losses import DiceLoss
from torch.amp import autocast
from loss_fn import TopologyAwareLoss
import numpy as np
from torch.amp import autocast, GradScaler
from datetime import datetime



def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    
    save_path = target_dir_path / model_name
    torch.save(model.state_dict(), save_path)


size=48
epochs=100
batch_size=20
kernel_size=3
U_Net_Name="U_Net(Topo_Loss)"
warmup_epochs=10
lr=0.0005
timestamp = datetime.now().strftime("%m%d_%H%M")
# How many to skip before topo loss starts
N=2

# Patience counter idea was claude
best_val_loss = float('inf')
patience = 7  # Stop if no improvement for 7 eval cycles (21 epochs)
patience_counter = 0
best_model_name=""
target_dir = "U-Net_models"

# Copied from some of my previously written code
if __name__=="__main__":
    scaler = GradScaler()
    # Create writer (logs to ./runs folder)
    run_name = f"runs/{timestamp}_{U_Net_Name}_lr{lr}_bs{batch_size}"
    writer = SummaryWriter(log_dir=run_name)

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

    u_net=U_Net(size=size,classes=3,color=False,kernel_size=kernel_size)
    # u_net=U_Netx2(size=size,classes=3,color=False,kernel_size=kernel_size)
    u_net.to(device)

    optimizer=torch.optim.Adamax(params=u_net.parameters(),lr=lr,weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    loss_fn=nn.CrossEntropyLoss(weight=weights.to(device),ignore_index=3)  # [background, nerve,vessel]
    dice_loss_3d = DiceLoss(
        include_background=False,
        to_onehot_y=True,
        softmax=True,
        smooth_nr=1e-5,
        smooth_dr=1e-5
    )
    topo_loss = TopologyAwareLoss(alpha=5e-6, beta=1e-4, warmup_epochs=warmup_epochs)

    for i in tqdm(range(epochs)):
        u_net.train()
        train_loss_sum=0
        train_sample_count=0
        for batch, (X,y) in enumerate(train_dataloader):
            X=X.to(device)
            y=y.to(device)
            with autocast(device_type=device):
                x_pred=u_net(X)
                ce_loss=loss_fn(x_pred,y)
                dice_loss=dice_loss_3d(x_pred,y.unsqueeze(1))
            T_loss=topo_loss(x_pred, y, i, N=N)

            # Check for problems BEFORE they become NaN
            if ce_loss.item() > 100:
                print("  WARNING: CE loss spiking!")
            if dice_loss.item() > 10:
                print("  WARNING: Dice loss spiking!")
            if T_loss > 10:
                print("  WARNING: Omega spiking!")

            loss = (ce_loss + dice_loss) * T_loss

            if torch.isnan(loss):
                print("  >>> LOSS IS NAN <<<")
                print(f"  CE nan: {torch.isnan(ce_loss)}")
                print(f"  Dice nan: {torch.isnan(dice_loss)}")
                print(f"  Omega nan or inf: {np.isnan(T_loss) or np.isinf(T_loss)}")
                raise ValueError("NaN detected - stopping to debug")
            
            optimizer.zero_grad()
            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(u_net.parameters(), max_norm=1.0)
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(u_net.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            batch_size = X.shape[0]
            train_loss_sum += loss.item() * batch_size
            train_sample_count += batch_size
            

            

        if i%3==0 and i>warmup_epochs:
            u_net.eval()
            with torch.no_grad():
                sample_count = 0
                loss_sum=0
                for (X,y) in test_dataloader:
                    X=X.to(device)
                    y=y.to(device)
                    with autocast(device_type=device):
                        x_pred=u_net(X)
                        ce_loss=loss_fn(x_pred,y)
                        dice_loss=dice_loss_3d(x_pred,y.unsqueeze(1))
                        loss = (ce_loss + dice_loss)
                    batch_size = X.shape[0]
                    loss_sum += loss.item() * batch_size
                    sample_count += batch_size
            print(f"Epoch {i}, Batch {batch}")
            print(f"  CE: {ce_loss.item():.6f}")
            print(f"  Dice: {dice_loss.item():.6f}")
            print(f"  Pred min/max: {x_pred.min().item():.4f} / {x_pred.max().item():.4f}")
            train_loss=train_loss_sum/train_sample_count
            val_loss = loss_sum / sample_count
            scheduler.step(val_loss)
            writer.add_scalar('Loss/train', train_loss, i)
            writer.add_scalar('Loss/val', val_loss, i)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], i)  # Track LR too

            if val_loss<best_val_loss:
                best_val_loss=val_loss
                patience_counter=0
                save_model(model=u_net,target_dir="U-Net_models",model_name=f"MRI_{U_Net_Name}_E{i}_K{kernel_size}")
                best_model_name=f"MRI_{U_Net_Name}_E{i}_K{kernel_size}"
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
