import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from monai.losses import DiceLoss
from loss_fn import TopologyAwareLoss
from modified_cldice_3d import soft_cldice_loss
from load_data import compute_class_weights
from datetime import datetime
from pathlib import Path
import os
import numpy as np
from load_data import get_dataloaders
from model import U_Net_small,U_Net
from cotraining import CoTrainer
from KD import KD
size=48
epochs=120
batch_size=20
kernel_size=3
# VERIFY: Adjust paths and hyperparameters
ROOT_DIR = "/home/vgogoi1/projects/TN_Research/2023 Patients"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize pseudo_labels dict FIRST
pseudo_labels = {}

# Get dataloaders with shared pseudo_labels
train_loader, train_loader_no_aug, test_loader = get_dataloaders(
    ROOT_DIR,
    batch_size=batch_size,
    pseudo_labels=pseudo_labels
)

model_a = U_Net(size=size,classes=3,color=False,kernel_size=kernel_size)
model_b = U_Net_small(size=size,classes=3,color=False,kernel_size=kernel_size)
path_a=r"/home/vgogoi1/projects/TN_Research/U-Net/checkpoints/Rips_TopoLoss(300 epochs)_20261027_004906/model_a_latest.pt"
path_b=r"/home/vgogoi1/projects/TN_Research/U-Net/checkpoints/Rips_TopoLoss(300 epochs)_20261027_004906/model_b_latest.pt"
# Create trainer

# trainer = CoTrainer(
#     model_a=model_a,
#     model_b=model_b,
#     train_loader=train_loader,
#     train_loader_no_aug=train_loader_no_aug,
#     test_loader=test_loader,
#     device=device,
#     root_dir=ROOT_DIR,
#     lr_a=1e-4,
#     lr_b=1e-3,
#     thresholds=[0.99, 0.95],
#     stagnation_patience=6,
#     experiment_name="Rips_TopoLoss(300 epochs)" 
# )

# # Link the shared dict
# trainer.pseudo_labels = pseudo_labels

# # Train
# trainer.train(total_epochs=300, warmup_epochs=15, update_frequency=1)

trainer= KD(model_a=model_a,model_b=model_b,path_model_a=path_a,path_model_b=path_b,root_dir=ROOT_DIR,experiment_name="KD with CCTopoLossTeachers",stagnation_patience=10,alpha=0.15)
trainer.train(total_epochs=300,warmup_epochs=15,update_frequency=1)