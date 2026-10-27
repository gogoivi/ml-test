import torch
from model import U_Net
import torchvision
from torchvision.transforms import ToTensor
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
from modified_cldice_3d import soft_cldice_loss
import torch.nn.functional as F
from loss_fn import merge
from model import U_Net,U_Net_small

# Constants
size=48
epochs=120
batch_size=20
kernel_size=3
root_dir=r""
path_model_a=""
path_model_b=""
T=1 #temperature
alpha=0.5

# Setup random seed
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model Declaration
teacher_a = U_Net(size=size,classes=3,color=False,kernel_size=kernel_size)
teacher_b = U_Net_small(size=size,classes=3,color=False,kernel_size=kernel_size)
# Loaded best pseudo labeled models
a_dict=torch.load(path_model_a)
a_dict=a_dict['model_state_dict']
b_dict=torch.load(path_model_b)
b_dict=b_dict['model_state_dict']
teacher_a.load_state_dict(a_dict)
teacher_b.load_state_dict(b_dict)
teacher_a.eval()
teacher_b.eval()
for param in teacher_a.parameters():
    param.requires_grad = False
for param in teacher_b.parameters():
    param.requires_grad = False
student=U_Net(size=size,classes=3,color=False,kernel_size=kernel_size)
teacher_a.to(device)
teacher_b.to(device)
student.to(device)

transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize((size,size)),
    torchvision.transforms.ToTensor()
])

weights = compute_class_weights(root_dir, num_classes=3)

# Create dataloaders
train_dataloader, test_dataloader = get_dataloaders(root_dir, batch_size=2)

optimizer=torch.optim.Adamax(params=student.parameters(),lr=lr,weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=5, factor=0.5
)

CE_Loss=nn.CrossEntropyLoss(weight=weights)

dice_loss_3d = DiceLoss(
    include_background=False,
    to_onehot_y=False,
    softmax=True,
    smooth_nr=1e-5,
    smooth_dr=1e-5
)
topo_loss = TopologyAwareLoss(alpha=5e-6, beta=1e-4, warmup_epochs=warmup_epochs)


#Loss fn basic idea (x is dataloader output)
#Not sure if I have to softmax?
with torch.no_grad():
    teacher_logits_a = teacher_a(x)
    teacher_logits_b = teacher_b(x)

# Should also make it so teacher logits and softmax both use y if they both agree (eg it is at 100% for that class if they agree - and same thing if y is anything except background - but if gt says background then use teacher logits instead)
#  This should become the new gt that everything uses - except for the hard loss (needs grounding in original gt)

#Pred value: X Batch 3 channel 48^3 img
student_logits=student(x)

studentlog_prob = F.log_softmax(student_logits / T, dim=1)   # log-probabilities
teacher_prob_a = F.softmax(teacher_logits_a / T, dim=1)    # probabilities
teacher_prob_b=F.softmax(teacher_logits_b / T,dim=1)
soft_y,hard_y=merge(y,teacher_prob_a,teacher_prob_b,[0.8,0.8])

kl_loss=F.kl_div(studentlog_prob,soft_y,reduction='batchmean')*T*T # Since the KD paper says to mult by T^2

soft_loss=(1-alpha)*(kl_loss+soft_cldice_loss(student_logits,soft_y,class_weights=weights)+topo_loss(student_logits,hard_y))
hard_loss=(alpha)*(CE_Loss(student_logits,y)+soft_cldice_loss(student_logits,y,class_weights=weights)+topo_loss(student_logits,y)) 













