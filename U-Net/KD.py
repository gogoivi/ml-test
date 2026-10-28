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



class KD:
    def __init__(self,model_a,model_b,path_model_a,path_model_b,root_dir:str,size=48,epochs=120,
                 batch_size=20,kernel_size=3,alpha:float=0.5,warmup_epochs:int=15,lr=0.0005,thresholds:list=[0.8,0.8],
                 stagnation_patience:int=5,save_dir="checkpoints",experiment_name="Knowledge Distill"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.student=U_Net(size=size,classes=3,color=False,kernel_size=kernel_size)
        self.teacher_a=model_a
        self.teacher_b=model_b
        a_dict=torch.load(path_model_a)
        a_dict=a_dict['model_state_dict']
        b_dict=torch.load(path_model_b)
        b_dict=b_dict['model_state_dict']
        self.teacher_a.load_state_dict(a_dict)
        self.teacher_b.load_state_dict(b_dict)
        self.teacher_a.eval()
        self.teacher_b.eval()
        self.thresholds = thresholds
        self.stagnation_patience = stagnation_patience
        for param in self.teacher_a.parameters():
            param.requires_grad = False
        for param in self.teacher_b.parameters():
            param.requires_grad = False
        teacher_a.to(device)
        teacher_b.to(device)
        student.to(device)
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((size,size)),
            torchvision.transforms.ToTensor()
        ])

        self.weights = compute_class_weights(root_dir, num_classes=3)

        # Create dataloaders
        self.train_dataloader, self.test_dataloader = get_dataloaders(root_dir, batch_size=2)

        optimizer=torch.optim.Adamax(params=self.student.parameters(),lr=lr,weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        self.student_scaler=GradScaler()

        self.CE_Loss=nn.CrossEntropyLoss(weight=self.weights)

        self.dice_loss_3d = DiceLoss(
            include_background=False,
            to_onehot_y=False,
            softmax=True,
            smooth_nr=1e-5,
            smooth_dr=1e-5
        )
        self.topo_loss = TopologyAwareLoss(alpha=5e-6, beta=1e-4, warmup_epochs=warmup_epochs)

        # Logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(save_dir) / f"{experiment_name}_{timestamp}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.save_dir / "tensorboard")

        # Tracking
        self.best_metric = float('inf')
        self.global_step = 0

        def train_loss(self,x,gt,pred_a,pred_b,T:int=1):
            """Can change to whatever loss is appropriate - right now am doing topo+CE (or KLDiv) +clDice+Dice"""
            soft_y,hard_y=merge(gt,F.softmax(pred_a/T,dim=1),F.softmax(pred_b/T,dim=1))
            studentlog_prob = F.log_softmax(x / T, dim=1)   # log-probabilities
            kl_loss=F.kl_div(studentlog_prob,soft_y,reduction='batchmean')*T*T
            topo=self.topo_loss(x,hard_y)
            cl=soft_cldice_loss(x,soft_y)
            dice=self.dice_loss_3d(x,hard_y)
            soft_loss=(1-self.alpha)*(kl_loss+topo+cl+dice)

            topo=self.topo_loss(x,gt)
            cl=soft_cldice_loss(x,gt)
            dice=self.dice_loss_3d(x,gt)
            CE=self.CE_Loss(x,gt)
            hard_loss=self.alpha*(topo+cl+dice+CE)
            return hard_loss+soft_loss

        def validation(self,x,y,T:int=1):
            """
            This is the validation loss - which is dice - at the moment will be using the hard merged y as the dice loss eval - but can change to gt if needed
            :param x: Takes in MRI data
            :param y: gt
            """
            self.student.eval()
            
            # Accumulators for Dice calculation
            intersection = {1: 0, 2: 0}  # nerve, vessel
            union = {1: 0, 2: 0}

            
            total_loss = 0
            
            with torch.no_grad():
                for X, y, idx in self.test_dataloader:
                    X = X.to(self.device)
                    y = y.to(self.device)
                    
                    with autocast(device_type='cuda'):
                        pred = self.student(X)
                        y_a=self.teacher_a
                        y_b=self.teacher_b

                    _,y=merge(y,F.softmax(y_a/T,dim=1),F.softmax(y_b/T,dim=1))
                    pred = pred.float()

                    # Simple CE+Dice for validation loss (no topo) but uses merged gt
                    val_loss = self.ce_loss_fn(pred, y) + self.dice_loss_fn(pred, y.unsqueeze(1))
                    
                    total_loss += val_loss.item()
                    
                    # Get predictions
                    pred_class = torch.argmax(pred, dim=1)  # [B, D, H, W]
                    
                    # Compute Dice per class
                    for c in [1, 2]:
                        pred_c = (pred_class == c)
                        gt_c = (y == c)
                        
                        intersection[c] += (pred_c & gt_c).sum().item()
                        union[c] += pred_c.sum().item() + gt_c.sum().item()
            
            # Compute Dice scores
            dice = {}
            for c in [1, 2]:
                dice[c] = (2 * intersection[c] + 1e-5) / (union[c] + 1e-5)
            
            avg_val_loss = total_loss / len(self.test_loader)
            
            return dice, avg_val_loss


