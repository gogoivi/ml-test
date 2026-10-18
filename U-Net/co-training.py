from model import U_Net,U_Net_small
import datetime
from torch import nn
import torch
from torch.utils.data import Dataset, DataLoader
from load_data import MRIDataset
import torch.nn.functional as F

size=48
epochs=120
batch_size=20
kernel_size=3
U_Net_Name="U_Net(Topo_Loss_clDice)"
warmup_epochs=10
lr=0.0005
timestamp = datetime.now().strftime("%m%d_%H%M")
# How many to skip before topo loss starts
N=2

# Patience counter idea was claude
best_val_loss = float('inf')
patience = 9  # Stop if no improvement for 9 eval cycles (27 epochs)
patience_counter = 0
best_model_name=""
target_dir = "U-Net_models"
ROOT_DIR = r"C:\Users\gogoi\Desktop\ml-test\2023 Patients"

        
class CoTrainer:
    def __init__(self,model_a:nn.Module,model_b:nn.Module,get_dataloader_func,thresholds:list=[0.99, 0.95],batch_size:int=20,stagnation:int=5):
        """
        Docstring for __init__
        
        :param self: Description
        :param model_a: Description
        :type model_a: nn.Module
        :param model_b: Description
        :type model_b: nn.Module
        :param get_dataloader_func: Function that should return three dataloaders in the following order: train_loader, train_no_aug_loader,test_loader and should require a root_dir and psuedo_label input
        get item for each loader should also return idx of the data being used in order to properly make the pseudo labels dict
        :param thresholds: Description
        :type thresholds: list
        :param batch_size: Description
        :type batch_size: int
        :param stagnation: How many epochs of no improvement in validation before the models start to pseudo label
        :type stagnation: int
        """
        self.model_a = model_a
        self.model_b = model_b 
        self.pseudo_labels = {}  # Track pseudo-labels
        self.thresholds=thresholds
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_loader,self.train_loader_no_aug,self.test_loader=get_dataloader_func(root_dir=ROOT_DIR,batch_size=batch_size,pseudo_labels=self.pseudo_labels)
        
    def train_epoch(self):
        for batch, (X,y,idx) in enumerate(self.train_loader):
            X=X.to(self.device)
            y=y.to(self.device)


    def generate_pseudo_labels(self, sample_idx, pred_a, pred_b, gt):
        """
        Args:
            sample_idx: name of the specific example being inputted
            pred_a: [C, D, H, W] - softmax from model_a logits
            pred_b: [C, D, H, W] - softmax from model_b logits
            gt: [D, H, W] - ground truth labels (0, 1, or 2)
        """
        
        is_background = (gt == 0)
        
        # Initialize if needed
        if sample_idx not in self.pseudo_labels:
            self.pseudo_labels[sample_idx] = {
                'mask': torch.zeros_like(gt, dtype=torch.bool),
                'classes': torch.zeros_like(gt, dtype=torch.long),
                'confidence': torch.zeros_like(gt, dtype=torch.float),
            }
        
        num_foreground_classes = pred_a.shape[0] - 1
        
        for c in range(num_foreground_classes):
            class_idx = c + 1 
            
            # Both models confident about this class
            confident = (pred_a[class_idx] > self.thresholds[c]) & (pred_b[class_idx] > self.thresholds[c])
            
            # Only where GT is background
            new_mask = is_background & confident
            
            # Calculate confidence for these pixels
            avg_confidence = (pred_a[class_idx] + pred_b[class_idx])/2  # Average of both model's predictions for class_idx

            self.pseudo_labels[sample_idx]['mask'] |= new_mask
            self.pseudo_labels[sample_idx]['classes'][new_mask] = class_idx
            self.pseudo_labels[sample_idx]['confidence'][new_mask] = avg_confidence[new_mask]

    def remove_uncertain_pseudos(self,sample_idx, pred_a, pred_b):

        num_foreground_classes = pred_a.shape[0] - 1

        # Nothing to check
        if sample_idx not in self.pseudo_labels:
            return 0
        
        # Iterate through classes
        for c in range(num_foreground_classes):
            class_idx = c + 1 
            class_mask = self.pseudo_labels[sample_idx]['mask'] & (self.pseudo_labels[sample_idx]['classes'] == class_idx)

            # Nothing for this class
            if not class_mask.any():
                        continue
            remove = ((pred_a[class_idx] < (self.thresholds[c]-0.1)) | (pred_b[class_idx] < (self.thresholds[c]-0.1))) & class_mask
            self.pseudo_labels[sample_idx]['confidence'][remove]=0
            self.pseudo_labels[sample_idx]['mask'][remove]=False
            self.pseudo_labels[sample_idx]['classes'][remove]=0

    def update_all_pseudo_labels(self):
        """Run through training data w/o augmentation to generate pseudo-labels."""
        self.model_a.eval()
        self.model_b.eval()
        
        with torch.no_grad():
            for X, y, indices in self.train_loader_no_aug:
                X, y = X.to(self.device), y.to(self.device)
                
                pred_a = F.softmax(self.model_a(X), dim=1)
                pred_b = F.softmax(self.model_b(X), dim=1)
                
                for i in range(X.shape[0]):
                    idx = indices[i].item()
                    self.generate_pseudo_labels(idx, pred_a[i], pred_b[i], y[i])
                    self.remove_uncertain_pseudos(idx,pred_a[i],pred_b[i])
        
        self.model_a.train()
        self.model_b.train()
    
    def train(self, num_epochs):
        for epoch in range(num_epochs):
            # Train both models
            loss_a = self.train_epoch(self.model_a, ...)
            loss_b = self.train_epoch(self.model_b, ...)
            
            # Check if stagnated
            if self.check_stagnation():
                self.generate_pseudo_labels(...)
                self.reset_patience()
            
            # Periodically clean up bad pseudos
            if epoch % 10 == 0:
                self.remove_uncertain_pseudos()

    def merge_gt_with_pseudo(self, y, sample_indices,weight:int=0.8):
        """
        Combine ground truth with pseudo-labels 
        
        Args:
            y: [B, D, H, W] - ground truth labels
            sample_indices: [B] - which samples these are
        
        Returns:
            y_merged: [B, D, H, W] - labels with pseudo-labels added
        """
        y_merged = y.clone()
        
        for i in range(y.shape[0]):
            idx = sample_indices[i].item()
            
            if idx not in self.pseudo_labels:
                continue
            
            pseudo = self.pseudo_labels[idx]
            
            # pseudo-labels AND gt is background
            apply_mask = pseudo['mask'] & (y[i] == 0)
            
            # Override labels
            y_merged[i][apply_mask] = pseudo['classes'][apply_mask]

        
        return y_merged