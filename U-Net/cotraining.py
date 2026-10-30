import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from monai.losses import DiceLoss
from loss_fn import TopologyAwareLoss, Cubical_Complex_Loss_Differentiable
from modified_cldice_3d import soft_cldice_loss
from load_data import compute_class_weights
from datetime import datetime
from pathlib import Path
import os
import numpy as np

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
    def __init__(
        self,
        model_a,
        model_b,
        train_loader,
        train_loader_no_aug,
        test_loader,
        device,
        root_dir,
        lr_a=1e-4,
        lr_b=1e-4,
        thresholds=[[0.99, 0.95]],  # [nerve, vessel]
        stagnation_patience=5,
        save_dir="checkpoints",
        experiment_name="cotraining",
        warmup_epochs=15,          # ADD: 0 for rounds after first
        pseudo_labels=None,        # ADD: carry forward between rounds
        epochs=20,                 # ADD: per-round epoch count
    ):
        """
        Docstring for __init__
        
        :param self: Description
        :param model_a: Description
        :type model_a: nn.Module
        :param model_b: Description
        :type model_b: nn.Module
        :param threshold_array: Description
        :type thresholds: Array of lists
        :param batch_size: Description
        :type batch_size: int
        :param stagnation: How many epochs of no improvement in validation before the models start to pseudo label
        :type stagnation: int
        """
        self.model_a = model_a
        self.model_b = model_b
        self.train_loader = train_loader
        self.train_loader_no_aug = train_loader_no_aug
        self.test_loader = test_loader
        self.device = device
        self.root_dir = root_dir
        self.threshold_array = thresholds
        self.stagnation_patience = stagnation_patience
        
        
        # Optimizers - adjust for different settings
        self.optimizer_a = torch.optim.AdamW(model_a.parameters(), lr=lr_a, weight_decay=1e-5)
        self.optimizer_b = torch.optim.SGD(model_b.parameters(), lr=lr_b, momentum=0.99, weight_decay=1e-5)
        
        # Instead of always creating empty dict:
        self.pseudo_labels = pseudo_labels if pseudo_labels is not None else {}
        
        # Store warmup and epochs
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        
        self.scheduler_a = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_a, T_max=epochs+warmup_epochs
        )
        self.scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_b, T_max=epochs+warmup_epochs
        )
        
        #scalers
        self.scaler_a = GradScaler()
        self.scaler_b = GradScaler()
        
        # Loss
        weights = compute_class_weights(root_dir, num_classes=3)
        self.ce_loss_fn = nn.CrossEntropyLoss(weight=weights.to(device), ignore_index=3)
        self.dice_loss_fn = DiceLoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            smooth_nr=1e-5,
            smooth_dr=1e-5
        )
        
        # Initialize topo loss with warmup
        # self.topo_loss_fn = TopologyAwareLoss(alpha=5e-6, beta=1e-4, warmup_epochs=warmup_epochs)
        self.topo_loss_fn=Cubical_Complex_Loss_Differentiable(warmup_epochs=self.warmup_epochs)
        
        # Logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(save_dir) / f"{experiment_name}_{timestamp}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.save_dir / "tensorboard")
        
        # Tracking
        self.best_metric_a = float('inf')
        self.best_metric_b = float('inf')
        self.global_step = 0

    # def compute_loss(self, pred, gt, epoch, warmup_epochs):
    #     """
    #     Compute best loss from individual analysis (CE + Dice + clDice + Non-diff Topo)
    #     """
    #     ce_loss = self.ce_loss_fn(pred, gt)
    #     dice_loss = self.dice_loss_fn(pred, gt.unsqueeze(1))
    #     cldice_loss = soft_cldice_loss(pred, gt)

    #     # For topo loss - will return 1 until it gets to after warmup epochs - then it will output a scaling factor for the base loss - in this case CE + Dice + clDice
    #     topo_mult = self.topo_loss_fn(pred, gt, epoch, N=warmup_epochs)
        
    #     # Combining loss - using arbitrary weights
    #     base_loss = ce_loss + dice_loss + 0.5 * cldice_loss
    #     # total_loss = base_loss * topo_mult
    #     total_loss = base_loss * topo_mult
        
    #     loss_dict = {
    #         'ce': ce_loss.item(),
    #         'dice': dice_loss.item(),
    #         'cldice': cldice_loss.item(),
    #         'topo_mult': topo_mult.item() if torch.is_tensor(topo_mult) else topo_mult,
    #         'total': total_loss.item()
    #     }
        
    #     return total_loss, loss_dict

    def compute_loss(self, pred, gt, epoch, lam=1e-3):
        """
        Compute best loss from individual analysis (CE + Dice + clDice + Diff Topo)
        """
        ce_loss = self.ce_loss_fn(pred, gt)
        dice_loss = self.dice_loss_fn(pred, gt.unsqueeze(1))
        cldice_loss = soft_cldice_loss(pred, gt)

        topo_loss, topo_log = self.topo_loss_fn(pred, gt, epoch)
        
        # Combining loss - using arbitrary weights
        base_loss = ce_loss + dice_loss + 0.5 * cldice_loss
        total_loss = base_loss + lam * topo_loss
        
        loss_dict = {
            'ce': ce_loss.item(),
            'dice': dice_loss.item(),
            'cldice': cldice_loss.item(),
            'topo': topo_log,
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict

    def train_epoch(self, epoch):
        """
        Train both models for one epoch.
        """
        self.model_a.train()
        self.model_b.train()
        
        total_loss_a = 0
        total_loss_b = 0
        loss_dicts_a = []
        loss_dicts_b = []
        
        for batch_idx, (X, y, idx) in enumerate(self.train_loader):
            X = X.to(self.device)
            y = y.to(self.device)
            
            self.optimizer_a.zero_grad()
            
            with autocast(device_type='cuda'):
                pred_a = self.model_a(X)
                loss_a, loss_dict_a = self.compute_loss(pred_a, y, epoch)
            
            self.scaler_a.scale(loss_a).backward()
            self.scaler_a.step(self.optimizer_a)
            self.scaler_a.update()
            
            self.optimizer_b.zero_grad()
            
            with autocast(device_type='cuda'):
                pred_b = self.model_b(X)
                loss_b, loss_dict_b = self.compute_loss(pred_b, y, epoch)
            
            self.scaler_b.scale(loss_b).backward()
            self.scaler_b.step(self.optimizer_b)
            self.scaler_b.update()
            
            # Total loss
            total_loss_a += loss_a.item()
            total_loss_b += loss_b.item()
            loss_dicts_a.append(loss_dict_a)
            loss_dicts_b.append(loss_dict_b)
            
            self.global_step += 1
        
        # Average losses   
        num_batches = len(self.train_loader)
        avg_loss_a = total_loss_a / num_batches
        avg_loss_b = total_loss_b / num_batches
        
        # Average loss dicts (Claude)
        avg_dict_a = {k: np.mean([d[k] for d in loss_dicts_a]) for k in loss_dicts_a[0]}
        avg_dict_b = {k: np.mean([d[k] for d in loss_dicts_b]) for k in loss_dicts_b[0]}
        
        return avg_loss_a, avg_loss_b, avg_dict_a, avg_dict_b

    def validate(self):
        """
        This is a Claude fn - will change to custom loss when able

        Validate both models. Returns Dice scores (higher = better).
        
        MODIFY THIS when you have a better metric for incomplete annotations.
        
        Returns:
            dice_a, dice_b: per-class Dice scores dict for each model
            val_loss_a, val_loss_b: validation losses
        """
        self.model_a.eval()
        self.model_b.eval()
        
        # Accumulators for Dice calculation
        intersection_a = {1: 0, 2: 0}  # nerve, vessel
        union_a = {1: 0, 2: 0}
        intersection_b = {1: 0, 2: 0}
        union_b = {1: 0, 2: 0}
        
        total_loss_a = 0
        total_loss_b = 0
        
        with torch.no_grad():
            for X, y, idx in self.test_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                
                with autocast(device_type='cuda'):
                    pred_a = self.model_a(X)
                    pred_b = self.model_b(X)

                pred_a = pred_a.float()
                pred_b = pred_b.float()
                # Simple CE+Dice for validation loss (no topo)
                val_loss_a = self.ce_loss_fn(pred_a, y) + self.dice_loss_fn(pred_a, y.unsqueeze(1))
                val_loss_b = self.ce_loss_fn(pred_b, y) + self.dice_loss_fn(pred_b, y.unsqueeze(1))
                
                total_loss_a += val_loss_a.item()
                total_loss_b += val_loss_b.item()
                
                # Get predictions
                pred_a_class = torch.argmax(pred_a, dim=1)  # [B, D, H, W]
                pred_b_class = torch.argmax(pred_b, dim=1)
                
                # Compute Dice per class
                for c in [1, 2]:
                    pred_a_c = (pred_a_class == c)
                    pred_b_c = (pred_b_class == c)
                    gt_c = (y == c)
                    
                    intersection_a[c] += (pred_a_c & gt_c).sum().item()
                    intersection_b[c] += (pred_b_c & gt_c).sum().item()
                    union_a[c] += pred_a_c.sum().item() + gt_c.sum().item()
                    union_b[c] += pred_b_c.sum().item() + gt_c.sum().item()
        
        # Compute Dice scores
        dice_a = {}
        dice_b = {}
        for c in [1, 2]:
            dice_a[c] = (2 * intersection_a[c] + 1e-5) / (union_a[c] + 1e-5)
            dice_b[c] = (2 * intersection_b[c] + 1e-5) / (union_b[c] + 1e-5)
        
        avg_val_loss_a = total_loss_a / len(self.test_loader)
        avg_val_loss_b = total_loss_b / len(self.test_loader)
        
        return dice_a, dice_b, avg_val_loss_a, avg_val_loss_b

    def save_checkpoint(self, epoch, model_name, is_best=False):
        """Save model checkpoint. (model_name can either be 'a' or 'b')"""
        model = self.model_a if model_name == 'a' else self.model_b
        optimizer = self.optimizer_a if model_name == 'a' else self.optimizer_b
        scheduler = self.scheduler_a if model_name == 'a' else self.scheduler_b
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'pseudo_labels': self.pseudo_labels,
        }
        
        # Save latest
        path = self.save_dir / f"model_{model_name}_latest.pt"
        torch.save(checkpoint, path)
        
        # Save best
        if is_best:
            path = self.save_dir / f"model_{model_name}_best.pt"
            torch.save(checkpoint, path)

    def log_metrics(self, epoch, train_dict_a, train_dict_b, dice_a, dice_b, val_loss_a, val_loss_b, pseudo_stats):
        """Log metrics to TensorBoard and console. (Also claude - will change to work with unspecified num classes later - this is just for test    )"""
        
        # TensorBoard
        self.writer.add_scalar('Train/Loss_A', train_dict_a['total'], epoch)
        self.writer.add_scalar('Train/Loss_B', train_dict_b['total'], epoch)
        self.writer.add_scalar('Val/Loss_A', val_loss_a, epoch)
        self.writer.add_scalar('Val/Loss_B', val_loss_b, epoch)
        self.writer.add_scalar('Val/Dice_Nerve_A', dice_a[1], epoch)
        self.writer.add_scalar('Val/Dice_Nerve_B', dice_b[1], epoch)
        self.writer.add_scalar('Val/Dice_Vessel_A', dice_a[2], epoch)
        self.writer.add_scalar('Val/Dice_Vessel_B', dice_b[2], epoch)
        # Add voxel-level pseudo-label logging
        self.writer.add_scalar('PseudoLabels/Samples', pseudo_stats['samples'], epoch)
        self.writer.add_scalar('PseudoLabels/TotalVoxels', pseudo_stats['total_voxels'], epoch)
        self.writer.add_scalar('PseudoLabels/NerveVoxels', pseudo_stats['nerve_voxels'], epoch)
        self.writer.add_scalar('PseudoLabels/VesselVoxels', pseudo_stats['vessel_voxels'], epoch)
        
        # Console 
        print(f"  Pseudo-labels: {pseudo_stats['samples']} samples, "
            f"{pseudo_stats['total_voxels']} voxels (N: {pseudo_stats['nerve_voxels']}, V: {pseudo_stats['vessel_voxels']})")
        print(f"\nEpoch {epoch}:")
        print(f"  Model A - Train: {train_dict_a['total']:.4f} | Val: {val_loss_a:.4f} | "
              f"Dice N: {dice_a[1]:.4f} V: {dice_a[2]:.4f}")
        print(f"  Model B - Train: {train_dict_b['total']:.4f} | Val: {val_loss_b:.4f} | "
              f"Dice N: {dice_b[1]:.4f} V: {dice_b[2]:.4f}")

    def count_pseudo_labels(self):
        """Count total pseudo-labeled pixels."""
        total = 0
        for idx, pseudo in self.pseudo_labels.items():
            total += pseudo['mask'].sum().item()
        return total

    def train(self, update_frequency=1):
        """
        Main training loop with co-training and pseudo-labels. (structure and basic code provided by claude as I have already done training loops multiple times and wanted something done quick)
        
        Args:
            total_epochs: maximum epochs to train
            warmup_epochs: epochs before topo loss activates
            update_frequency: how often to validate and potentially update pseudo-labels
        """
        self.model_a.to(self.device)
        self.model_b.to(self.device)
        

        
        # Tracking
        stagnation_count = 0
        pseudo_mode = False
        best_combined_dice = 0
        for i in range(len(self.threshold_array)):
            self.thresholds=self.threshold_array[i]
            if i!=0:
                self.epochs=temp
                self.warmup_epochs=0
                self.topo_loss_fn.warmup_epochs=0
                self.scheduler_a = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer_a, T_max=self.epochs
                )
                self.scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer_b, T_max=self.epochs
                )
                stagnation_count=0
            else:
                temp=self.epochs
                self.epochs+=self.warmup_epochs
            for epoch in range(self.epochs):
                # train 1 epoch
                train_loss_a, train_loss_b, train_dict_a, train_dict_b = self.train_epoch(epoch)

                self.scheduler_a.step()
                self.scheduler_b.step()
                
                # validate every N
                if epoch % update_frequency == 0:
                    dice_a, dice_b, val_loss_a, val_loss_b = self.validate()
                    
                    #  Right now verification metric is avg dice - but should change to appropriate later (essentially replace lines 345 to 348 with relevant new metric)
                    current_metric = (dice_a[2] + dice_b[2]) / 2
                    
                    if current_metric > best_combined_dice:
                        best_combined_dice = current_metric
                        stagnation_count = 0
                        
                        # Save best models
                        self.save_checkpoint(epoch, 'a', is_best=True)
                        self.save_checkpoint(epoch, 'b', is_best=True)
                        print(f"  New best! Combined Dice: {current_metric:.4f}")
                    else:
                        stagnation_count += 1
                        print(f"  No improvement. Stagnation: {stagnation_count}/{self.stagnation_patience}")
                    
                    # Pseudo mode?
                    if stagnation_count >= self.stagnation_patience and not pseudo_mode:
                        pseudo_mode = True
                        stagnation_count=0
                        print("\n" + "="*50)
                        print("ENTERING PSEUDO-LABEL MODE")
                        print("="*50 + "\n")
                    
                    # Update pseudo-labels if pseudo mode
                    if pseudo_mode:
                        print("  Updating pseudo-labels...")
                        self.update_all_pseudo_labels()
                    
                    # Logs
                    pseudo_count = len(self.pseudo_labels)
                    self.log_metrics(epoch+i*self.epochs, train_dict_a, train_dict_b, dice_a, dice_b, 
                                val_loss_a, val_loss_b,self.get_pseudo_label_stats())
                    
                    # Save latest models
                    self.save_checkpoint(epoch, 'a', is_best=False)
                    self.save_checkpoint(epoch, 'b', is_best=False)
        
        self.writer.close()
        print(f"\nDone. Best combined Dice: {best_combined_dice:.4f}")
        print(f"Checkpoints saved to: {self.save_dir}")
        
        return best_combined_dice

    def generate_pseudo_labels(self, sample_idx, pred_a, pred_b, gt):

        is_background = (gt == 0)
        
        # Initialize for sample if not in the dict alr
        if sample_idx not in self.pseudo_labels:
            self.pseudo_labels[sample_idx] = {
                'mask': torch.zeros_like(gt, dtype=torch.bool).cpu(),
                'classes': torch.zeros_like(gt, dtype=torch.long).cpu(),
                'confidence': torch.zeros_like(gt, dtype=torch.float).cpu(),
            }
        
        num_foreground_classes = pred_a.shape[0] - 1
        
        for c in range(num_foreground_classes):
            class_idx = c + 1 
            
            # Both models confident about class (above threshold specified in threshold list)
            confident = (pred_a[class_idx] > self.thresholds[c]) & (pred_b[class_idx] > self.thresholds[c])
            
            # Only can be a pseudo label when gt is background
            new_mask = is_background & confident
            new_mask=new_mask.cpu()
            # Avg the softmax for each of these to get avg confidence
            avg_confidence = (pred_a[class_idx] + pred_b[class_idx])/2  # Average of both model's predictions for that class

            self.pseudo_labels[sample_idx]['mask'] |= new_mask
            self.pseudo_labels[sample_idx]['classes'][new_mask] = class_idx
            self.pseudo_labels[sample_idx]['confidence'][new_mask] = avg_confidence[new_mask].cpu()

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

            # For remove I made the threshold it needed to stay above 10% lower - so it doesnt need to pred with 95 pct plus certainty everytime - it just needs to stay above ~85 ish)
            remove = ((pred_a[class_idx] < max(self.thresholds[c] - 0.1, 0.7)) | (pred_b[class_idx] < max(self.thresholds[c] - 0.1, 0.7))) & class_mask.to(self.device)
            remove=remove.cpu()
            self.pseudo_labels[sample_idx]['confidence'][remove]=0
            self.pseudo_labels[sample_idx]['mask'][remove]=False
            self.pseudo_labels[sample_idx]['classes'][remove]=0

    def update_all_pseudo_labels(self):
        self.model_a.eval()
        self.model_b.eval()
        
        with torch.no_grad():
            for X, y, indices in self.train_loader_no_aug:
                X, y = X.to(self.device), y.to(self.device)
                
                pred_a = F.softmax(self.model_a(X), dim=1)
                pred_b = F.softmax(self.model_b(X), dim=1)
                
                # remove
                for c in range(pred_a.shape[1] - 1):
                    class_idx = c + 1
                    remove_thresh = max(self.thresholds[c] - 0.1, 0.7)
                    
                    # batch
                    uncertain = (
                        (pred_a[:, class_idx] < remove_thresh) | 
                        (pred_b[:, class_idx] < remove_thresh)
                    )
                    uncertain_cpu = uncertain.cpu()
                    
                    for i in range(X.shape[0]):
                        idx = indices[i].item()
                        if idx not in self.pseudo_labels:
                            continue

                        class_mask = (
                            self.pseudo_labels[idx]['mask'] & 
                            (self.pseudo_labels[idx]['classes'] == class_idx)
                        )
                        remove = uncertain_cpu[i] & class_mask
                        
                        if remove.any():
                            self.pseudo_labels[idx]['mask'][remove] = False
                            self.pseudo_labels[idx]['classes'][remove] = 0
                            self.pseudo_labels[idx]['confidence'][remove] = 0
                
                is_background = (y == 0)
                
                for c in range(pred_a.shape[1] - 1):
                    class_idx = c + 1
                    
                    confident = (
                        (pred_a[:, class_idx] > self.thresholds[c]) & 
                        (pred_b[:, class_idx] > self.thresholds[c])
                    )
                    new_masks = is_background & confident
                    avg_conf = (pred_a[:, class_idx] + pred_b[:, class_idx]) / 2
                    
                    new_masks_cpu = new_masks.cpu()
                    avg_conf_cpu = avg_conf.cpu()
                    
                    for i in range(X.shape[0]):
                        idx = indices[i].item()
                        
                        if idx not in self.pseudo_labels:
                            self.pseudo_labels[idx] = {
                                'mask': torch.zeros_like(y[0], dtype=torch.bool).cpu(),
                                'classes': torch.zeros_like(y[0], dtype=torch.long).cpu(),
                                'confidence': torch.zeros_like(y[0], dtype=torch.float).cpu(),
                            }
                        
                        sample_mask = new_masks_cpu[i]
                        self.pseudo_labels[idx]['mask'] |= sample_mask
                        self.pseudo_labels[idx]['classes'][sample_mask] = class_idx
                        self.pseudo_labels[idx]['confidence'][sample_mask] = avg_conf_cpu[i][sample_mask]
        
        self.model_a.train()
        self.model_b.train()

    def merge_gt_with_pseudo(self, y, sample_indices):
        y_merged = y.clone()
        
        for i in range(y.shape[0]):
            idx = sample_indices[i].item()
            
            if idx not in self.pseudo_labels:
                continue
            
            pseudo = self.pseudo_labels[idx]
            
            # Move to same device as y for the comparison
            mask_on_device = pseudo['mask'].to(y.device)
            classes_on_device = pseudo['classes'].to(y.device)
            
            apply_mask = mask_on_device & (y[i] == 0)
            y_merged[i][apply_mask] = classes_on_device[apply_mask]
        
        return y_merged

    def get_pseudo_label_stats(self):
        """Get detailed pseudo-label statistics."""
        num_samples = len(self.pseudo_labels)
        
        if num_samples == 0:
            return {'samples': 0, 'total_voxels': 0, 'nerve_voxels': 0, 'vessel_voxels': 0}
        
        total_voxels = 0
        nerve_voxels = 0
        vessel_voxels = 0
        
        for _, pseudo in self.pseudo_labels.items():
            mask = pseudo['mask']
            classes = pseudo['classes']
            
            total_voxels += mask.sum().item()
            nerve_voxels += ((mask) & (classes == 1)).sum().item()
            vessel_voxels += ((mask) & (classes == 2)).sum().item()
        
        return {
            'samples': num_samples,
            'total_voxels': total_voxels,
            'nerve_voxels': nerve_voxels,
            'vessel_voxels': vessel_voxels
        }