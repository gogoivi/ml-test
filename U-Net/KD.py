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
from loss_fn import CC_Loss,Cubical_Complex_Loss_Differentiable
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

class KD:
    def __init__(self,model_a,model_b,path_model_a,path_model_b,root_dir:str,size=48,epochs=120,
                 batch_size=20,kernel_size=3,alpha:float=0.5,warmup_epochs:int=15,lr=0.0005,thresholds:list=[0.8,0.8],lam=1e-4,
                 stagnation_patience:int=5,save_dir="checkpoints",experiment_name="Knowledge Distill"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.student=U_Net(size=size,classes=3,color=False,kernel_size=kernel_size)
        self.teacher_a=model_a
        self.teacher_b=model_b
        self.alpha=alpha
        self.lam=lam
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
        self.teacher_a.to(self.device)
        self.teacher_b.to(self.device)
        self.student.to(self.device)
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((size,size)),
            torchvision.transforms.ToTensor()
        ])

        self.weights = compute_class_weights(root_dir, num_classes=3)
        self.weights=self.weights.to(self.device)

        # Create dataloaders
        self.train_dataloader, self.test_dataloader = get_dataloaders(root_dir, batch_size=batch_size)

        self.optimizer=torch.optim.Adamax(params=self.student.parameters(),lr=lr,weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        self.scaler=GradScaler()

        self.CE_Loss=nn.CrossEntropyLoss(weight=self.weights)

        self.dice_loss_3d = DiceLoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            smooth_nr=1e-5,
            smooth_dr=1e-5
        )
        self.topo_loss = Cubical_Complex_Loss_Differentiable(warmup_epochs=warmup_epochs,lam=self.lam)

        # Logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(save_dir) / f"{experiment_name}_{timestamp}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.save_dir / "tensorboard")

        # Tracking
        self.best_metric = float('inf')
        self.global_step = 0

    def train_loss(self,x,gt,pred_a,pred_b,epoch,T:int=1):
        """Can change to whatever loss is appropriate - right now am doing topo+CE (or KLDiv) +clDice+Dice"""
        soft_y,hard_y=merge(gt,F.softmax(pred_a/T,dim=1),F.softmax(pred_b/T,dim=1),thresh=self.thresholds)
        studentlog_prob = F.log_softmax(x / T, dim=1)   # log-probabilities

        topo_hard, topo_hard_log = self.topo_loss(x, gt, epoch)
        cl_hard = soft_cldice_loss(x, gt)
        dice_hard = self.dice_loss_3d(x, gt.unsqueeze(1))
        CE = self.CE_Loss(x, gt)
        hard_loss = self.alpha * (topo_hard + cl_hard + dice_hard + CE)
        # Changed kl loss from batch mean to mean becasue it was eclising everything else - even though technically batchmean is correct
        kl_loss = F.kl_div(studentlog_prob, soft_y, reduction='none').mean() * (T ** 2)
        topo_soft,topo_soft_log = self.topo_loss(x, hard_y, epoch)
        cl_soft = soft_cldice_loss(x, soft_y)
        dice_soft = self.dice_loss_3d(x, hard_y.unsqueeze(1))
        soft_loss = (1 - self.alpha) * (kl_loss + topo_soft + cl_soft + dice_soft)
        total_loss = hard_loss + soft_loss

        loss_dict = {
            'hard_loss': hard_loss.item(),
            'soft_loss': soft_loss.item(),
            'dice_hard': dice_hard.item(),
            'dice_soft': dice_soft.item(),
            'cldice_hard': cl_hard.item(),
            'cldice_soft': cl_soft.item(),
            'kl_loss': kl_loss.item(),
            'topo_hard': topo_hard_log.item() if torch.is_tensor(topo_hard_log) else topo_hard_log,
            'topo_soft': topo_soft_log.item() if torch.is_tensor(topo_soft_log) else topo_soft_log,
            'total': total_loss.item()
        }

        return total_loss, loss_dict

    def validation(self,T:int=1):
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
            for X, y in self.test_dataloader:
                X = X.to(self.device)
                y = y.to(self.device)
                
                with autocast(device_type='cuda'):
                    pred = self.student(X)
                    y_a=self.teacher_a(X)
                    y_b=self.teacher_b(X)

                _, hard_y = merge(y, F.softmax(y_a / T, dim=1), F.softmax(y_b / T, dim=1),self.thresholds)
                pred = pred.float()

                # Simple CE+Dice for validation loss (no topo) but uses merged gt
                val_loss = self.CE_Loss(pred, hard_y) + self.dice_loss_3d(pred, hard_y.unsqueeze(1))
                
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
        
        avg_val_loss = total_loss / len(self.test_dataloader)
        
        return dice, avg_val_loss

    def train_epoch(self,epoch):
        self.student.train()
        
        total_loss = 0
        loss_dicts = []
        
        for batch_idx, (X, y) in enumerate(self.train_dataloader):
            X = X.to(self.device)
            y = y.to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast(device_type='cuda'):
                pred_a = self.teacher_a(X)
                pred_b=self.teacher_b(X)
                pred_student=self.student(X)
                loss, loss_dict = self.train_loss(pred_student, y, pred_a,pred_b,epoch)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Total loss
            total_loss += loss.item()
            loss_dicts.append(loss_dict)
            
            self.global_step += 1
        
        # Average losses   
        num_batches = len(self.train_dataloader)
        avg_loss = total_loss / num_batches
        
        # Average loss dicts (Claude)
        avg_dict = {k: np.mean([d[k] for d in loss_dicts]) for k in loss_dicts[0]}
        
        return avg_loss, avg_dict
    
    def log_metrics(self, epoch, train_dict, dice, val_loss):
        """Log metrics to TensorBoard and console."""
        
        # TensorBoard
        self.writer.add_scalar('Train/Loss_A', train_dict['total'], epoch)
        self.writer.add_scalar('Val/Loss_A', val_loss, epoch)
        self.writer.add_scalar('Val/Dice_Nerve_A', dice[1], epoch)
        self.writer.add_scalar('Val/Dice_Vessel_A', dice[2], epoch)
        
        # Console 
        print(f"\nEpoch {epoch}:")
        print(f"  Model - Train: {train_dict['total']:.4f} | Val: {val_loss:.4f} | "
              f"Dice N: {dice[1]:.4f} V: {dice[2]:.4f}")

    def train(self, total_epochs=100, warmup_epochs=10, update_frequency=1):
        """
        Main training loop with co-training and pseudo-labels. (structure and basic code provided by claude as I have already done training loops multiple times and wanted something done quick)
        
        Args:
            total_epochs: maximum epochs to train
            warmup_epochs: epochs before topo loss activates
            update_frequency: how often to validate and potentially update pseudo-labels
        """
        
        # Tracking
        stagnation_count = 0
        best_combined_dice = 0
        
        for epoch in range(total_epochs):
            # train 1 epoch
            train_loss, train_dict = self.train_epoch(epoch)

            # validate every N
            if epoch % update_frequency == 0:
                dice, val_loss = self.validation()
                self.scheduler.step(val_loss)
                combined_dice = (dice[1] + dice[2]) / 2
                if combined_dice > best_combined_dice:
                    best_combined_dice = combined_dice
                    stagnation_count = 0
                    
                    # Save best models
                    self.save_checkpoint(epoch, f'Student', is_best=True)
                    print(f"  New best! Combined Dice: {combined_dice:.4f}")
                else:
                    stagnation_count += 1
                    print(f"  No improvement. Stagnation: {stagnation_count}/{self.stagnation_patience}")
                
                self.log_metrics(epoch, train_dict,dice, val_loss)
                
                # Save latest models
                self.save_checkpoint(epoch, f'Student', is_best=False)

                if stagnation_count>self.stagnation_patience:
                    self.writer.close()
                    print(f"\nDone. Best combined Dice: {best_combined_dice:.4f}")
                    print(f"Checkpoints saved to: {self.save_dir}")
                    break
        
        self.writer.close()
        print(f"\nDone. Best combined Dice: {best_combined_dice:.4f}")
        print(f"Checkpoints saved to: {self.save_dir}")
        
        return best_combined_dice

    def save_checkpoint(self, epoch, model_name, is_best=False):
        """Save model checkpoint. (model_name can either be 'a' or 'b')"""
        model = self.student
        optimizer = self.optimizer
        scheduler = self.scheduler
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        
        # Save latest
        path = self.save_dir / f"model_{model_name}_latest.pt"
        torch.save(checkpoint, path)
        
        # Save best
        if is_best:
            path = self.save_dir / f"model_{model_name}_best.pt"
            torch.save(checkpoint, path)

        

# def test_cc_loss_gradients():
#     """
#     Claude FN
#     Tests that CC_Loss produces non-zero, finite gradients.
#     Run this before training to verify the loss is differentiable.
#     """
#     from loss_fn import CC_Loss

#     print("=" * 60)
#     print("Testing CC_Loss gradient flow")
#     print("=" * 60)

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     loss_fn = CC_Loss(warmup_epochs=0)  # warmup=0 so topo is active immediately

#     # Create fake data with requires_grad
#     pred = torch.randn(1, 3, 16, 16, 16, device=device, requires_grad=True)
#     gt = torch.randint(0, 3, (1, 16, 16, 16), device=device)

#     # --- Test 1: Does it compute at all? ---
#     print("\n[Test 1] Forward pass...")
#     try:
#         loss = loss_fn(pred, gt, epoch=5)
#         print(f"  PASS — Loss value: {loss.item():.6f}")
#     except Exception as e:
#         print(f"  FAIL — {e}")
#         return False

#     # --- Test 2: Does backward() work? ---
#     print("\n[Test 2] Backward pass...")
#     try:
#         loss.backward()
#         print(f"  PASS — backward() completed")
#     except Exception as e:
#         print(f"  FAIL — {e}")
#         return False

#     # --- Test 3: Are gradients non-None? ---
#     print("\n[Test 3] Gradients exist...")
#     if pred.grad is None:
#         print("  FAIL — pred.grad is None (no gradient computed)")
#         return False
#     else:
#         print(f"  PASS — grad shape: {pred.grad.shape}")

#     # --- Test 4: Are gradients non-zero? ---
#     print("\n[Test 4] Gradients non-zero...")
#     grad_norm = pred.grad.norm().item()
#     if grad_norm == 0:
#         print("  FAIL — gradient norm is exactly 0")
#         return False
#     else:
#         print(f"  PASS — gradient norm: {grad_norm:.6f}")

#     # --- Test 5: Are gradients finite (no NaN/Inf)? ---
#     print("\n[Test 5] Gradients finite...")
#     if torch.isnan(pred.grad).any():
#         nan_count = torch.isnan(pred.grad).sum().item()
#         print(f"  FAIL — {nan_count} NaN values in gradients")
#         return False
#     if torch.isinf(pred.grad).any():
#         inf_count = torch.isinf(pred.grad).sum().item()
#         print(f"  FAIL — {inf_count} Inf values in gradients")
#         return False
#     print(f"  PASS — all gradients finite")

#     # --- Test 6: Does it work through a model? ---
#     print("\n[Test 6] Full model gradient flow...")
#     from model import U_Net
#     model = U_Net(size=16, classes=3, color=False, kernel_size=3).to(device)
#     x_input = torch.randn(1, 1, 16, 16, 16, device=device)
#     gt_input = torch.randint(0, 3, (1, 16, 16, 16), device=device)

#     model_pred = model(x_input)
#     model_loss = loss_fn(model_pred, gt_input, epoch=5)
#     model_loss.backward()

#     # Check that at least some model params got gradients
#     params_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.norm() > 0)
#     total_params = sum(1 for p in model.parameters())
#     print(f"  {params_with_grad}/{total_params} parameters received non-zero gradients")

#     if params_with_grad == 0:
#         print("  FAIL — no model parameters got gradients")
#         return False
#     print("  PASS")

#     # --- Test 7: Numerical gradient check (finite differences) ---
#     print("\n[Test 7] Numerical gradient check (finite differences)...")
#     eps = 1e-3
#     pred2 = torch.randn(1, 3, 8, 8, 8, device=device, requires_grad=True)
#     gt2 = torch.randint(0, 3, (1, 8, 8, 8), device=device)

#     # Analytical gradient
#     loss2 = loss_fn(pred2, gt2, epoch=5)
#     loss2.backward()
#     analytical_grad = pred2.grad.clone()

#     # Numerical gradient for a few random elements
#     mismatches = 0
#     checks = 10
#     for _ in range(checks):
#         idx = tuple(torch.randint(0, s, (1,)).item() for s in pred2.shape)

#         pred_plus = pred2.detach().clone()
#         pred_plus[idx] += eps
#         pred_minus = pred2.detach().clone()
#         pred_minus[idx] -= eps

#         loss_plus = loss_fn(pred_plus, gt2, epoch=5)
#         loss_minus = loss_fn(pred_minus, gt2, epoch=5)

#         numerical = (loss_plus.item() - loss_minus.item()) / (2 * eps)
#         analytical = analytical_grad[idx].item()

#         if abs(analytical) > 1e-5 or abs(numerical) > 1e-5:
#             rel_error = abs(analytical - numerical) / (max(abs(analytical), abs(numerical)) + 1e-8)
#             if rel_error > 0.1:  # 10% tolerance
#                 mismatches += 1

#     if mismatches > checks // 2:
#         print(f"  WARNING — {mismatches}/{checks} elements had >10% relative error")
#         print(f"  (This may be OK for topology-based losses with discrete operations)")
#     else:
#         print(f"  PASS — {checks - mismatches}/{checks} elements matched within tolerance")

#     print("\n" + "=" * 60)
#     print("ALL TESTS PASSED" if params_with_grad > 0 else "SOME TESTS FAILED")
#     print("=" * 60)
#     return True


# if __name__ == "__main__":
#     test_cc_loss_gradients()
