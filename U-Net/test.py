import torch
from modified_cldice_3d import soft_cldice_loss
# Test shapes
batch_size = 2
num_classes = 3
D, H, W = 48, 48, 48

pred = torch.randn(batch_size, num_classes, D, H, W)
target = torch.randint(0, num_classes, (batch_size, D, H, W))

loss = soft_cldice_loss(pred, target, class_weights=[1.0, 1.0])
print(f"Loss: {loss.item():.4f}")
print(f"Loss shape: {loss.shape}")  # Should be scalar  