# Claude generated this code as I thought it wasnt related to the actual model architecture and thus I did not need to learn it as in depth
# load_data.py

import os
import torch
from torch.utils.data import Dataset, DataLoader
import tifffile
import nibabel as nib
import numpy as np

class MRIDataset(Dataset):
    """
    Loads 3D TIFF and NIfTI volumes and their segmentation masks.
    Combines multiple data sources into one dataset.
    """
    def __init__(self, root_dir, transform=None, augment=False):
        """
        Args:
            root_dir: Parent folder containing all subfolders
            transform: Optional transforms to apply
            augment: Whether to apply data augmentation
        """
        self.root_dir = root_dir
        self.transform = transform
        self.augment = augment
        
        # Augmentation pipeline
        if augment:
            from monai.transforms import (
                Compose,
                RandFlipd,
                RandRotate90d,
                RandGaussianNoised,
                RandAdjustContrastd,
            )
            self.aug_transforms = Compose([
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
                RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
                RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.8, 1.2)),
            ])
        
        # Build list of (input_path, target_path, file_type) tuples
        self.samples = []
        
        # === TIFF DATA (Original 2023 Patients) ===
        
        # Add ipsilateral TIFF samples
        ipsi_input_dir = os.path.join(root_dir, "2023_Ipsilateral Input")
        ipsi_target_dir = os.path.join(root_dir, "2023_Ipsilateral Target")
        self._add_tiff_samples(ipsi_input_dir, ipsi_target_dir)
        
        # Add contralateral TIFF samples
        contra_input_dir = os.path.join(root_dir, "2023_Contralateral Input")
        contra_target_dir = os.path.join(root_dir, "2023_Contralateral Target")
        self._add_tiff_samples(contra_input_dir, contra_target_dir)
        
        # === NIfTI DATA (OpenNeuro) ===
        
        # Add left-sided NIfTI samples
        nifti_input_left = os.path.join(root_dir, "OpenNeuro Cropped MRI L-sided NIfTI")
        nifti_target_left = os.path.join(root_dir, "OpenNeuro Manual Segmentations L-sided NIfTI")
        self._add_nifti_samples(nifti_input_left, nifti_target_left, side='left')
        
        # Add right-sided NIfTI samples
        nifti_input_right = os.path.join(root_dir, "OpenNeuro Cropped MRI R-sided NIfTI")
        nifti_target_right = os.path.join(root_dir, "OpenNeuro Manual Segmentations R-sided NIfTI")
        self._add_nifti_samples(nifti_input_right, nifti_target_right, side='right')
        
        print(f"Loaded {len(self.samples)} total samples")
    
    def _add_tiff_samples(self, input_dir, target_dir):
        """Match TIFF input and target files by MRN number (starting from targets)."""
        if not os.path.exists(input_dir) or not os.path.exists(target_dir):
            print(f"Warning: Directory not found - {input_dir} or {target_dir}")
            return
        
        input_files = sorted(os.listdir(input_dir))
        target_files = sorted(os.listdir(target_dir))
        
        # Start from TARGETS (the limiting factor)
        for target_file in target_files:
            if not target_file.endswith(('.tif', '.tiff')):
                continue
            
            # Extract MRN number (everything before _mask)
            mrn = target_file.replace('_mask.tiff', '').replace('_mask.tif', '')
            
            # Look for matching input file with _cropped suffix
            input_file = f"{mrn}_cropped.tiff"
            input_file_alt = f"{mrn}_cropped.tif"
            
            if input_file in input_files:
                input_path = os.path.join(input_dir, input_file)
                target_path = os.path.join(target_dir, target_file)
                self.samples.append((input_path, target_path, 'tiff'))
            elif input_file_alt in input_files:
                input_path = os.path.join(input_dir, input_file_alt)
                target_path = os.path.join(target_dir, target_file)
                self.samples.append((input_path, target_path, 'tiff'))
    
    def _add_nifti_samples(self, input_dir, target_dir, side):
        """Match NIfTI input and target files (starting from targets)."""
        if not os.path.exists(input_dir) or not os.path.exists(target_dir):
            print(f"Warning: NIfTI directory not found - {input_dir} or {target_dir}")
            return
        
        input_files = sorted(os.listdir(input_dir))
        target_files = sorted(os.listdir(target_dir))
        
        # Start from TARGETS (the limiting factor)
        for target_file in target_files:
            if not target_file.endswith('.nii.gz'):
                continue
            
            # Extract subject number from target: "10_left_manual.nii.gz" -> "10"
            # Format: {number}_{side}_manual.nii.gz
            parts = target_file.replace('.nii.gz', '').split('_')
            if len(parts) >= 3 and parts[-1] == 'manual':
                subject_num = parts[0]
                
                # Look for matching input: "{number}_cropped_{side}.nii.gz"
                input_file = f"{subject_num}_cropped_{side}.nii.gz"
                
                if input_file in input_files:
                    input_path = os.path.join(input_dir, input_file)
                    target_path = os.path.join(target_dir, target_file)
                    self.samples.append((input_path, target_path, 'nifti'))
                else:
                    print(f"Warning: No matching input for {target_file}")
    
    def _load_volume(self, path, file_type):
        """Load a 3D volume from either TIFF or NIfTI format."""
        if file_type == 'tiff':
            return tifffile.imread(path)
        elif file_type == 'nifti':
            nii = nib.load(path)
            return nii.get_fdata()
        else:
            raise ValueError(f"Unknown file type: {file_type}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_path, target_path, file_type = self.samples[idx]
        
        # Load volumes
        x = self._load_volume(input_path, file_type).astype(np.float32)
        y = self._load_volume(target_path, file_type).astype(np.int64)
        
        # Remap label 3 (uncertain) to 0 (background)
        y[y == 3] = 0
        
        # Normalize input to [0, 1]
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        
        # Add channel dimension: (D, H, W) -> (1, D, H, W)
        x = np.expand_dims(x, axis=0)
        
        # Convert to tensors
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        
        # Apply augmentation
        if self.augment:
            data = {"image": x, "label": y.unsqueeze(0).float()}
            data = self.aug_transforms(data)
            x = data["image"]
            y = data["label"].squeeze(0).long()
        
        if self.transform:
            x, y = self.transform(x, y)
        
        return x, y


def get_dataloaders(root_dir, batch_size=2, train_split=0.8, num_workers=0, transform=None):
    """Creates train and test dataloaders."""
    # Create dataset WITHOUT augmentation first (for splitting)
    full_dataset = MRIDataset(root_dir, transform=transform, augment=False)
    
    total_samples = len(full_dataset)
    train_size = int(train_split * total_samples)
    test_size = total_samples - train_size
    
    # Get indices for split
    indices = list(range(total_samples))
    np.random.seed(42)  # Reproducible split
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # Create separate datasets with/without augmentation
    train_dataset = MRIDataset(root_dir, transform=transform, augment=True)
    test_dataset = MRIDataset(root_dir, transform=transform, augment=False)
    
    # Use Subset to apply the split
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
    
    print(f"Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader


def compute_class_weights(root_dir, num_classes=3, dampen=0.5):
    """Compute class weights based on frequency across ALL samples."""
    dataset = MRIDataset(root_dir)
    
    class_counts = torch.zeros(num_classes)
    
    print("Computing class frequencies across all samples...")
    for i in range(len(dataset)):
        _, y = dataset[i]
        for c in range(num_classes):
            class_counts[c] += (y == c).sum().item()
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(dataset)} samples")
    
    total_voxels = class_counts.sum()
    frequencies = class_counts / total_voxels
    
    print(f"\n=== Class Frequencies (All Samples) ===")
    class_names = ['Background', 'Nerve', 'Vessel']
    for c in range(num_classes):
        print(f"  {class_names[c]}: {frequencies[c]*100:.4f}%")
    
    weights = 1.0 / (frequencies + 1e-8)
    weights = weights ** dampen
    weights = weights / weights.sum() * num_classes
    
    print(f"\n=== Class Weights ===")
    print(f"  Tensor: {weights}")
    
    return weights


def inspect_data(root_dir):
    """Check your data before training."""
    dataset = MRIDataset(root_dir)
    
    if len(dataset) == 0:
        print("ERROR: No samples found!")
        return None, None
    
    x, y = dataset[0]
    
    print(f"\n=== Data Inspection ===")
    print(f"Input shape: {x.shape}")
    print(f"Input dtype: {x.dtype}")
    print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"\nTarget shape: {y.shape}")
    print(f"Target dtype: {y.dtype}")
    print(f"Unique labels in target: {torch.unique(y).tolist()}")
    
    return x, y


def check_all_labels(root_dir, num_classes=3):
    """Find any files with unexpected label values."""
    dataset = MRIDataset(root_dir)
    
    print("Checking all target files for invalid labels...")
    print(f"Expected labels: 0 to {num_classes - 1}\n")
    
    problem_files = []
    
    for i in range(len(dataset)):
        input_path, target_path, file_type = dataset.samples[i]
        y = dataset._load_volume(target_path, file_type)
        unique_labels = np.unique(y)
        
        invalid = [l for l in unique_labels if l < 0 or l >= num_classes]
        
        if invalid:
            filename = os.path.basename(target_path)
            print(f"PROBLEM: {filename}")
            print(f"  Found labels: {unique_labels.tolist()}")
            print(f"  Invalid: {invalid}")
            problem_files.append(target_path)
    
    if not problem_files:
        print("All files OK!")
    else:
        print(f"\n{len(problem_files)} files have invalid labels.")
    
    return problem_files

def visualize_slice(x, y, slice_idx=None, pred=None):
    """
    Visualize a single 2D slice from the 3D volume.
    
    Args:
        x: Input tensor (1, D, H, W) or (D, H, W)
        y: Target tensor (D, H, W)
        slice_idx: Which slice to show (default: middle slice)
        pred: Optional prediction tensor (D, H, W) to compare
    """
    import matplotlib.pyplot as plt
    
    # Remove channel dim if present
    if x.dim() == 4:
        x = x.squeeze(0)
    
    # Convert to numpy
    x = x.cpu().numpy()
    y = y.cpu().numpy()
    
    # Default to middle slice
    if slice_idx is None:
        slice_idx = x.shape[0] // 2
    
    # Create figure
    if pred is not None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        pred = pred.cpu().numpy()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    # Input image
    axes[0].imshow(x[slice_idx], cmap='gray')
    axes[0].set_title(f'Input (slice {slice_idx})')
    axes[0].axis('off')
    
    # Ground truth
    axes[1].imshow(y[slice_idx], cmap='tab10', vmin=0, vmax=2)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction (if provided)
    if pred is not None:
        axes[2].imshow(pred[slice_idx], cmap='tab10', vmin=0, vmax=2)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_prediction_3d(x, y_true, y_pred, threshold=0.5):
    """
    Visualize 3D segmentation using napari (since you already have it).
    
    Args:
        x: Input volume (1, D, H, W) or (D, H, W)
        y_true: Ground truth labels (D, H, W)
        y_pred: Model prediction - either logits (3, D, H, W) or class labels (D, H, W)
    """
    import napari
    
    # Handle input shape
    if x.dim() == 4:
        x = x.squeeze(0)
    x = x.cpu().numpy()
    y_true = y_true.cpu().numpy()
    
    # Convert logits to class predictions if needed
    if y_pred.dim() == 4:  # (C, D, H, W) logits
        y_pred = torch.argmax(y_pred, dim=0)
    y_pred = y_pred.cpu().numpy()
    
    # Launch napari viewer
    viewer = napari.Viewer()
    
    # Add layers
    viewer.add_image(x, name='MRI Input', colormap='gray')
    viewer.add_labels(y_true.astype(int), name='Ground Truth')
    viewer.add_labels(y_pred.astype(int), name='Prediction')
    
    napari.run()


def evaluate_and_visualize(model, test_loader, device, num_samples=3):
    """
    Run model on test samples and visualize results.
    Also computes per-class accuracy metrics.
    """
    model.eval()
    
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            if i >= num_samples:
                break
            
            X = X.to(device)
            pred_logits = model(X)  # (B, C, D, H, W)
            pred_classes = torch.argmax(pred_logits, dim=1)  # (B, D, H, W)
            
            # Compute metrics for this batch
            for b in range(X.shape[0]):
                print(f"\n=== Sample {i * test_loader.batch_size + b} ===")
                
                y_true = y[b]
                y_pred = pred_classes[b].cpu()
                
                # Per-class metrics
                for c, name in enumerate(['Background', 'Nerve', 'Vessel']):
                    true_positive = ((y_pred == c) & (y_true == c)).sum().item()
                    predicted = (y_pred == c).sum().item()
                    actual = (y_true == c).sum().item()
                    
                    precision = true_positive / (predicted + 1e-8)
                    recall = true_positive / (actual + 1e-8)
                    
                    print(f"  {name}: Precision={precision:.3f}, Recall={recall:.3f}, "
                          f"Predicted={predicted}, Actual={actual}")
                
                # Visualize in napari
                visualize_prediction_3d(X[b], y_true, pred_logits[b])


# Run this to test your data loading
if __name__ == "__main__":
    ROOT_DIR = r"C:\Users\gogoi\Desktop\ml-test\2023 Patients"
    
    check_all_labels(ROOT_DIR)
    x, y = inspect_data(ROOT_DIR)
    
    train_loader, test_loader = get_dataloaders(ROOT_DIR, batch_size=2)
    
    for X_batch, y_batch in train_loader:
        print(f"\nBatch shapes: X={X_batch.shape}, y={y_batch.shape}")
        break