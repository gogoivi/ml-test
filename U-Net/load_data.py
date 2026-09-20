# Claude generated this code as I thought it wasnt related to the actual model architecture and thus I did not need to learn it as in depth
# load_data.py

import os
import torch
from torch.utils.data import Dataset, DataLoader
import tifffile
import numpy as np

class MRIDataset(Dataset):
    """
    Loads 3D TIFF volumes and their segmentation masks.
    Combines ipsilateral and contralateral data into one dataset.
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Parent folder containing the 4 subfolders
            transform: Optional transforms to apply (for data augmentation later)
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Build list of (input_path, target_path) pairs
        self.samples = []
        
        # Add ipsilateral samples
        ipsi_input_dir = os.path.join(root_dir, "2023_Ipsilateral Input")
        ipsi_target_dir = os.path.join(root_dir, "2023_Ipsilateral Target")
        self._add_samples(ipsi_input_dir, ipsi_target_dir)
        
        # Add contralateral samples
        contra_input_dir = os.path.join(root_dir, "2023_Contralateral Input")
        contra_target_dir = os.path.join(root_dir, "2023_Contralateral Target")
        self._add_samples(contra_input_dir, contra_target_dir)
        
        print(f"Loaded {len(self.samples)} total samples")
    
    def _add_samples(self, input_dir, target_dir):
        """Match input and target files by MRN number."""
        if not os.path.exists(input_dir) or not os.path.exists(target_dir):
            print(f"Warning: Directory not found - {input_dir} or {target_dir}")
            return
        
        input_files = sorted(os.listdir(input_dir))
        target_files = sorted(os.listdir(target_dir))
        
        for input_file in input_files:
            if not input_file.endswith(('.tif', '.tiff')):
                continue
            
            # Extract MRN number (everything before _cropped)
            mrn = input_file.replace('_cropped.tiff', '').replace('_cropped.tif', '')
            
            # Look for matching target file with _mask suffix
            target_file = f"{mrn}_mask.tiff"
            target_file_alt = f"{mrn}_mask.tif"  # In case extension differs
            
            if target_file in target_files:
                input_path = os.path.join(input_dir, input_file)
                target_path = os.path.join(target_dir, target_file)
                self.samples.append((input_path, target_path))
            elif target_file_alt in target_files:
                input_path = os.path.join(input_dir, input_file)
                target_path = os.path.join(target_dir, target_file_alt)
                self.samples.append((input_path, target_path))
            # else:
            #     print(f"Warning: No matching target for {input_file}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_path, target_path = self.samples[idx]
        
        # Load TIFF files as numpy arrays
        # tifffile reads as shape (D, H, W) or (H, W) for 2D
        x = tifffile.imread(input_path).astype(np.float32)
        y = tifffile.imread(target_path).astype(np.int64)  # Labels must be long/int64
        
        # Normalize input to [0, 1] range (adjust if your data is different)
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        
        # Add channel dimension: (D, H, W) -> (1, D, H, W)
        x = np.expand_dims(x, axis=0)
        
        # Convert to tensors
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        
        # Apply transforms if any (for data augmentation)
        if self.transform:
            x, y = self.transform(x, y)
        
        return x, y


def get_dataloaders(root_dir, batch_size=2, train_split=0.8, num_workers=0,transform=None):
    """
    Creates train and test dataloaders.
    
    Args:
        root_dir: Parent folder containing the 4 subfolders
        batch_size: Batch size (keep small for 3D volumes - memory intensive!)
        train_split: Fraction of data for training
        num_workers: Parallel data loading workers (0 for Windows compatibility)
    
    Returns:
        train_loader, test_loader
    """
    # Create full dataset
    full_dataset = MRIDataset(root_dir,transform=transform)
    
    # Split into train/test
    total_samples = len(full_dataset)
    train_size = int(train_split * total_samples)
    test_size = total_samples - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size]
    )
    
    print(f"Train samples: {train_size}, Test samples: {test_size}")
    
    # Create dataloaders
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


def inspect_data(root_dir):
    """
    Utility function to check your data before training.
    Run this first to verify everything loads correctly!
    """
    dataset = MRIDataset(root_dir)
    
    if len(dataset) == 0:
        print("ERROR: No samples found!")
        return
    
    # Load first sample
    x, y = dataset[0]
    
    print(f"\n=== Data Inspection ===")
    print(f"Input shape: {x.shape}")
    print(f"Input dtype: {x.dtype}")
    print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"\nTarget shape: {y.shape}")
    print(f"Target dtype: {y.dtype}")
    print(f"Unique labels in target: {torch.unique(y).tolist()}")
    
    # Count class frequencies (useful for setting weights)
    total_voxels = y.numel()
    print(f"\nClass frequencies:")
    for label in torch.unique(y):
        count = (y == label).sum().item()
        percentage = 100 * count / total_voxels
        print(f"  Class {label}: {percentage:.2f}%")
    
    return x, y

def compute_class_weights(root_dir, num_classes=3,dampen=0.5):
    """
    Compute class weights based on frequency across ALL samples.
    Returns weights inversely proportional to class frequency.
    """
    dataset = MRIDataset(root_dir)
    
    # Count total voxels per class
    class_counts = torch.zeros(num_classes)
    
    print("Computing class frequencies across all samples...")
    for i in range(len(dataset)):
        _, y = dataset[i]
        for c in range(num_classes):
            class_counts[c] += (y == c).sum().item()
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(dataset)} samples")
    
    # Compute frequencies
    total_voxels = class_counts.sum()
    frequencies = class_counts / total_voxels
    
    print(f"\n=== Class Frequencies (All Samples) ===")
    class_names = ['Background', 'Nerve', 'Vessel']
    for c in range(num_classes):
        print(f"  {class_names[c]}: {frequencies[c]*100:.4f}%")
    
    # Use dampened version:
    weights = 1.0 / (frequencies + 1e-8)
    weights = weights ** dampen  # <-- Add this line
    
    # Normalize
    weights = weights / weights.sum() * num_classes
    
    return weights

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

def check_all_labels(root_dir):
    """Find any target files with unexpected label values."""
    dataset = MRIDataset(root_dir)
    
    print("Checking all target files for invalid labels...")
    print("Expected labels: 0, 1, 2\n")
    
    problem_files = []
    
    for i in range(len(dataset)):
        input_path, target_path = dataset.samples[i]
        y = tifffile.imread(target_path)
        unique_labels = np.unique(y)
        
        # Check for any label outside [0, 1, 2]
        invalid = [l for l in unique_labels if l < 0 or l > 2]
        
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
    # Change this to your actual data path
    ROOT_DIR = r"C:\Users\gogoi\Desktop\ml-test\2023 Patients"

    check_all_labels(ROOT_DIR)
    
    # First, inspect the data
    x, y = inspect_data(ROOT_DIR)
    
    # Visualize a slice
    visualize_slice(x, y)
    
    # Test dataloader creation
    train_loader, test_loader = get_dataloaders(ROOT_DIR, batch_size=2)
    
    # Test loading a batch
    for X_batch, y_batch in train_loader:
        print(f"\nBatch shapes: X={X_batch.shape}, y={y_batch.shape}")
        break