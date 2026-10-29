# load_data.py

import os
import json
import shutil
import torch
from torch.utils.data import Dataset, DataLoader
import tifffile
import nibabel as nib
import numpy as np


def is_nnunet_format(root_dir, dataset_name="Dataset001_MRI"):
    """Check if the data is already in nnU-Net format."""
    dataset_dir = os.path.join(root_dir, dataset_name)
    required_dirs = ["imagesTr", "labelsTr"]
    required_file = "dataset.json"
    
    if not os.path.exists(dataset_dir):
        return False
    
    for d in required_dirs:
        if not os.path.exists(os.path.join(dataset_dir, d)):
            return False
    
    if not os.path.exists(os.path.join(dataset_dir, required_file)):
        return False
    
    # Check that there are actually files in the directories
    images_dir = os.path.join(dataset_dir, "imagesTr")
    labels_dir = os.path.join(dataset_dir, "labelsTr")
    
    if len(os.listdir(images_dir)) == 0 or len(os.listdir(labels_dir)) == 0:
        return False
    
    return True


def convert_to_nnunet_format(root_dir, dataset_name="Dataset001_MRI", dataset_id=1):
    """
    Convert existing folder structure to nnU-Net format.
    
    Original structure:
    - 2023_Ipsilateral Input/ -> {mrn}_cropped.tiff
    - 2023_Ipsilateral Target/ -> {mrn}_mask.tiff
    - 2023_Contralateral Input/ -> {mrn}_cropped.tiff
    - 2023_Contralateral Target/ -> {mrn}_mask.tiff
    - OpenNeuro Cropped MRI L-sided NIfTI/ -> {num}_cropped_left.nii.gz
    - OpenNeuro Manual Segmentations L-sided NIfTI/ -> {num}_left_manual.nii.gz
    - OpenNeuro Cropped MRI R-sided NIfTI/ -> {num}_cropped_right.nii.gz
    - OpenNeuro Manual Segmentations R-sided NIfTI/ -> {num}_right_manual.nii.gz
    
    nnU-Net structure:
    - Dataset001_MRI/
      - imagesTr/
        - MRI_0000_0000.nii.gz
      - labelsTr/
        - MRI_0000.nii.gz
      - dataset.json
    """
    dataset_dir = os.path.join(root_dir, dataset_name)
    images_dir = os.path.join(dataset_dir, "imagesTr")
    labels_dir = os.path.join(dataset_dir, "labelsTr")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    case_id = 0
    training_cases = []
    
    # Helper function to save volume as NIfTI
    def save_as_nifti(volume, output_path):
        nii = nib.Nifti1Image(volume, affine=np.eye(4))
        nib.save(nii, output_path)
    
    # Helper function to clean labels (remap 3 -> 0)
    def clean_labels(label_vol):
        """Remap label 3 (uncertain) to 0 (background)."""
        label_vol = label_vol.copy()
        label_vol[label_vol == 3] = 0
        return label_vol
    
    # Process TIFF data
    tiff_pairs = [
        ("2023_Ipsilateral Input", "2023_Ipsilateral Target"),
        ("2023_Contralateral Input", "2023_Contralateral Target"),
    ]
    
    for input_folder, target_folder in tiff_pairs:
        input_dir = os.path.join(root_dir, input_folder)
        target_dir = os.path.join(root_dir, target_folder)
        
        if not os.path.exists(input_dir) or not os.path.exists(target_dir):
            print(f"Skipping {input_folder} - directory not found")
            continue
        
        input_files = sorted(os.listdir(input_dir))
        target_files = sorted(os.listdir(target_dir))
        
        for target_file in target_files:
            if not target_file.endswith(('.tif', '.tiff')):
                continue
            
            mrn = target_file.replace('_mask.tiff', '').replace('_mask.tif', '')
            input_file = f"{mrn}_cropped.tiff"
            input_file_alt = f"{mrn}_cropped.tif"
            
            input_path = None
            if input_file in input_files:
                input_path = os.path.join(input_dir, input_file)
            elif input_file_alt in input_files:
                input_path = os.path.join(input_dir, input_file_alt)
            
            if input_path:
                target_path = os.path.join(target_dir, target_file)
                
                # Load volumes
                input_vol = tifffile.imread(input_path).astype(np.float32)
                target_vol = tifffile.imread(target_path).astype(np.int16)
                
                # Clean labels: remap 3 -> 0
                target_vol = clean_labels(target_vol)
                
                # Save as NIfTI with nnU-Net naming
                case_name = f"MRI_{case_id:04d}"
                save_as_nifti(input_vol, os.path.join(images_dir, f"{case_name}_0000.nii.gz"))
                save_as_nifti(target_vol, os.path.join(labels_dir, f"{case_name}.nii.gz"))
                
                training_cases.append(case_name)
                case_id += 1
                print(f"Converted: {mrn} -> {case_name}")
    
    # Process NIfTI data
    nifti_pairs = [
        ("OpenNeuro Cropped MRI L-sided NIfTI", "OpenNeuro Manual Segmentations L-sided NIfTI", "left"),
        ("OpenNeuro Cropped MRI R-sided NIfTI", "OpenNeuro Manual Segmentations R-sided NIfTI", "right"),
    ]
    
    for input_folder, target_folder, side in nifti_pairs:
        input_dir = os.path.join(root_dir, input_folder)
        target_dir = os.path.join(root_dir, target_folder)
        
        if not os.path.exists(input_dir) or not os.path.exists(target_dir):
            print(f"Skipping {input_folder} - directory not found")
            continue
        
        input_files = sorted(os.listdir(input_dir))
        target_files = sorted(os.listdir(target_dir))
        
        for target_file in target_files:
            if not target_file.endswith('.nii.gz'):
                continue
            
            parts = target_file.replace('.nii.gz', '').split('_')
            if len(parts) >= 3 and parts[-1] == 'manual':
                subject_num = parts[0]
                input_file = f"{subject_num}_cropped_{side}.nii.gz"
                
                if input_file in input_files:
                    input_path = os.path.join(input_dir, input_file)
                    target_path = os.path.join(target_dir, target_file)
                    
                    # Load NIfTI, clean labels, and save
                    # (instead of just copying)
                    input_nii = nib.load(input_path)
                    target_nii = nib.load(target_path)
                    
                    input_vol = input_nii.get_fdata().astype(np.float32)
                    target_vol = target_nii.get_fdata().astype(np.int16)
                    
                    # Clean labels: remap 3 -> 0
                    target_vol = clean_labels(target_vol)
                    
                    case_name = f"MRI_{case_id:04d}"
                    save_as_nifti(input_vol, os.path.join(images_dir, f"{case_name}_0000.nii.gz"))
                    save_as_nifti(target_vol, os.path.join(labels_dir, f"{case_name}.nii.gz"))
                    
                    training_cases.append(case_name)
                    case_id += 1
                    print(f"Converted: {subject_num}_{side} -> {case_name}")
    
    # Create dataset.json
    dataset_json = {
        "channel_names": {
            "0": "MRI"
        },
        "labels": {
            "background": 0,
            "nerve": 1,
            "vessel": 2
        },
        "numTraining": len(training_cases),
        "file_ending": ".nii.gz",
        "name": dataset_name,
        "dataset_id": dataset_id,
        "reference": "",
        "licence": "",
        "description": "MRI Nerve/Vessel Segmentation Dataset",
        "training": [
            {
                "image": f"./imagesTr/{case}_0000.nii.gz",
                "label": f"./labelsTr/{case}.nii.gz"
            } for case in training_cases
        ]
    }
    
    with open(os.path.join(dataset_dir, "dataset.json"), 'w') as f:
        json.dump(dataset_json, f, indent=4)
    
    print(f"\nConverted {len(training_cases)} cases to nnU-Net format in {dataset_dir}")
    print(f"Label 3 (uncertain) has been remapped to 0 (background) in all files.")
    return dataset_dir


def ensure_nnunet_format(root_dir, dataset_name="Dataset001_MRI"):
    """Check if data is in nnU-Net format, convert if not. Returns the dataset directory."""
    if is_nnunet_format(root_dir, dataset_name):
        print(f"Data already in nnU-Net format at {os.path.join(root_dir, dataset_name)}")
        return os.path.join(root_dir, dataset_name)
    else:
        print("Converting data to nnU-Net format...")
        return convert_to_nnunet_format(root_dir, dataset_name)


class MRIDataset(Dataset):
    def __init__(self, root_dir, transform=None, augment=False, file_type='all', dataset_name="Dataset001_MRI",pseudo_labels=None):
        """
        Args:
            root_dir: Parent folder containing all subfolders
            transform: Optional transforms to apply
            augment: Whether to apply data augmentation
            file_type: 'all', 'tiff', or 'nifti' - which files to load (only used for original format)
            dataset_name: Name of the nnU-Net dataset folder
        """
        self.root_dir = root_dir
        self.transform = transform
        self.augment = augment
        self.file_type = file_type
        self.dataset_name = dataset_name
        self.pseudo_labels = pseudo_labels
        
        
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
        
        self.samples = []
        
        # Ensure data is in nnU-Net format (convert if necessary)
        self.dataset_dir = ensure_nnunet_format(root_dir, dataset_name)
        
        # Load from nnU-Net format
        self._load_nnunet_samples()
        
        print(f"Loaded {len(self.samples)} total samples")
    
    def _load_nnunet_samples(self):
        """Load samples from nnU-Net folder structure."""
        images_dir = os.path.join(self.dataset_dir, "imagesTr")
        labels_dir = os.path.join(self.dataset_dir, "labelsTr")
        
        # Read dataset.json for the list of training cases
        dataset_json_path = os.path.join(self.dataset_dir, "dataset.json")
        
        if os.path.exists(dataset_json_path):
            with open(dataset_json_path, 'r') as f:
                dataset_info = json.load(f)
            
            for case in dataset_info.get("training", []):
                # Paths in dataset.json are relative, convert to absolute
                image_path = os.path.join(self.dataset_dir, case["image"].replace("./", ""))
                label_path = os.path.join(self.dataset_dir, case["label"].replace("./", ""))
                
                if os.path.exists(image_path) and os.path.exists(label_path):
                    self.samples.append((image_path, label_path, 'nifti'))
                else:
                    print(f"Warning: Missing files for case - {image_path} or {label_path}")
        else:
            # Fallback: scan directories directly
            label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.nii.gz')])
            
            for label_file in label_files:
                # Label: MRI_0000.nii.gz -> Image: MRI_0000_0000.nii.gz
                case_name = label_file.replace('.nii.gz', '')
                image_file = f"{case_name}_0000.nii.gz"
                
                image_path = os.path.join(images_dir, image_file)
                label_path = os.path.join(labels_dir, label_file)
                
                if os.path.exists(image_path):
                    self.samples.append((image_path, label_path, 'nifti'))
                else:
                    print(f"Warning: No matching image for {label_file}")
    
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

        if self.pseudo_labels is not None:
            # Merge pseudo-labels (before augmentation)
            if idx in self.pseudo_labels:        
                pseudo = self.pseudo_labels[idx]
                
                # Where we have pseudo-labels AND gt is background
                apply_mask = pseudo['mask'] & (y == 0)
                
                # Override labels
                y[apply_mask] = pseudo['classes'][apply_mask]
    
        # Apply augmentation
        if self.augment:
            data = {"image": x, "label": y.unsqueeze(0).float()}
            data = self.aug_transforms(data)
            x = data["image"]
            y = data["label"].squeeze(0).long()
        
        if self.transform:
            x, y = self.transform(x, y)

        if self.pseudo_labels is not None:
            return x, y,idx
        else:
            return x,y


def get_dataloaders(root_dir, batch_size=20, train_split=0.8, num_workers=0, transform=None, file_type='all', dataset_name="Dataset001_MRI",pseudo_labels=None):
    """Creates train and test dataloaders."""
    full_dataset = MRIDataset(root_dir, transform=transform, augment=False, file_type=file_type, dataset_name=dataset_name,pseudo_labels=pseudo_labels)
    
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
    train_dataset = MRIDataset(root_dir=root_dir, augment=True, pseudo_labels=pseudo_labels)
    train_no_aug_dataset=MRIDataset(root_dir=root_dir,augment=False,pseudo_labels=pseudo_labels)
    test_dataset = MRIDataset(root_dir=root_dir, augment=False, pseudo_labels=pseudo_labels)
    
    # Use Subset to apply the split
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    train_no_aug_dataset=torch.utils.data.Subset(train_no_aug_dataset,train_indices)
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
    
    print(f"Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    train_no_aug_loader = DataLoader(
        train_no_aug_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    if pseudo_labels is not None:
        return train_loader, train_no_aug_loader,test_loader
    else:
        return train_loader,test_loader


def compute_class_weights(root_dir, num_classes=3, dampen=0.5, dataset_name="Dataset001_MRI"):
    """Compute class weights based on frequency across ALL samples."""
    dataset = MRIDataset(root_dir, dataset_name=dataset_name)
    
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


def inspect_data(root_dir, dataset_name="Dataset001_MRI"):
    """Check your data before training."""
    dataset = MRIDataset(root_dir, dataset_name=dataset_name)
    
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


def check_all_labels(root_dir, num_classes=3, dataset_name="Dataset001_MRI"):
    """Find any files with unexpected label values."""
    dataset = MRIDataset(root_dir, dataset_name=dataset_name)
    
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