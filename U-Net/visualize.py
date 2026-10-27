import os
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from scipy.ndimage import label as scipy_label
import random
import json
from datetime import datetime

# Your imports - adjust paths as needed
from model import U_Net
from load_data import MRIDataset


def connected_comps(array, class_value):
    """
    Find connected components for a specific class value.
    
    How it works:
    1. Create binary mask where target class = 1, everything else = 0
    2. scipy.ndimage.label finds all connected regions
    3. Returns array where each connected component has unique integer ID
    
    Args:
        array: 3D segmentation array with multiple classes
        class_value: The class to find components for
    
    Returns:
        Labeled array where each connected component has a unique integer (1, 2, 3, ...)
    """
    binary_mask = (array == class_value).astype(np.uint8)
    labeled_array, num_features = scipy_label(binary_mask)
    return labeled_array


def post_process_segmentation(pred_array, nerve_class=1, vessel_class=2, 
                               vessel_min_size=50):
    """
    Apply class-specific post-processing:
    
    NERVE (class 1): Keep ONLY the largest connected component
        - Rationale: There's typically one main nerve trunk
        - Small disconnected predictions are likely false positives
    
    VESSEL (class 2): Remove small components, keep ALL large ones
        - Rationale: Multiple separate vessels are anatomically correct
        - Only remove tiny specks that are likely noise
        - Long continuous structures naturally have high voxel counts, so they're kept
    
    Args:
        pred_array: Raw prediction array (D, H, W) with class labels 0/1/2
        nerve_class: Class label for nerve (default 1)
        vessel_class: Class label for vessel (default 2)
        vessel_min_size: Minimum voxel count for vessel components to keep
    
    Returns:
        Post-processed segmentation array
    """
    result = np.zeros_like(pred_array)
    
    # === NERVE: Keep only the largest connected component ===
    nerve_mask = (pred_array == nerve_class)
    if nerve_mask.sum() > 0:
        labeled_nerve, num_components = scipy_label(nerve_mask.astype(np.uint8))
        
        # Find the largest component
        max_size = 0
        largest_label = 0
        for label_id in range(1, num_components + 1):
            size = (labeled_nerve == label_id).sum()
            if size > max_size:
                max_size = size
                largest_label = label_id
        
        # Keep only the largest
        if largest_label > 0:
            result[labeled_nerve == largest_label] = nerve_class
            
        print(f"    Nerve: {num_components} components found, kept largest ({max_size} voxels)")
    
    # === VESSEL: Remove small components, keep all large ones ===
    vessel_mask = (pred_array == vessel_class)
    if vessel_mask.sum() > 0:
        labeled_vessel, num_components = scipy_label(vessel_mask.astype(np.uint8))
        
        kept_count = 0
        removed_count = 0
        
        for label_id in range(1, num_components + 1):
            component_mask = (labeled_vessel == label_id)
            component_size = component_mask.sum()
            
            # Keep if larger than threshold (long vessels will easily pass this)
            if component_size >= vessel_min_size:
                result[component_mask] = vessel_class
                kept_count += 1
            else:
                removed_count += 1
        
        print(f"    Vessel: {num_components} components found, kept {kept_count}, removed {removed_count} (< {vessel_min_size} voxels)")
    
    return result


# ============== MODEL LOADING ==============

def load_model(checkpoint_path, size=48, num_classes=3, kernel_size=3, device='cuda'):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to best_model.pth
        size: Model size parameter (must match training)
        num_classes: Number of output classes (must match training)
        kernel_size: Kernel size (must match training)
        device: 'cuda' or 'cpu'
    
    Returns:
        Loaded model in eval mode
    """
    print(f"  load_model: checking path exists...")
    print(f"  Exists: {os.path.exists(checkpoint_path)}")
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Initialize model with same architecture as training
    model = U_Net(
        size=size,
        classes=num_classes,
        color=False,
        kernel_size=kernel_size
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Print checkpoint info
    print(f"Loaded model from: {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    if 'val_loss' in checkpoint:
        print(f"  Val loss: {checkpoint['val_loss']:.4f}")
    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        print(f"  Dice class 1: {metrics.get('dice_class_1', 'N/A'):.4f}")
        print(f"  Dice class 2: {metrics.get('dice_class_2', 'N/A'):.4f}")
    
    return model


# ============== INFERENCE ==============

def run_inference(model, image, device='cuda'):
    """
    Run model inference on a single image.
    
    Args:
        model: Trained model in eval mode
        image: Input tensor (1, D, H, W) from dataloader
        device: Device to run on
    
    Returns:
        Predicted class labels (D, H, W) as numpy array
    """
    model.eval()
    
    # Add batch dimension: (1, D, H, W) -> (1, 1, D, H, W)
    if image.dim() == 4:
        image = image.unsqueeze(0)
    
    image = image.to(device)
    
    with torch.no_grad():
        pred_logits = model(image)  # (1, num_classes, D, H, W)
        pred_classes = torch.argmax(pred_logits, dim=1)  # (1, D, H, W)
    
    return pred_classes[0].cpu().numpy()  # (D, H, W)


# ============== FILE SAVING ==============

def save_nifti(array, output_path):
    """
    Save a 3D numpy array as a NIfTI file (.nii.gz).
    
    Args:
        array: 3D numpy array (D, H, W)
        output_path: Path to save (should end in .nii.gz)
    """
    nii = nib.Nifti1Image(array, affine=np.eye(4))
    nib.save(nii, output_path)


def save_sample_results(output_dir, sample_name, image_np, gt_np, pred_raw, pred_postproc):
    """
    Save all volumes for a single sample as NIfTI files in a grouped folder.
    
    Creates:
        output_dir/sample_name/
            ├── image.nii.gz                    (preprocessed MRI)
            ├── ground_truth.nii.gz             (labels)
            ├── prediction_raw.nii.gz           (raw model output)
            └── prediction_postprocessed.nii.gz (after post-processing)
    
    Args:
        output_dir: Root output directory
        sample_name: Name for this sample's folder
        image_np: Preprocessed MRI (D, H, W) float32
        gt_np: Ground truth labels (D, H, W) int
        pred_raw: Raw prediction (D, H, W) int
        pred_postproc: Post-processed prediction (D, H, W) int
    """
    sample_dir = os.path.join(output_dir, sample_name)
    os.makedirs(sample_dir, exist_ok=True)
    
    save_nifti(image_np.astype(np.float32), os.path.join(sample_dir, "image.nii.gz"))
    save_nifti(gt_np.astype(np.int16), os.path.join(sample_dir, "ground_truth.nii.gz"))
    save_nifti(pred_raw.astype(np.int16), os.path.join(sample_dir, "prediction_raw.nii.gz"))
    save_nifti(pred_postproc.astype(np.int16), os.path.join(sample_dir, "prediction_postprocessed.nii.gz"))
    
    print(f"  Saved 4 NIfTI files to: {sample_dir}")


def compute_dice(pred, target, class_id):
    """Compute Dice score for a single class."""
    pred_c = (pred == class_id).astype(float)
    target_c = (target == class_id).astype(float)
    
    intersection = (pred_c * target_c).sum()
    union = pred_c.sum() + target_c.sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (2 * intersection) / union


# ============== MAIN ==============

def main():
    """Main function to load model, run inference on random samples, and save results."""
    SEED = 0
    random.seed(SEED)
    # ============== CONFIGURATION ==============
    # UPDATE THIS PATH to your trained model!
    CONFIG = {
        'checkpoint_path': r"/home/vgogoi1/projects/TN_Research/U-Net/checkpoints/Cubical_TopoLoss(300 epochs)_20261026_083611/model_a_best.pt",
        'data_root': r"/home/vgogoi1/projects/TN_Research/2023 Patients",
        'output_dir': r"/home/vgogoi1/projects/TN_Research/inference_results",
        'file_type': 'all',  # 'all', 'nifti', or 'tiff'
        'num_samples': 3,    # How many random samples to process
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # Model architecture (must match training!)
        'model_size': 48,
        'num_classes': 3,
        'kernel_size': 3,
        
        # Post-processing settings
        'vessel_min_size': 50,  # Minimum voxels for vessel components
    }
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(CONFIG['output_dir'], f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("MODEL INFERENCE AND RESULTS SAVING")
    print("=" * 60)
    print(f"Device: {CONFIG['device']}")
    print(f"Data: {CONFIG['file_type']}")
    print(f"Samples to process: {CONFIG['num_samples']}")
    print(f"Output directory: {output_dir}")
    print()
    
    # ============== LOAD MODEL ==============
    print("Loading model...")
    model = load_model(
        checkpoint_path=CONFIG['checkpoint_path'],
        size=CONFIG['model_size'],
        num_classes=CONFIG['num_classes'],
        kernel_size=CONFIG['kernel_size'],
        device=CONFIG['device']
    )
    print()
    
    # ============== LOAD DATASET ==============
    print("Loading dataset...")
    dataset = MRIDataset(
        root_dir=CONFIG['data_root'],
        file_type=CONFIG['file_type'],
        augment=False  # No augmentation for inference!
    )
    print()
    
    # ============== SELECT RANDOM SAMPLES ==============
    num_samples = min(CONFIG['num_samples'], len(dataset))
    random_indices = random.sample(range(len(dataset)), num_samples)
    
    print(f"Randomly selected indices: {random_indices}")
    print()
    
    # ============== PROCESS EACH SAMPLE ==============
    all_metrics = []
    
    for i, idx in enumerate(random_indices):
        print(f"{'=' * 60}")
        print(f"SAMPLE {i+1}/{num_samples} (Dataset index: {idx})")
        print(f"{'=' * 60}")
        
        # Get sample info
        input_path, target_path, file_type = dataset.samples[idx]
        sample_name = os.path.basename(input_path).replace('.nii.gz', '').replace('.tiff', '').replace('.tif', '')
        print(f"  File: {sample_name}")
        print(f"  Type: {file_type}")
        
        image, ground_truth = dataset[idx]
        
        image_np = image[0].numpy()  # Remove channel dim -> (D, H, W)
        gt_np = ground_truth.numpy()
        
        # Run inference
        print("  Running inference...")
        pred_raw = run_inference(model, image, CONFIG['device'])
        
        # Post-process
        print("  Post-processing...")
        pred_postproc = post_process_segmentation(
            pred_raw,
            nerve_class=1,
            vessel_class=2,
            vessel_min_size=CONFIG['vessel_min_size']
        )
        
        # Compute metrics
        print("  Computing metrics...")
        dice_nerve_raw = compute_dice(pred_raw, gt_np, class_id=1)
        dice_vessel_raw = compute_dice(pred_raw, gt_np, class_id=2)
        dice_nerve_post = compute_dice(pred_postproc, gt_np, class_id=1)
        dice_vessel_post = compute_dice(pred_postproc, gt_np, class_id=2)
        
        print(f"\n  Results:")
        print(f"  {'':20} {'Nerve':>12} {'Vessel':>12}")
        print(f"  {'-'*46}")
        print(f"  {'Ground Truth voxels':20} {(gt_np == 1).sum():>12} {(gt_np == 2).sum():>12}")
        print(f"  {'Raw pred voxels':20} {(pred_raw == 1).sum():>12} {(pred_raw == 2).sum():>12}")
        print(f"  {'Post-proc voxels':20} {(pred_postproc == 1).sum():>12} {(pred_postproc == 2).sum():>12}")
        print(f"  {'-'*46}")
        print(f"  {'Dice (raw)':20} {dice_nerve_raw:>12.4f} {dice_vessel_raw:>12.4f}")
        print(f"  {'Dice (post-proc)':20} {dice_nerve_post:>12.4f} {dice_vessel_post:>12.4f}")
        print()
        
        # Save NIfTI files
        folder_name = f"sample_{i+1:03d}_{sample_name}"
        save_sample_results(output_dir, folder_name, image_np, gt_np, pred_raw, pred_postproc)
        
        # Store metrics
        sample_metrics = {
            'sample_index': idx,
            'sample_name': sample_name,
            'file_type': file_type,
            'folder_name': folder_name,
            'gt_nerve_voxels': int((gt_np == 1).sum()),
            'gt_vessel_voxels': int((gt_np == 2).sum()),
            'raw_nerve_voxels': int((pred_raw == 1).sum()),
            'raw_vessel_voxels': int((pred_raw == 2).sum()),
            'postproc_nerve_voxels': int((pred_postproc == 1).sum()),
            'postproc_vessel_voxels': int((pred_postproc == 2).sum()),
            'dice_nerve_raw': round(dice_nerve_raw, 4),
            'dice_vessel_raw': round(dice_vessel_raw, 4),
            'dice_nerve_postproc': round(dice_nerve_post, 4),
            'dice_vessel_postproc': round(dice_vessel_post, 4),
        }
        all_metrics.append(sample_metrics)
    
    # ============== SAVE SUMMARY ==============
    summary = {
        'config': {
            'checkpoint_path': CONFIG['checkpoint_path'],
            'data_root': CONFIG['data_root'],
            'file_type': CONFIG['file_type'],
            'num_samples': num_samples,
            'vessel_min_size': CONFIG['vessel_min_size'],
            'seed': SEED,
        },
        'aggregate': {
            'mean_dice_nerve_raw': round(np.mean([m['dice_nerve_raw'] for m in all_metrics]), 4),
            'mean_dice_vessel_raw': round(np.mean([m['dice_vessel_raw'] for m in all_metrics]), 4),
            'mean_dice_nerve_postproc': round(np.mean([m['dice_nerve_postproc'] for m in all_metrics]), 4),
            'mean_dice_vessel_postproc': round(np.mean([m['dice_vessel_postproc'] for m in all_metrics]), 4),
        },
        'samples': all_metrics,
    }
    
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    # ============== PRINT FINAL SUMMARY ==============
    print("=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print(f"Summary saved to: {summary_path}")
    print()
    print("Output structure:")
    print(f"  {output_dir}/")
    print(f"  ├── summary.json")
    for m in all_metrics:
        print(f"  ├── {m['folder_name']}/")
        print(f"  │   ├── image.nii.gz")
        print(f"  │   ├── ground_truth.nii.gz")
        print(f"  │   ├── prediction_raw.nii.gz")
        print(f"  │   └── prediction_postprocessed.nii.gz")
    print()
    print("Aggregate Dice Scores:")
    print(f"  {'':20} {'Nerve':>12} {'Vessel':>12}")
    print(f"  {'-'*46}")
    print(f"  {'Mean Dice (raw)':20} {summary['aggregate']['mean_dice_nerve_raw']:>12.4f} {summary['aggregate']['mean_dice_vessel_raw']:>12.4f}")
    print(f"  {'Mean Dice (post)':20} {summary['aggregate']['mean_dice_nerve_postproc']:>12.4f} {summary['aggregate']['mean_dice_vessel_postproc']:>12.4f}")
    print()
    print("To view in napari, open a sample folder and drag all 4 files in,")
    print("or run:  napari <folder>/image.nii.gz <folder>/ground_truth.nii.gz <folder>/prediction_postprocessed.nii.gz")


if __name__ == "__main__":
    main()

# To run
# napari sample_001_MRI_0042/image.nii.gz sample_001_MRI_0042/ground_truth.nii.gz sample_001_MRI_0042/prediction_postprocessed.nii.gz