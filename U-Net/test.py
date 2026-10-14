import numpy as np
import torch
import torch.nn.functional as F
from loss_fn import get_persistence_info, match_diagrams, compute_topo_gradient
# Adjust import path as needed

def debug_matching_stats(pred_volume, target_volume, label=""):
    # Threshold pred like the loss does
    pred_binary = (pred_volume > 0.5).astype(float)
    pred_info = get_persistence_info(pred_binary)
    
    # Replace with continuous values
    for p in pred_info:
        p['birth_value'] = pred_volume[p['birth_coords']]
        if p['death_coords'] is not None:
            p['death_value'] = pred_volume[p['death_coords']]
    
    target_info = get_persistence_info(target_volume)
    
    # ADD THIS DEBUG PRINT
    print(f"\nPred features by dimension:")
    for dim in range(3):
        feats = [p for p in pred_info if p['dim'] == dim]
        finite = [p for p in feats if p['death_coords'] is not None]
        infinite = [p for p in feats if p['death_coords'] is None]
        print(f"  H{dim}: {len(finite)} finite, {len(infinite)} infinite")
    
    print(f"\nTarget features by dimension:")
    for dim in range(3):
        feats = [t for t in target_info if t['dim'] == dim]
        finite = [t for t in feats if t['death_coords'] is not None]
        infinite = [t for t in feats if t['death_coords'] is None]
        print(f"  H{dim}: {len(finite)} finite, {len(infinite)} infinite")
    
    matches = match_diagrams(pred_info, target_info)
    
    # Count match types
    matched_pairs = [(p, t) for p, t in matches if p is not None and t is not None]
    pred_to_diagonal = [(p, t) for p, t in matches if p is not None and t is None]
    missing_targets = [(p, t) for p, t in matches if p is None and t is not None]
    
    print(f"\n{'='*50}")
    print(f"DIAGNOSTIC: {label}")
    print(f"{'='*50}")
    print(f"Pred features total: {len(pred_info)}")
    print(f"Target features total: {len(target_info)}")
    print(f"")
    print(f"Matched pairs (pred ↔ target): {len(matched_pairs)}")
    print(f"Extra in pred (pred → diagonal): {len(pred_to_diagonal)}")
    print(f"Missing in pred (target unmatched): {len(missing_targets)}")
    
    # Compute gradient and analyze contributions
    grad = compute_topo_gradient(matches, pred_volume)
    
    # Compute gradient contributions separately
    grad_matched = np.zeros_like(pred_volume)
    grad_extra = np.zeros_like(pred_volume)
    grad_missing = np.zeros_like(pred_volume)
    
    for pred_dot, target_dot in matches:
        if pred_dot is not None and target_dot is not None:
            grad_matched[pred_dot['birth_coords']] += 2 * (pred_dot['birth_value'] - target_dot['birth_value'])
            if pred_dot['death_coords'] is not None:  # ADD THIS CHECK
                target_death = 0.0 if target_dot['death_value'] == np.inf else target_dot['death_value']
                grad_matched[pred_dot['death_coords']] += 2 * (pred_dot['death_value'] - target_death)
                
        elif pred_dot is not None and target_dot is None:
            d_eff = 0.0 if pred_dot['death_value'] == np.inf else pred_dot['death_value']
            diagonal_point = (pred_dot['birth_value'] + d_eff) / 2
            grad_extra[pred_dot['birth_coords']] += 2 * (pred_dot['birth_value'] - diagonal_point)
            if pred_dot['death_coords'] is not None:  # ADD THIS CHECK
                grad_extra[pred_dot['death_coords']] += 2 * (pred_dot['death_value'] - diagonal_point)
                
        elif pred_dot is None and target_dot is not None:
            birth_coords = target_dot['birth_coords']
            grad_missing[birth_coords] += 2 * (pred_volume[birth_coords] - target_dot['birth_value'])
    
    print(f"\nGradient magnitudes (sum of abs):")
    print(f"  From matched pairs:     {np.abs(grad_matched).sum():.4f}")
    print(f"  From extra (→diagonal): {np.abs(grad_extra).sum():.4f}")
    print(f"  From missing targets:   {np.abs(grad_missing).sum():.4f}")
    print(f"  TOTAL:                  {np.abs(grad).sum():.4f}")
    
    # Show ratio
    total = np.abs(grad).sum()
    if total > 0:
        print(f"\nGradient contribution ratios:")
        print(f"  Matched:  {100*np.abs(grad_matched).sum()/total:.1f}%")
        print(f"  Extra:    {100*np.abs(grad_extra).sum()/total:.1f}%")
        print(f"  Missing:  {100*np.abs(grad_missing).sum()/total:.1f}%")

    print(f"\nPred features details:")
    for i, p in enumerate(pred_info):
        print(f"  {i}: dim={p['dim']}, birth={p['birth_value']:.3f}, death={p['death_value'] if p['death_value'] != np.inf else 'inf'}, finite={p['death_coords'] is not None}")

    print(f"\nTarget features details:")
    for i, t in enumerate(target_info):
        print(f"  {i}: dim={t['dim']}, birth={t['birth_value']:.3f}, death={t['death_value'] if t['death_value'] != np.inf else 'inf'}, finite={t['death_coords'] is not None}")
    
    # Show birth/death value distributions
    print(f"\nBirth/Death value distributions:")
    if pred_info:
        pred_births = [p['birth_value'] for p in pred_info if p['death_coords'] is not None]
        pred_deaths = [p['death_value'] for p in pred_info if p['death_coords'] is not None]
        if pred_births:
            print(f"  Pred births:  min={min(pred_births):.3f}, max={max(pred_births):.3f}, mean={np.mean(pred_births):.3f}")
        else:
            print("No pred births")
        if pred_deaths:
            print(f"  Pred deaths:  min={min(pred_deaths):.3f}, max={max(pred_deaths):.3f}, mean={np.mean(pred_deaths):.3f}")
        else:
            print("No pred deaths")
    
    if target_info:
        tgt_births = [t['birth_value'] for t in target_info if t['death_coords'] is not None]
        tgt_deaths = [t['death_value'] for t in target_info if t['death_coords'] is not None]
        if tgt_births:
            print(f"  Target births: min={min(tgt_births):.3f}, max={max(tgt_births):.3f}, mean={np.mean(tgt_births):.3f}")
            print(f"  Target deaths: min={min(tgt_deaths):.3f}, max={max(tgt_deaths):.3f}, mean={np.mean(tgt_deaths):.3f}")
    
    return {
        'n_pred': len(pred_info),
        'n_target': len(target_info),
        'n_matched': len(matched_pairs),
        'n_extra': len(pred_to_diagonal),
        'n_missing': len(missing_targets),
        'grad_matched': np.abs(grad_matched).sum(),
        'grad_extra': np.abs(grad_extra).sum(),
        'grad_missing': np.abs(grad_missing).sum(),
    }


def test_with_synthetic():
    """Test with synthetic data to understand behavior"""
    print("\n" + "="*60)
    print("TEST 1: Synthetic - Perfect prediction")
    print("="*60)
    target = np.zeros((16, 16, 16))
    target[4:12, 7:9, 7:9] = 1.0  # A tube
    pred = target.copy()  # Perfect
    debug_matching_stats(pred, target, "Perfect prediction")
    
    print("\n" + "="*60)
    print("TEST 2: Synthetic - Good prediction (soft probabilities)")
    print("="*60)
    pred_soft = np.where(target > 0, 0.85, 0.15)  # Soft version
    pred_soft += np.random.normal(0, 0.05, pred_soft.shape)
    pred_soft = np.clip(pred_soft, 0, 1)
    debug_matching_stats(pred_soft, target, "Soft prediction (~0.85 vs 0.15)")
    
    print("\n" + "="*60)
    print("TEST 3: Synthetic - Broken structure")
    print("="*60)
    pred_broken = target.copy()
    pred_broken[7:9, :, :] = 0  # Break the tube
    debug_matching_stats(pred_broken, target, "Broken tube")


def test_with_real_data():
    """Test with your actual data"""
    from load_data import get_dataloaders  # Adjust import
    ROOT_DIR = r"C:\Users\gogoi\Desktop\ml-test\2023 Patients"
    
    train_loader, val_loader = get_dataloaders(ROOT_DIR, batch_size=2)
    
    # Get one batch
    batch = next(iter(val_loader))
    images, targets = batch
    
    # Simulate a "decent" prediction (you could also load a trained model)
    # For now, let's just create a noisy version of target
    target_np = (targets[0] == 1).float().cpu().numpy()  # Class 1
    
    # Simulate prediction: target + noise, then sigmoid-like transform
    pred_np = target_np * 0.7 + 0.15 + np.random.normal(0, 0.1, target_np.shape)
    pred_np = np.clip(pred_np, 0, 1)
    
    debug_matching_stats(pred_np, target_np, "Real data - Class 1 (simulated pred)")
    
    # Class 2
    target_np_c2 = (targets[0] == 2).float().cpu().numpy()
    pred_np_c2 = target_np_c2 * 0.7 + 0.15 + np.random.normal(0, 0.1, target_np_c2.shape)
    pred_np_c2 = np.clip(pred_np_c2, 0, 1)
    
    debug_matching_stats(pred_np_c2, target_np_c2, "Real data - Class 2 (simulated pred)")

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
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    from model import U_Net  # Your model
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
    print(f"  Val loss: {checkpoint.get('val_loss', 'unknown'):.4f}" if 'val_loss' in checkpoint else "")
    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        print(f"  Dice class 1: {metrics.get('dice_class_1', 'N/A'):.4f}")
        print(f"  Dice class 2: {metrics.get('dice_class_2', 'N/A'):.4f}")
    
    return model

def test_with_trained_model():
    """Test with actual model predictions"""

    CONFIG = {
        'checkpoint_path': r"C:\Users\gogoi\Desktop\ml-test\overnight_results\comparison_20261011_061709\20261011_070638_CE_Dice_clDice\checkpoints\best_model.pth",
        'data_root': r"C:\Users\gogoi\Desktop\ml-test\2023 Patients",
        'file_type': 'all',  # 'all', 'nifti', or 'tiff'
        'num_samples': 3,    # How many random samples to visualize
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # Model architecture (must match training!)
        'model_size': 48,
        'num_classes': 3,
        'kernel_size': 3,
        
        # Post-processing settings
        'vessel_min_size': 50,  # Minimum voxels for vessel components
    }

    
    from load_data import get_dataloaders
    
    ROOT_DIR = r"C:\Users\gogoi\Desktop\ml-test\2023 Patients"
    _, val_loader = get_dataloaders(ROOT_DIR, batch_size=1)
    
    # Load your best CE+Dice checkpoint
    model = load_model(
        checkpoint_path=CONFIG['checkpoint_path'],
        size=CONFIG['model_size'],
        num_classes=CONFIG['num_classes'],
        kernel_size=CONFIG['kernel_size'],
        device=CONFIG['device']
    )
    model.eval()
    
    _, val_loader = get_dataloaders(ROOT_DIR, batch_size=1)
    
    # Test multiple samples
    results = []
    for i, (images, targets) in enumerate(val_loader):
        if i >= 10:  # Test first 10 samples
            break
            
        with torch.no_grad():
            logits = model(images.cuda())
            probs = torch.softmax(logits, dim=1)
        
        # Class 1
        pred_np = probs[0, 1].cpu().numpy()
        target_np = (targets[0] == 1).float().cpu().numpy()
        
        stats = debug_matching_stats(pred_np, target_np, f"Sample {i} - Class 1")
        results.append(stats)
        # In your loop, also test Class 2
        pred_np_c2 = probs[0, 2].cpu().numpy()  # Class 2
        target_np_c2 = (targets[0] == 2).float().cpu().numpy()
        stats_c2 = debug_matching_stats(pred_np_c2, target_np_c2, f"Sample {i} - Class 2")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY ACROSS ALL SAMPLES")
    print("="*60)
    total_matched = sum(r['n_matched'] for r in results)
    total_extra = sum(r['n_extra'] for r in results)
    total_missing = sum(r['n_missing'] for r in results)
    total_grad = sum(r['grad_matched'] + r['grad_extra'] for r in results)
    
    print(f"Total matched: {total_matched}")
    print(f"Total extra (noise): {total_extra}")
    print(f"Total missing: {total_missing}")
    print(f"Total gradient magnitude: {total_grad:.4f}")
    
    # Find worst samples
    print("\nSamples with topological errors:")
    for i, r in enumerate(results):
        if r['n_extra'] > 0 or r['n_missing'] > 0:
            print(f"  Sample {i}: {r['n_extra']} extra, {r['n_missing']} missing")

if __name__ == "__main__":
    # test_with_synthetic()
    
    # Uncomment to test with real data:
    # test_with_real_data()

    test_with_trained_model()