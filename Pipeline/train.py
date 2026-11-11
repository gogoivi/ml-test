#!/usr/bin/env python3
"""
train_vae.py — Train 3D β-TCVAE on localized MRI crops.

Expects:
    - Completed crop pipeline with index.json in crop_dir
    - model.py with ConvVAE3D, beta_tcvae_loss, latent_traversal

Outputs:
    output_dir/
        checkpoints/best.pt, epoch_XXXX.pt
        viz/epoch_XXXX_recon.png, _traversal.png, _tsne.png
        tb_logs/   (TensorBoard)
        config.json, history.json, train.log
"""

import argparse
import gc
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # No display on server — render to file only
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import ConvVAE3D, beta_tcvae_loss, latent_traversal


# ═══════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════

class CropDataset(Dataset):
    """
    Reads the crop pipeline output (index.json + .npz files).
    Each .npz has left_crop and right_crop.
    
    We treat left and right as SEPARATE training samples because:
    - Doubles dataset: 229 patients → 458 crops
    - VAE learns side-agnostic anatomy features
    - Left/right identity may emerge as its own latent factor
    """

    def __init__(self, records, crop_dir, augment=False):
        self.crop_dir = Path(crop_dir)
        self.augment = augment

        # Flatten: each record → 2 items (left, right)
        self.items = []
        for rec in records:
            self.items.append((rec, "left"))
            self.items.append((rec, "right"))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec, side = self.items[idx]

        with np.load(self.crop_dir / rec["crop_file"]) as f:
            vol = torch.from_numpy(f[f"{side}_crop"].copy()).float()

        vol = vol.unsqueeze(0)  # (1, 64, 64, 64)

        if self.augment:
            vol = self._augment(vol)

        return {
            "volume": vol,
            "case_id": rec["case_id"],
            "side": side,
            "patient": rec["mri_path"],
        }

    def _augment(self, x):
        """
        MILD augmentation only. Heavy augmentation destroys the
        subtle anatomical variations we want the VAE to capture.
        """
        # Random flips — anatomy is roughly symmetric
        for dim in [1, 2, 3]:
            if random.random() > 0.5:
                x = x.flip(dim)

        # Tiny intensity jitter — prevents overfitting to exact values
        x = x * (1.0 + 0.03 * torch.randn(1))  # scale ±3%
        x = x + 0.03 * torch.randn(1)            # shift ±3%
        return x


def split_by_patient(records, val_frac=0.15, seed=42):
    """
    Split by PATIENT, not by crop.
    
    Why? Left and right of the same brain share anatomy. If left
    is in training and right in validation, the model has already
    "seen" that brain — that's data leakage, and val metrics
    would be falsely optimistic.
    """
    patients = sorted({r["mri_path"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(patients)

    n_val = max(1, int(len(patients) * val_frac))
    val_patients = set(patients[:n_val])

    train_recs = [r for r in records if r["mri_path"] not in val_patients]
    val_recs   = [r for r in records if r["mri_path"] in val_patients]

    return train_recs, val_recs, val_patients


# ═══════════════════════════════════════════════════════════════
# VISUALIZATIONS  (all save to disk, nothing displayed)
# ═══════════════════════════════════════════════════════════════

def save_reconstruction_grid(model, dataset, device, save_path, n_samples=6):
    """
    Side-by-side: original middle slice | reconstruction | absolute difference.
    Quick visual check that the VAE is learning meaningful features.
    """
    model.eval()
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    with torch.no_grad():
        for row, idx in enumerate(indices):
            vol = dataset[idx]["volume"].unsqueeze(0).to(device)
            recon, _, _, _ = model(vol)

            mid = vol.shape[2] // 2  # middle axial slice
            orig = vol[0, 0, mid].cpu().numpy()
            rec  = recon[0, 0, mid].cpu().numpy()
            diff = np.abs(orig - rec)

            axes[row, 0].imshow(orig, cmap="gray")
            axes[row, 0].set_title(f"Original [{dataset[idx]['case_id'][:15]}]")
            axes[row, 1].imshow(rec, cmap="gray")
            axes[row, 1].set_title("Reconstructed")
            im = axes[row, 2].imshow(diff, cmap="hot")
            axes[row, 2].set_title(f"Diff (max={diff.max():.3f})")

            for ax in axes[row]:
                ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_traversal_overlay(model, sample_vol, device, save_path, delta=2.0):
    """
    YOUR described visualization:
    - Encode one sample → z
    - Perturb each z_dim by delta
    - Each dimension gets a unique COLOR
    - OPACITY = how much that voxel changed
    
    Shows which brain regions each latent dimension controls.
    """
    model.eval()
    diff_maps = latent_traversal(model, sample_vol, device, delta=delta)
    z_dim = diff_maps.shape[0]
    mid = diff_maps.shape[1] // 2

    cmap = plt.colormaps["tab20"].resampled(z_dim)

    cols = min(4, z_dim)
    rows = math.ceil(z_dim / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    orig_slice = sample_vol[0, mid].numpy()  # (64, 64), middle axial

    for d in range(z_dim):
        ax = axes[d]
        ax.imshow(orig_slice, cmap="gray", alpha=0.5)

        diff_slice = diff_maps[d, mid].numpy()
        diff_norm = diff_slice / (diff_slice.max() + 1e-8)

        # Build RGBA overlay: dimension's color + opacity from diff magnitude
        color = np.array(cmap(d))[:3]
        overlay = np.zeros((*diff_slice.shape, 4))
        overlay[..., :3] = color
        overlay[..., 3] = diff_norm * 0.8

        ax.imshow(overlay)
        ax.set_title(f"z_{d}", fontsize=10)
        ax.axis("off")

    for d in range(z_dim, len(axes)):
        axes[d].set_visible(False)

    plt.suptitle("Latent Traversal — Region controlled by each dimension", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_latent_space_plot(model, dataset, device, save_path, max_samples=400):
    """
    Encode all samples → t-SNE to 2D → color by side and data source.
    
    What to look for:
    - Left/right forming separate clusters → model learned laterality
    - OpenNeuro vs Hopkins separation → dataset/scanner bias (concerning)
    - Smooth gradients → well-organized latent space
    """
    from sklearn.manifold import TSNE

    model.eval()
    latents, sides, sources = [], [], []

    n = min(max_samples, len(dataset))
    with torch.no_grad():
        for idx in range(n):
            item = dataset[idx]
            vol = item["volume"].unsqueeze(0).to(device)
            mu, _ = model.encode(vol)
            latents.append(mu[0].cpu().numpy())
            sides.append(item["side"])
            sources.append(
                "OpenNeuro" if "OpenNeuro" in item["patient"] else "Hopkins"
            )

    latents = np.stack(latents)
    perp = min(30, n // 4)
    coords = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(latents)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for side, c in [("left", "tab:blue"), ("right", "tab:red")]:
        mask = np.array([s == side for s in sides])
        ax1.scatter(coords[mask, 0], coords[mask, 1], c=c, label=side, alpha=0.6, s=20)
    ax1.set_title("Colored by side")
    ax1.legend()

    for src, c in [("OpenNeuro", "tab:green"), ("Hopkins", "tab:orange")]:
        mask = np.array([s == src for s in sources])
        ax2.scatter(coords[mask, 0], coords[mask, 1], c=c, label=src, alpha=0.6, s=20)
    ax2.set_title("Colored by data source")
    ax2.legend()

    plt.suptitle("Latent Space (t-SNE)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_extremes(model, dataset, device, save_path, n_extreme=5):
    """
    For each z dimension: find the crops with HIGHEST and LOWEST values.
    Show their middle slices side by side.
    
    This is arguably MORE informative than traversal because it shows
    REAL anatomy, not decoder hallucinations. If z_3 separates
    "large ventricle" from "small ventricle," you'll see it directly.
    """
    model.eval()
    all_mu, all_slices = [], []

    with torch.no_grad():
        for idx in range(len(dataset)):
            item = dataset[idx]
            vol = item["volume"].unsqueeze(0).to(device)
            mu, _ = model.encode(vol)
            all_mu.append(mu[0].cpu())
            mid = vol.shape[2] // 2
            all_slices.append(vol[0, 0, mid].cpu().numpy())

    all_mu = torch.stack(all_mu)
    z_dim = all_mu.shape[1]

    fig, axes = plt.subplots(
        z_dim, 2 * n_extreme,
        figsize=(2.5 * 2 * n_extreme, 2.5 * z_dim)
    )

    for d in range(z_dim):
        values = all_mu[:, d]
        low_idx  = values.argsort()[:n_extreme]
        high_idx = values.argsort()[-n_extreme:]

        for j, idx in enumerate(low_idx):
            axes[d, j].imshow(all_slices[idx], cmap="gray")
            axes[d, j].set_title(f"{values[idx]:.1f}", fontsize=7)
            axes[d, j].axis("off")

        for j, idx in enumerate(high_idx):
            col = n_extreme + j
            axes[d, col].imshow(all_slices[idx], cmap="gray")
            axes[d, col].set_title(f"{values[idx]:.1f}", fontsize=7)
            axes[d, col].axis("off")

        axes[d, 0].set_ylabel(f"z_{d}", fontsize=9, rotation=0, labelpad=25)

    plt.suptitle(
        f"← {n_extreme} LOWEST values  |  {n_extreme} HIGHEST values →",
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, device, beta, dataset_size):
    model.train()
    totals = {"loss": 0, "recon": 0, "mi": 0, "tc": 0, "dimwise_kl": 0}
    n = 0

    for batch in loader:
        vol = batch["volume"].to(device)
        x_recon, mu, log_var, z = model(vol)

        loss, parts = beta_tcvae_loss(
            vol, x_recon, mu, log_var, z,
            beta=beta, dataset_size=dataset_size,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        for k, v in parts.items():
            totals[k] += v
        totals["loss"] += loss.item()
        n += 1

    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def val_epoch(model, loader, device, beta, dataset_size):
    model.eval()
    totals = {"loss": 0, "recon": 0, "mi": 0, "tc": 0, "dimwise_kl": 0}
    n = 0

    for batch in loader:
        vol = batch["volume"].to(device)
        x_recon, mu, log_var, z = model(vol)

        loss, parts = beta_tcvae_loss(
            vol, x_recon, mu, log_var, z,
            beta=beta, dataset_size=dataset_size,
        )

        for k, v in parts.items():
            totals[k] += v
        totals["loss"] += loss.item()
        n += 1

    return {k: v / max(n, 1) for k, v in totals.items()}


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train 3D β-TCVAE")
    parser.add_argument("--crop-dir", required=True,
                        help="Directory with index.json + .npz files from crop pipeline")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--z-dim", type=int, default=16,
                        help="Latent dimensions. 16 is a good start for ~458 samples")
    parser.add_argument("--beta", type=float, default=4.0,
                        help="TC weight. Higher = more disentangled, worse reconstruction")
    parser.add_argument("--beta-warmup", type=int, default=30,
                        help="Epochs to linearly ramp β from 0 → target. "
                             "Prevents posterior collapse: gives encoder time to "
                             "learn useful features before disentanglement pressure")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Larger = better TC estimate. 32 is good for 458 samples")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--viz-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    # --- Output structure ---
    out = Path(args.output_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "viz").mkdir(exist_ok=True)
    (out / "tb_logs").mkdir(exist_ok=True)

    # --- Logging: both file AND terminal (visible in tmux + saved) ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(out / "train.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger(__name__)

    config = vars(args)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    log.info(f"Config: {json.dumps(config)}")

    # --- Reproducibility ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name()}")

    # --- Load crop index ---
    crop_dir = Path(args.crop_dir)
    index_path = crop_dir / "index.json"
    if not index_path.is_file():
        log.error(f"No index.json found at {index_path}")
        log.error("Run the crop pipeline first (infer_loc_crops.py --mode infer)")
        sys.exit(1)

    with index_path.open() as f:
        all_records = json.load(f)
    log.info(f"Total MRIs in index: {len(all_records)}")

    # --- Split by patient ---
    train_recs, val_recs, val_patients = split_by_patient(
        all_records, val_frac=args.val_frac, seed=args.seed,
    )
    log.info(f"Train: {len(train_recs)} MRIs → {len(train_recs)*2} crops")
    log.info(f"Val:   {len(val_recs)} MRIs → {len(val_recs)*2} crops")

    train_set = CropDataset(train_recs, crop_dir, augment=True)
    val_set   = CropDataset(val_recs,   crop_dir, augment=False)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,   # TC estimation needs consistent batch size
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    dataset_size = len(train_set)  # For TC denominator in loss

    # --- Model ---
    model = ConvVAE3D(z_dim=args.z_dim, in_channels=1, input_size=64).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Parameters: {n_params:,}")

    optimizer = Adam(model.parameters(), lr=args.lr)
    # Cosine decay: lr gradually drops to 1% of initial over training
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    writer = SummaryWriter(out / "tb_logs")

    # --- Fixed visualization sample (same crop every epoch for comparison) ---
    viz_sample = val_set[0]["volume"]  # (1, 64, 64, 64)

    best_val_loss = float("inf")
    history = []

    log.info(f"\n{'='*60}")
    log.info(f"Training: {args.epochs} epochs, β={args.beta} (warmup={args.beta_warmup})")
    log.info(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # --- β warmup: linearly ramp from 0 to target ---
        # Without this, strong TC penalty from epoch 1 causes "posterior
        # collapse" — all z dimensions go to N(0,1), encoding nothing useful.
        # Warmup lets the encoder learn FIRST, then slowly enforces independence.
        if epoch <= args.beta_warmup:
            beta = args.beta * (epoch / args.beta_warmup)
        else:
            beta = args.beta

        train_m = train_epoch(model, train_loader, optimizer, device, beta, dataset_size)
        val_m   = val_epoch(model, val_loader, device, beta, dataset_size)
        scheduler.step()

        dt = time.time() - t0

        log.info(
            f"E{epoch:03d}/{args.epochs} ({dt:.0f}s) β={beta:.2f} | "
            f"T: loss={train_m['loss']:.2f} recon={train_m['recon']:.2f} "
            f"tc={train_m['tc']:.3f} mi={train_m['mi']:.3f} | "
            f"V: loss={val_m['loss']:.2f} recon={val_m['recon']:.2f}"
        )

        # --- TensorBoard ---
        for prefix, m in [("train", train_m), ("val", val_m)]:
            for k, v in m.items():
                writer.add_scalar(f"{prefix}/{k}", v, epoch)
        writer.add_scalar("schedule/beta", beta, epoch)
        writer.add_scalar("schedule/lr", scheduler.get_last_lr()[0], epoch)

        # --- Best checkpoint ---
        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss,
                "config": config,
            }, out / "checkpoints" / "best.pt")
            log.info(f"  ★ New best val loss: {best_val_loss:.4f}")

        # --- Periodic checkpoint ---
        if epoch % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": config,
            }, out / "checkpoints" / f"epoch_{epoch:04d}.pt")

        # --- Visualizations ---
        if epoch % args.viz_every == 0 or epoch == 1:
            log.info("  Saving visualizations...")
            save_reconstruction_grid(
                model, val_set, device,
                out / "viz" / f"epoch_{epoch:04d}_recon.png",
            )
            save_traversal_overlay(
                model, viz_sample, device,
                out / "viz" / f"epoch_{epoch:04d}_traversal.png",
            )
            # t-SNE is slow — run less often
            if epoch % (args.viz_every * 4) == 0 or epoch == args.epochs:
                save_latent_space_plot(
                    model, val_set, device,
                    out / "viz" / f"epoch_{epoch:04d}_tsne.png",
                )

            # Free GPU cache after heavy viz
            if device.type == "cuda":
                torch.cuda.empty_cache()

        history.append({
            "epoch": epoch, "beta": beta,
            "train": train_m, "val": val_m, "time_s": dt,
        })

    # --- Final visualizations on ALL data ---
    log.info("\nFinal visualizations (all data)...")
    all_dataset = CropDataset(all_records, crop_dir, augment=False)

    save_extremes(model, all_dataset, device, out / "viz" / "extremes.png")
    save_latent_space_plot(model, all_dataset, device, out / "viz" / "final_tsne.png")

    # --- Save history ---
    (out / "history.json").write_text(json.dumps(history, indent=2))
    writer.close()

    log.info(f"\nDone. Best val loss: {best_val_loss:.4f}")
    log.info(f"Output: {out}")


if __name__ == "__main__":
    main()