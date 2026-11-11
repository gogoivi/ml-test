#!/usr/bin/env python3
"""Infer/save S0 left and right MRI crops, or load saved crops with a DataLoader."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path("/home/jzhou169/tn_locseg")
DEFAULT_PREPARED = ROOT / "prepared_data"
DEFAULT_CHECKPOINT = (
    ROOT / "analysis/sal_e2e_20261015/deploy_ckpt/S0_seed20261015.pth"
)
LOCALIZER_PREFIX = "loc_net.net."


def canonical_path(path):
    return Path(path).expanduser().resolve()


def discover_mris(prepared_dir, input_paths=None):
    """Get unique MRI paths from info.json, or use explicit paths if provided."""
    if input_paths:
        paths = sorted({canonical_path(p) for p in input_paths})
    else:
        paths = set()
        for info_path in sorted(Path(prepared_dir).glob("*/info.json")):
            with info_path.open() as f:
                info = json.load(f)
            if info.get("nii_path"):
                paths.add(canonical_path(info["nii_path"]))
        paths = sorted(paths)

    if not paths:
        raise FileNotFoundError(
            f"No MRI paths found in prepared metadata under {prepared_dir}"
        )

    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} MRI path(s) are unavailable; examples: "
            + ", ".join(map(str, missing[:5]))
        )
    return paths


def load_localizer(checkpoint_path, device):
    """Build the S0 localizer and strictly load its pretrained weights."""
    import sys

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "analysis/sal_e2e_20261015"))

    import sal_model
    from models.loc_encoder import soft_argmax_3d

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing pretrained checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    state = checkpoint["network_weights"]
    localizer_state = {
        key[len(LOCALIZER_PREFIX):]: value
        for key, value in state.items()
        if key.startswith(LOCALIZER_PREFIX)
    }
    if not localizer_state:
        raise RuntimeError(
            f"No checkpoint weights found under {LOCALIZER_PREFIX!r}"
        )

    model = sal_model.AnatLocBSal(joint=False, k=0.0)
    # Strict loading fails if the pretrained localizer weights are incomplete.
    model.load_state_dict(localizer_state, strict=True)

    del checkpoint, state, localizer_state
    return model.to(device).eval(), soft_argmax_3d


def infer_and_save(mri_path, model, soft_argmax_3d, device, output_dir):
    """Predict both centers, crop the preprocessed MRI, and save crops plus metadata."""
    import nibabel as nib
    from nibabel.processing import resample_to_output
    from scipy.ndimage import gaussian_filter
    import torch.nn.functional as F

    mri_path = canonical_path(mri_path)
    nii = nib.load(str(mri_path))

    # S0 whole-brain preprocessing: 0.6-mm RTO grid, per-volume z-score,
    # then Gaussian blur with sigma 1 voxel.
    rto = resample_to_output(
        nii, voxel_sizes=(0.6, 0.6, 0.6), order=3, cval=0.0
    )
    volume = rto.get_fdata(dtype=np.float32)
    if volume.ndim == 4:
        volume = volume[..., 0]
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D MRI, got {volume.shape} for {mri_path}")

    volume = (volume - float(volume.mean())) / (float(volume.std()) + 1e-8)
    volume = gaussian_filter(volume, sigma=1.0).astype(np.float32)

    shape = np.asarray(volume.shape, dtype=np.int64)
    if np.any(shape < 64):
        raise ValueError(
            f"MRI is too small for a 64^3 crop: {mri_path}, shape={tuple(shape)}"
        )

    x = torch.from_numpy(np.ascontiguousarray(volume))[None, None].to(device)
    x128 = F.interpolate(
        x, size=(128, 128, 128), mode="trilinear", align_corners=False
    )

    crops = {}
    centers = {}
    starts = {}

    with torch.inference_mode():
        heat_left, heat_right, _ = model(x128)

        # The model's head convention is head 0 = low axis-0 ("left"),
        # head 1 = high axis-0 ("right").
        for side, heatmap in (("left", heat_left), ("right", heat_right)):
            center = torch.round(
                soft_argmax_3d(heatmap)[0]
                * torch.as_tensor(
                    shape - 1, device=device, dtype=torch.float32
                )
            ).to(torch.int64).cpu().numpy()

            # Deployed hard crop: clamp(round(center) - 32, 0, shape - 64).
            start = np.minimum(np.maximum(center - 32, 0), shape - 64)
            slices = tuple(
                slice(int(lo), int(lo + 64)) for lo in start
            )
            crop = np.ascontiguousarray(volume[slices])
            if crop.shape != (64, 64, 64):
                raise RuntimeError(
                    f"Unexpected {side} crop shape {crop.shape} for {mri_path}"
                )

            crops[side] = crop
            centers[side] = center.tolist()
            starts[side] = start.tolist()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path_hash = hashlib.sha256(str(mri_path).encode()).hexdigest()[:10]
    case_id = f"{mri_path.name}_{path_hash}"
    crop_file = f"{case_id}.npz"

    with (output_dir / crop_file).open("wb") as f:
        np.savez_compressed(
            f,
            left_crop=crops["left"],
            right_crop=crops["right"],
        )

    record = {
        "case_id": case_id,
        "mri_path": str(mri_path),
        "crop_file": crop_file,
        "crop_shape": [64, 64, 64],
        "array_axes": ["axis0", "axis1", "axis2"],
        "voxel_spacing_mm": [0.6, 0.6, 0.6],
        "head_0": "left",
        "head_1": "right",
        "center_voxels_rto": centers,
        "crop_start_voxels_rto": starts,
    }
    return record


class SavedCropDataset(Dataset):
    """Load saved crops without importing or running the localization model."""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        index_path = self.output_dir / "index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"Missing saved-crop index: {index_path}")
        with index_path.open() as f:
            self.records = json.load(f)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        crop_path = self.output_dir / record["crop_file"]

        with np.load(crop_path) as saved:
            left = torch.from_numpy(saved["left_crop"].copy()).float()[None]
            right = torch.from_numpy(saved["right_crop"].copy()).float()[None]

        return {
            "left": left,      # (1, 64, 64, 64)
            "right": right,    # (1, 64, 64, 64)
            "case_id": record["case_id"],
            "mri_path": record["mri_path"],
        }


def make_dataloader(output_dir, batch_size=1, shuffle=False, num_workers=0):
    return DataLoader(
        SavedCropDataset(output_dir),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("infer", "load"), required=True)
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to save crops to, or load saved crops from",
    )
    parser.add_argument("--prepared-dir", default=str(DEFAULT_PREPARED))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--input", nargs="*", default=None,
        help="Optional MRI path(s); otherwise discover paths from info.json",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    if args.mode == "infer":
            device = torch.device(args.device)
            model, soft_argmax_3d = load_localizer(args.checkpoint, device)
            mri_paths = discover_mris(args.prepared_dir, args.input)

            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # --- Resume support: load existing index if present ---
            index_path = output_dir / "index.json"
            if index_path.is_file():
                with index_path.open() as f:
                    records = json.load(f)
                done_mris = {r["mri_path"] for r in records}
                print(f"Resuming: {len(records)} already completed")
            else:
                records = []
                done_mris = set()

            failed = []

            for i, mri_path in enumerate(mri_paths):
                # --- Skip already processed (check .npz file on disk) ---
                path_hash = hashlib.sha256(
                    str(canonical_path(mri_path)).encode()
                ).hexdigest()[:10]
                expected_npz = output_dir / f"{Path(mri_path).name}_{path_hash}.npz"
                if expected_npz.is_file():
                    # Rebuild record for index
                    records.append({
                        "case_id": expected_npz.stem,
                        "mri_path": str(canonical_path(mri_path)),
                        "crop_file": expected_npz.name,
                        "crop_shape": [64, 64, 64],
                    })
                    done_mris.add(str(canonical_path(mri_path)))
                    print(f"[SKIP] {expected_npz.name} already exists")
                    continue

                try:
                    record = infer_and_save(
                        mri_path, model, soft_argmax_3d, device, output_dir
                    )
                    records.append(record)
                    done_mris.add(record["mri_path"])
                    print(
                        f"[{len(records)}/{len(mri_paths)}] Saved {record['crop_file']}"
                    )

                    # --- Incremental index save every 10 ---
                    if len(records) % 10 == 0:
                        temp_path = output_dir / "index.json.tmp"
                        temp_path.write_text(json.dumps(records, indent=2))
                        temp_path.replace(index_path)

                except Exception as e:
                    failed.append({"mri": str(mri_path), "error": str(e)})
                    print(f"[FAILED {i+1}/{len(mri_paths)}] {mri_path}: {e}")
                    continue

            # --- Final save ---
            temp_path = output_dir / "index.json.tmp"
            temp_path.write_text(json.dumps(records, indent=2))
            temp_path.replace(index_path)
            print(f"\nDone: {len(records)} succeeded, {len(failed)} failed")
            print(f"Saved index: {index_path}")

            if failed:
                fail_path = output_dir / "failures.json"
                fail_path.write_text(json.dumps(failed, indent=2))
                print(f"Failure log: {fail_path}")

    else:
        # This mode reads only the saved crop files and index; it does not
        # import or initialize the localization model.
        loader = make_dataloader(
            args.output_dir,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        for batch in loader:
            print(
                batch["case_id"],
                "left:", tuple(batch["left"].shape),
                "right:", tuple(batch["right"].shape),
            )


if __name__ == "__main__":
    main()


# /home/jzhou169/miniconda3/envs/tnproject/bin/python /home/vgogoi1/projects/TN_Research/Pipeline/infer_loc_crops.py \
#   --mode infer \
#   --output-dir /home/vgogoi1/projects/TN_Research/Pipeline/crop_data

# or if one input

# /home/jzhou169/miniconda3/envs/tnproject/bin/python /path/to/infer_loc_crops.py \
#   --mode infer \
#   --input /path/to/full_brain.nii.gz \
#   --output-dir /your/chosen/path/loc_crops

# import torch
# from infer_loc_crops import make_dataloader

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# loader = make_dataloader(
#     "/your/chosen/path/crop_data",
#     batch_size=4,
#     shuffle=False,
#     num_workers=0,
# )

# for batch in loader:
#     # Each tensor is (B, 1, 64, 64, 64), ready for a typical 3D model.
#     left = batch["left"].to(device)
#     right = batch["right"].to(device)

#     left_output = your_model(left)
#     right_output = your_model(right)
