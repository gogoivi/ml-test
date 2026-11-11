import argparse
import json
import random
from pathlib import Path

import napari
import numpy as np

HERE = Path(__file__).resolve().parent


def save_result(crop_dir: Path, checkpoint_path: Path, output_dir: Path) -> None:
    import torch
    from model import ConvVAE3D

    records = json.loads((crop_dir / "index.json").read_text())
    if not records:
        raise ValueError(f"No crop records found in {crop_dir / 'index.json'}")

    record = random.choice(records)
    side = random.choice(["left", "right"])

    with np.load(crop_dir / record["crop_file"]) as data:
        original = data[f"{side}_crop"].copy().astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    z_dim = checkpoint.get("config", {}).get("z_dim", 16)

    model = ConvVAE3D(z_dim=z_dim, in_channels=1, input_size=64).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    # Decode the latent mean for a repeatable reconstruction.
    x = torch.from_numpy(original)[None, None].to(device)
    with torch.inference_mode():
        mu, _ = model.encode(x)
        prediction = model.decode(mu)[0, 0].cpu().numpy().astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "sample.npz",
        original=original,
        prediction=prediction,
        difference=np.abs(original - prediction),
    )
    (output_dir / "sample.json").write_text(json.dumps({
        "case_id": record["case_id"],
        "side": side,
        "center_voxels_rto": record.get("center_voxels_rto", {}).get(side),
        "voxel_spacing_mm": record.get("voxel_spacing_mm", [0.6, 0.6, 0.6]),
    }, indent=2))

    print(f"Saved visualization data to {output_dir}")


def open_result(output_dir: Path) -> None:
    with np.load(output_dir / "sample.npz") as data:
        original = data["original"].copy()
        prediction = data["prediction"].copy()
        difference = data["difference"].copy()

    metadata = json.loads((output_dir / "sample.json").read_text())
    spacing = tuple(metadata.get("voxel_spacing_mm", [0.6, 0.6, 0.6]))
    limits = tuple(np.percentile(original, [1, 99]))

    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(original, name="Original", scale=spacing,
                     contrast_limits=limits, rendering="mip")
    viewer.add_image(prediction, name="Prediction (toggle)", scale=spacing,
                     contrast_limits=limits, rendering="mip", visible=False)
    viewer.add_image(difference, name="Absolute difference (toggle)",
                     scale=spacing, rendering="mip", visible=False)

    print(f"{metadata['case_id']} — {metadata['side']}")
    napari.run()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["save", "vis"], required=True)
    parser.add_argument("--crop-dir", type=Path, default=HERE / "crop_data")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=HERE / "vae_output" / "checkpoints" / "best.pt",
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "test_viz")
    args = parser.parse_args()

    if args.mode == "save":
        save_result(args.crop_dir, args.checkpoint, args.output_dir)
    else:
        open_result(args.output_dir)


if __name__ == "__main__":
    main()

# server:  python test.py --mode save --checkpoint vae_output/checkpoints/best.pt
# computer: python test.py --mode vis --output-dir test_viz