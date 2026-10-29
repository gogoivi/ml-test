from tkinter import Tk, filedialog
from pathlib import Path
import re

import nibabel as nib
import numpy as np
import napari


# ============================================================
# 1. Identify layer types from filenames
# ============================================================

def classify_file(file_path):
    """
    Classify a NIfTI file as:
        image
        ground_truth
        pred_raw
        pred_postproc

    Examples recognized:
        sample_001_MRI.nii.gz
        sample_001_ground_truth.nii.gz
        sample_001_pred_raw.nii.gz
        sample_001_pred_postproc.nii.gz
    """
    stem = Path(file_path).name.lower()

    # Remove NIfTI extensions.
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    elif stem.endswith(".nii"):
        stem = stem[:-4]

    # Convert separators to spaces so names such as pred_raw
    # and ground-truth are handled consistently.
    tokens = re.sub(r"[^a-z0-9]+", " ", stem).split()
    token_set = set(tokens)

    is_prediction = bool(
        token_set & {"pred", "prediction", "predictions"}
    )

    # Explicit ground-truth indicators.
    if (
        ("ground" in token_set and "truth" in token_set)
        or "groundtruth" in token_set
        or "gt" in token_set
        or ("annotation" in token_set and not is_prediction)
    ):
        return "ground_truth"

    # Post-processed prediction indicators.
    if (
        "postproc" in token_set
        or "postprocessed" in token_set
        or "postprocess" in token_set
        or ("post" in token_set and "proc" in token_set)
        or "processed" in token_set
        or "final" in token_set
        or "pp" in token_set
    ):
        return "pred_postproc"

    # Raw prediction.
    if "raw" in token_set:
        return "pred_raw"

    # If a prediction is named simply "prediction", classify it
    # as post-processed by default. Adjust this if your naming
    # convention uses that name for the raw prediction instead.
    if is_prediction:
        return "pred_postproc"

    # Other common ground-truth mask naming conventions.
    if token_set & {
        "mask", "seg", "segmentation",
        "label", "labels", "annotation", "annotations"
    }:
        return "ground_truth"

    # Files not identified as masks/predictions are treated as MRI.
    return "image"


# ============================================================
# 2. Determine which files belong to the same sample
# ============================================================

ROLE_WORDS = {
    "mri", "image", "img", "scan", "volume",
    "ground", "truth", "groundtruth", "gt",
    "mask", "seg", "segmentation",
    "label", "labels", "annotation", "annotations",
    "pred", "prediction", "predictions",
    "raw", "post", "proc", "postproc",
    "postprocessed", "postprocess", "processed",
    "final", "pp",
}


def get_sample_key(file_path):
    """
    Remove layer-specific words from the filename to identify
    the sample shared by its MRI, ground truth, and predictions.

    For example:
        patient_001_MRI.nii.gz           -> patient 001
        patient_001_ground_truth.nii.gz  -> patient 001
        patient_001_pred_raw.nii.gz      -> patient 001
    """
    name = Path(file_path).name.lower()

    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]

    tokens = re.sub(r"[^a-z0-9]+", " ", name).split()
    sample_tokens = [token for token in tokens
                     if token not in ROLE_WORDS]

    return " ".join(sample_tokens) or "sample"


# ============================================================
# 3. Load a NIfTI volume
# ============================================================

def load_volume(file_path, is_labels=False):
    """Load a 3D NIfTI volume, converting label values to integers."""
    img = nib.load(str(file_path))
    data = img.get_fdata()

    if data.ndim != 3:
        raise ValueError(
            f"Expected a 3D volume, but got shape {data.shape}"
        )

    if is_labels:
        # Avoid float-valued label IDs from nibabel's get_fdata().
        data = np.rint(data).astype(np.int32)

    return data


# ============================================================
# 4. Create one Napari viewer for each sample
# ============================================================

def visualize_single_sample(sample_name, sample_files):
    """
    Create one separate Napari viewer for a sample.

    Layers:
        1. MRI: grayscale, visible
        2. Ground Truth: opacity 0.7, visible
        3. Prediction (Raw): opacity 0.7, hidden
        4. Prediction (Post-Processed): opacity 0.7, visible

    sample_files maps layer types to their file paths.
    """
    title = f"Sample: {sample_name}"
    viewer = napari.Viewer(title=title)

    layer_names = {
        "image": "MRI",
        "ground_truth": "Ground Truth",
        "pred_raw": "Prediction (Raw)",
        "pred_postproc": "Prediction (Post-Processed)",
    }

    expected_roles = [
        "image",
        "ground_truth",
        "pred_raw",
        "pred_postproc",
    ]

    missing = [
        layer_names[role]
        for role in expected_roles
        if role not in sample_files
    ]

    if missing:
        print(
            f"WARNING — {sample_name}: missing "
            f"{', '.join(missing)}"
        )

    loaded_arrays = {}

    for role in expected_roles:
        if role not in sample_files:
            continue

        file_path = sample_files[role]
        print(f"  Loading {layer_names[role]}: {file_path.name}")

        try:
            loaded_arrays[role] = load_volume(
                file_path,
                is_labels=(role != "image")
            )
        except Exception as exc:
            print(f"  ERROR loading {file_path.name}: {exc}")

    # Warn if the layers do not occupy matching array grids.
    if loaded_arrays:
        shapes = {
            role: array.shape
            for role, array in loaded_arrays.items()
        }

        if len(set(shapes.values())) > 1:
            print(
                f"WARNING — {sample_name}: layer shapes differ: "
                f"{shapes}. Check that the files are aligned."
            )

    # Add layers in the same order as the reference visualization.
    if "image" in loaded_arrays:
        viewer.add_image(
            loaded_arrays["image"],
            name="MRI",
            colormap="gray",
        )

    if "ground_truth" in loaded_arrays:
        viewer.add_labels(
            loaded_arrays["ground_truth"],
            name="Ground Truth",
            opacity=0.7,
        )

    if "pred_raw" in loaded_arrays:
        viewer.add_labels(
            loaded_arrays["pred_raw"],
            name="Prediction (Raw)",
            opacity=0.7,
            visible=False,
        )

    if "pred_postproc" in loaded_arrays:
        viewer.add_labels(
            loaded_arrays["pred_postproc"],
            name="Prediction (Post-Processed)",
            opacity=0.7,
        )

    return viewer


# ============================================================
# 5. Select a folder and open all sample viewers
# ============================================================

def select_folder_and_load():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    try:
        print("Select the folder containing your NIfTI files.")
        folder_selected = filedialog.askdirectory(
            title="Select Folder Containing NIfTI Files"
        )
    finally:
        root.destroy()

    if not folder_selected:
        print("No folder selected. Exiting.")
        return

    folder_path = Path(folder_selected)

    # Search this folder and all its subfolders.
    nifti_files = sorted(
        path for path in folder_path.rglob("*")
        if path.is_file()
        and (
            path.name.lower().endswith(".nii")
            or path.name.lower().endswith(".nii.gz")
        )
    )

    if not nifti_files:
        print(f"No NIfTI files found in: {folder_path}")
        return

    print(f"\nFound {len(nifti_files)} NIfTI files.")

    # Group files by sample.
    groups = {}

    for file_path in nifti_files:
        role = classify_file(file_path)
        sample_key = get_sample_key(file_path)

        # When files are in separate sample subfolders, keep those
        # samples separate even if their filenames are identical.
        relative_parent = file_path.parent.relative_to(folder_path)

        if relative_parent != Path("."):
            group_key = f"{relative_parent} / {sample_key}"
        else:
            group_key = sample_key

        if group_key not in groups:
            groups[group_key] = {}

        if role in groups[group_key]:
            print(
                f"WARNING — multiple files classified as '{role}' "
                f"for sample '{group_key}'.\n"
                f"Keeping: {groups[group_key][role].name}\n"
                f"Ignoring: {file_path.name}"
            )
            continue

        groups[group_key][role] = file_path

    # One independent viewer for every sample.
    viewers = []

    for index, (sample_name, sample_files) in enumerate(
        sorted(groups.items()), start=1
    ):
        print(f"\n{'=' * 60}")
        print(f"Opening sample {index}/{len(groups)}: {sample_name}")
        print("=" * 60)

        viewer = visualize_single_sample(
            sample_name=sample_name,
            sample_files=sample_files,
        )

        viewers.append(viewer)

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"Opened {len(viewers)} Napari viewers.")
    print("\nTips:")
    print("  - Use the scroll wheel to navigate through slices.")
    print("  - Toggle layer visibility using the eye icon.")
    print("  - Prediction (Raw) is hidden by default.")
    print("  - Ground Truth and Prediction (Post-Processed) are visible.")
    print("  - Close all viewers to exit.")

    napari.run()


if __name__ == "__main__":
    select_folder_and_load()