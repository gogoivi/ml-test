from tkinter import Tk, filedialog
from pathlib import Path
import re
import traceback
import sys

import nibabel as nib
import numpy as np
import napari


# ============================================================
# CONFIG
# ============================================================

MAX_SAMPLES_TO_OPEN = 5        # Set to None to open all samples

# Which layer roles to actually load into the viewer.
# Comment out any you don't need to save memory.
ROLES_TO_LOAD = [
    "image",
    "ground_truth",
    # "pred_raw",          # Uncomment if you want raw predictions
    "pred_postproc",
]

# If True, use memory-mapped loading (much less RAM).
USE_MEMMAP = True


# ============================================================
# 1. Extract the MRI identifier from a folder or file name
# ============================================================

_MRI_ID_RE = re.compile(r"MRI[_\-]?\d+(?:[_\-]\d+)*", re.IGNORECASE)


def extract_mri_id(text: str) -> str | None:
    m = _MRI_ID_RE.search(text)
    if m:
        return re.sub(r"[\-]", "_", m.group(0)).upper()
    return None


# ============================================================
# 2. Identify layer types from filenames
# ============================================================

def classify_file(file_path):
    stem = Path(file_path).name.lower()

    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    elif stem.endswith(".nii"):
        stem = stem[:-4]

    tokens = re.sub(r"[^a-z0-9]+", " ", stem).split()
    token_set = set(tokens)

    is_prediction = bool(
        token_set & {"pred", "prediction", "predictions"}
    )

    if (
        ("ground" in token_set and "truth" in token_set)
        or "groundtruth" in token_set
        or "gt" in token_set
        or ("annotation" in token_set and not is_prediction)
    ):
        return "ground_truth"

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

    if "raw" in token_set:
        return "pred_raw"

    if is_prediction:
        return "pred_postproc"

    if token_set & {
        "mask", "seg", "segmentation",
        "label", "labels", "annotation", "annotations"
    }:
        return "ground_truth"

    return "image"


# ============================================================
# 3. Sample key fallback
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
# 4. Source label for distinguishing model folders
# ============================================================

def get_source_label(file_path, root_folder):
    try:
        rel = file_path.parent.relative_to(root_folder)
    except ValueError:
        rel = file_path.parent

    parts = rel.parts
    if not parts or parts == (".",):
        return ""

    cleaned = []
    for p in parts:
        if extract_mri_id(p) is not None and p != parts[0]:
            continue
        cleaned.append(p)

    return "/".join(cleaned) if cleaned else ""


# ============================================================
# 5. Load a NIfTI volume
# ============================================================

def load_volume(file_path, is_labels=False):
    """Load a 3D NIfTI volume, optionally memory-mapped."""
    try:
        if USE_MEMMAP:
            # Load with memory mapping — data stays on disk until
            # accessed slice-by-slice. Huge RAM savings.
            proxy = nib.load(str(file_path))
            data = proxy.dataobj

            if is_labels:
                # Labels must be integer. For memmap we need to
                # load into RAM but as int32 (smaller than float64).
                data = np.asarray(data)
                if data.ndim != 3:
                    raise ValueError(
                        f"Expected 3D, got shape {data.shape}"
                    )
                data = np.rint(data).astype(np.int32)
            else:
                # For images, use np.asarray to get a proper array
                # but keep it as float32 to save memory.
                data = np.asarray(data, dtype=np.float32)
                if data.ndim != 3:
                    raise ValueError(
                        f"Expected 3D, got shape {data.shape}"
                    )
        else:
            img = nib.load(str(file_path))
            data = img.get_fdata()

            if data.ndim != 3:
                raise ValueError(
                    f"Expected 3D, got shape {data.shape}"
                )

            if is_labels:
                data = np.rint(data).astype(np.int32)

        return data

    except Exception as exc:
        print(f"  ERROR loading {file_path}: {exc}")
        traceback.print_exc()
        return None


# ============================================================
# 6. Deduplicate identical layers (same MRI, same GT, etc.)
# ============================================================

def deduplicate_layers(layer_entries):
    """
    When multiple model folders each contain identical copies of
    the MRI image or ground truth, keep only one copy of each.

    Predictions are always kept (they differ per model).
    """
    seen_image = False
    seen_gt = False
    result = []

    for entry in layer_entries:
        role = entry["role"]

        if role == "image":
            if seen_image:
                continue
            seen_image = True

        if role == "ground_truth":
            if seen_gt:
                continue
            seen_gt = True

        result.append(entry)

    return result


# ============================================================
# 7. Create one Napari viewer for a merged sample
# ============================================================

def visualize_merged_sample(mri_id, layer_entries):
    title = f"Sample: {mri_id}"

    nice_role = {
        "image": "MRI",
        "ground_truth": "Ground Truth",
        "pred_raw": "Prediction (Raw)",
        "pred_postproc": "Prediction (Post-Processed)",
    }

    # Filter to only the roles we want.
    layer_entries = [
        e for e in layer_entries if e["role"] in ROLES_TO_LOAD
    ]

    if not layer_entries:
        print(f"  No layers to load for {mri_id} (all filtered).")
        return None

    # Deduplicate identical MRI/GT copies from different folders.
    layer_entries = deduplicate_layers(layer_entries)

    # Sort: image first, then GT, then predictions by source.
    order = {
        "image": 0, "ground_truth": 1,
        "pred_raw": 2, "pred_postproc": 3,
    }
    layer_entries.sort(
        key=lambda e: (order.get(e["role"], 99), e["source"])
    )

    total = len(layer_entries)
    print(f"  Will load {total} layer(s) after filtering/dedup.")

    # Load all data first, then open the viewer.
    loaded = []

    for i, entry in enumerate(layer_entries, start=1):
        role = entry["role"]
        file_path = entry["path"]
        source = entry["source"]

        base_name = nice_role.get(role, role)
        if source:
            layer_name = f"{base_name} [{source}]"
        else:
            layer_name = base_name

        print(
            f"  [{i}/{total}] Loading {layer_name}: "
            f"{file_path.name}",
            end="",
            flush=True,
        )

        data = load_volume(file_path, is_labels=(role != "image"))

        if data is not None:
            print(f"  shape={data.shape}  dtype={data.dtype}")
            loaded.append({
                "role": role,
                "name": layer_name,
                "data": data,
            })
        else:
            print("  SKIPPED (load failed)")

    if not loaded:
        print(f"  No layers loaded successfully for {mri_id}.")
        return None

    # Check shapes.
    shapes = {item["name"]: item["data"].shape for item in loaded}
    if len(set(shapes.values())) > 1:
        print(
            f"  WARNING — shapes differ: {shapes}. "
            f"Check alignment."
        )

    # Now open the viewer and add layers.
    print(f"  Opening Napari viewer: {title}")
    sys.stdout.flush()

    try:
        viewer = napari.Viewer(title=title)
    except Exception as exc:
        print(f"  ERROR creating viewer: {exc}")
        traceback.print_exc()
        return None

    for item in loaded:
        role = item["role"]
        try:
            if role == "image":
                viewer.add_image(
                    item["data"],
                    name=item["name"],
                    colormap="gray",
                )
            elif role == "ground_truth":
                viewer.add_labels(
                    item["data"],
                    name=item["name"],
                    opacity=0.7,
                )
            elif role == "pred_raw":
                viewer.add_labels(
                    item["data"],
                    name=item["name"],
                    opacity=0.7,
                    visible=False,
                )
            elif role == "pred_postproc":
                viewer.add_labels(
                    item["data"],
                    name=item["name"],
                    opacity=0.7,
                )
        except Exception as exc:
            print(
                f"  ERROR adding layer '{item['name']}': {exc}"
            )
            traceback.print_exc()

    return viewer


# ============================================================
# 8. Main: select folder and open viewers
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

    # ----------------------------------------------------------
    # Group files by MRI identifier.
    # ----------------------------------------------------------

    merged_groups: dict[str, list[dict]] = {}

    for file_path in nifti_files:
        role = classify_file(file_path)
        source = get_source_label(file_path, folder_path)

        mri_id = extract_mri_id(file_path.name)
        if mri_id is None:
            mri_id = extract_mri_id(file_path.parent.name)

        if mri_id is not None:
            group_key = mri_id
        else:
            sample_key = get_sample_key(file_path)
            relative_parent = file_path.parent.relative_to(
                folder_path
            )
            if relative_parent != Path("."):
                group_key = f"{relative_parent} / {sample_key}"
            else:
                group_key = sample_key

        if group_key not in merged_groups:
            merged_groups[group_key] = []

        already = any(
            e["role"] == role and e["source"] == source
            for e in merged_groups[group_key]
        )

        if already:
            existing = next(
                e for e in merged_groups[group_key]
                if e["role"] == role and e["source"] == source
            )
            print(
                f"WARNING — duplicate '{role}' for "
                f"'{group_key}' from source '{source}'.\n"
                f"  Keeping : {existing['path'].name}\n"
                f"  Ignoring: {file_path.name}"
            )
            continue

        merged_groups[group_key].append({
            "role": role,
            "path": file_path,
            "source": source,
        })

    # ----------------------------------------------------------
    # Report.
    # ----------------------------------------------------------
    print(f"\nIdentified {len(merged_groups)} unique sample(s):")

    for key in sorted(merged_groups):
        entries = merged_groups[key]
        sources = sorted(
            set(e["source"] for e in entries if e["source"])
        )
        roles = sorted(set(e["role"] for e in entries))

        # Count how many will actually load.
        loadable = [
            e for e in entries if e["role"] in ROLES_TO_LOAD
        ]

        source_info = (
            f" from {len(sources)} source(s)"
            if sources else ""
        )
        print(
            f"  {key}: {len(entries)} total, "
            f"{len(loadable)} after role filter "
            f"[{', '.join(roles)}]{source_info}"
        )

    print(f"\nRoles being loaded: {ROLES_TO_LOAD}")
    print(f"Memory-mapped loading: {USE_MEMMAP}")

    # ----------------------------------------------------------
    # Open viewers.
    # ----------------------------------------------------------
    samples_to_process = sorted(merged_groups.keys())

    if MAX_SAMPLES_TO_OPEN is not None:
        samples_to_process = samples_to_process[:MAX_SAMPLES_TO_OPEN]

    viewers = []

    for index, sample_key in enumerate(samples_to_process, start=1):
        layer_entries = merged_groups[sample_key]

        print(f"\n{'=' * 60}")
        print(
            f"Opening sample {index}/{len(samples_to_process)}: "
            f"{sample_key}  "
            f"({len(layer_entries)} total layers)"
        )
        print("=" * 60)

        try:
            viewer = visualize_merged_sample(
                mri_id=sample_key,
                layer_entries=layer_entries,
            )
            if viewer is not None:
                viewers.append(viewer)
        except Exception as exc:
            print(f"ERROR opening sample {sample_key}: {exc}")
            traceback.print_exc()

    # ----------------------------------------------------------
    # Summary.
    # ----------------------------------------------------------
    skipped = len(merged_groups) - len(samples_to_process)

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"Opened {len(viewers)} Napari viewer(s).")

    if skipped > 0:
        print(
            f"Skipped {skipped} additional sample(s) "
            f"(MAX_SAMPLES_TO_OPEN = {MAX_SAMPLES_TO_OPEN})."
        )

    print("\nTips:")
    print("  - Use the scroll wheel to navigate through slices.")
    print("  - Toggle layer visibility using the eye icon.")
    print("  - Prediction (Raw) layers are hidden by default.")
    print("  - Layers from different model folders are tagged")
    print("    with the folder name in square brackets.")
    print("  - Close all viewers to exit.")

    if viewers:
        napari.run()
    else:
        print("\nNo viewers were opened. Check errors above.")


if __name__ == "__main__":
    select_folder_and_load()