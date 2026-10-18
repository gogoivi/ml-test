import os
import json
import random
import shutil
from pathlib import Path

# --------------------------------------------------
# Configuration
# --------------------------------------------------

# Uses your nnUNet_raw environment variable if set.
raw_root = Path(
    os.environ.get(
        "nnUNet_raw",
        r"C:\Users\gogoi\Desktop\ml-test\nnUNet_raw"
    )
)

source = raw_root / "Dataset001_MRI"

# Choose a new dataset ID that is not already in use.
dataset_id = 2
dataset_name = "MRI_82_18"

output_dir = raw_root / f"Dataset{dataset_id:03d}_{dataset_name}"

# Store held-out ground-truth labels OUTSIDE the nnUNet dataset.
test_labels_dir = (
    source.parent.parent / "heldout_labels_82_18"
)

train_fraction = 0.82
random_seed = 42

# --------------------------------------------------
# Validate paths and protect existing data
# --------------------------------------------------

images_dir = source / "imagesTr"
labels_dir = source / "labelsTr"
dataset_json_path = source / "dataset.json"

if not images_dir.is_dir() or not labels_dir.is_dir():
    raise FileNotFoundError(
        f"Expected imagesTr and labelsTr inside {source}"
    )

if not dataset_json_path.is_file():
    raise FileNotFoundError(
        f"Missing dataset.json in {source}"
    )

if output_dir.exists():
    raise FileExistsError(
        f"Output already exists: {output_dir}\n"
        "Choose another unused dataset ID or inspect "
        "the existing folder before proceeding."
    )

if test_labels_dir.exists():
    raise FileExistsError(
        f"Test-label folder already exists: {test_labels_dir}\n"
        "Choose a new folder name or inspect it first."
    )

with open(dataset_json_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

file_ending = metadata.get("file_ending", ".nii.gz")

# --------------------------------------------------
# Identify cases by their ground-truth label files
# --------------------------------------------------

label_files = sorted(labels_dir.glob(f"*{file_ending}"))

if not label_files:
    raise RuntimeError(f"No labels found in {labels_dir}")

case_ids = [
    p.name[:-len(file_ending)]
    for p in label_files
]

# Verify that every case has at least one image channel.
case_images = {}

for case_id in case_ids:
    images = sorted(
        images_dir.glob(f"{case_id}_*{file_ending}")
    )

    if not images:
        raise FileNotFoundError(
            f"No image channels found for case {case_id}"
        )

    case_images[case_id] = images

# --------------------------------------------------
# Random 82/18 split, reproducible with the same seed
# --------------------------------------------------

random.Random(random_seed).shuffle(case_ids)

n_train = round(train_fraction * len(case_ids))

train_cases = sorted(case_ids[:n_train])
test_cases = sorted(case_ids[n_train:])

print(f"Total cases: {len(case_ids)}")
print(f"Training cases: {len(train_cases)}")
print(f"Held-out test cases: {len(test_cases)}")

# --------------------------------------------------
# Create the new nnU-Net dataset structure
# --------------------------------------------------

new_images_tr = output_dir / "imagesTr"
new_labels_tr = output_dir / "labelsTr"
new_images_ts = output_dir / "imagesTs"

for folder in (
    new_images_tr,
    new_labels_tr,
    new_images_ts,
    test_labels_dir,
):
    folder.mkdir(parents=True, exist_ok=True)

# Training images and labels
for case_id in train_cases:
    for image_path in case_images[case_id]:
        shutil.copy2(image_path, new_images_tr / image_path.name)

    label_path = labels_dir / f"{case_id}{file_ending}"
    shutil.copy2(label_path, new_labels_tr / label_path.name)

# Held-out test images; their labels are stored separately.
for case_id in test_cases:
    for image_path in case_images[case_id]:
        shutil.copy2(image_path, new_images_ts / image_path.name)

    label_path = labels_dir / f"{case_id}{file_ending}"
    shutil.copy2(label_path, test_labels_dir / label_path.name)

# Update dataset metadata for the new training subset.
metadata["numTraining"] = len(train_cases)

with open(output_dir / "dataset.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)

# Save the exact split for reproducibility.
split_info = {
    "seed": random_seed,
    "train_fraction": train_fraction,
    "training_cases": train_cases,
    "test_cases": test_cases,
}

with open(
    source.parent.parent / "split_82_18_manifest.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(split_info, f, indent=2)

print("\nSplit completed successfully.")
print(f"New dataset: {output_dir}")
print(f"Test labels: {test_labels_dir}")
print(f"Split manifest: {source.parent.parent / 'split_82_18_manifest.json'}")

"""preprocess: nnUNetv2_plan_and_preprocess -d 2 --verify_dataset_integrity
train with training set: nnUNetv2_train 2 3d_fullres all -tr nnUNetTrainer_50epochs
predict: nnUNetv2_predict `
  -i "C:\Users\gogoi\Desktop\ml-test\nnUNet_raw\Dataset002_MRI_82_18\imagesTs" `
  -o "C:\Users\gogoi\Desktop\ml-test\predictions_82_18" `
  -d 2 `
  -c 3d_fullres `
  -f all `
  -tr nnUNetTrainer_50epochs
Eval: nnUNetv2_evaluate_folder `
  "C:\Users\gogoi\Desktop\ml-test\heldout_labels_82_18" `
  "C:\Users\gogoi\Desktop\ml-test\predictions_82_18" `
  -djfile "C:\Users\gogoi\Desktop\ml-test\nnUNet_raw\Dataset002_MRI_82_18\dataset.json" `
  -pfile "PATH_TO_TRAINED_MODEL\plans.json"
  """

