# Test code
from load_data import get_dataloaders
train_loader, test_loader = get_dataloaders(r"C:\Users\gogoi\Desktop\ml-test\2023 Patients", batch_size=4)

for X, y, indices in train_loader:
    print(f"Batch shapes: X={X.shape}, y={y.shape}")
    print(f"Sample indices: {indices}")
    print(f"Indices type: {type(indices)}, dtype: {indices.dtype}")
    break