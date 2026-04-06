import yaml
from ultralytics import YOLO

# -------- INPUTS --------
model_path = "tobacco_v0.1.pt"                     # your trained YOLO model
dataset_path = r"C:\dataset\yolo_dataset"  # dataset root folder
train_path = "images/train"
val_path = "images/val"
test_path = "images/test"
# ------------------------

# Load YOLO model
model = YOLO(model_path)

# Extract class names
names = model.names

# Convert dict to list if needed
if isinstance(names, dict):
    class_names = list(names.values())
else:
    class_names = names

# Create yaml structure
data = {
    "path": dataset_path,
    "train": train_path,
    "val": val_path,
    "test": test_path,
    "nc": len(class_names),
    "names": class_names
}

# Save yaml
with open("data.yaml", "w") as f:
    yaml.dump(data, f, sort_keys=False)

print("data.yaml generated successfully!")