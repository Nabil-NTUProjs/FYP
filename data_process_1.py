import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.image as foui
import os
from PIL import Image
import numpy as np

# Define the save directory
save_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the current script
dataset_dir = os.path.join(save_dir, "dataset")  # Folder to save the dataset

# Create dataset directory is does not exists
os.makedirs(dataset_dir, exist_ok=True)

# Load a small sample of the dataset to inspect
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["detections"],  # Ensure you're using the correct label type
    max_samples=500  # Load a small number of samples for inspection
)

# Print the available fields in the dataset
print("Dataset fields:", dataset.get_field_schema())

# Print the available classes and label types
print("Available classes:", dataset.default_classes)

# Debugging: Print a sample to inspect its structure
print("Sample structure:", dataset.first())

# Count labels in the original dataset
try:
    label_counts = dataset.count_values("ground_truth.detections.label")
    print("Original dataset label counts:", label_counts)
except ValueError as e:
    print("Error counting labels:", e)

# Filter positive samples (with "person" class)
positive_samples = []
for sample in dataset:
    if sample.ground_truth is not None and sample.ground_truth.detections:
        if any(d.label == "person" for d in sample.ground_truth.detections):
            positive_samples.append(sample)

# Filter negative samples (without "person" class)
negative_samples = []
for sample in dataset:
    if sample.ground_truth is not None and sample.ground_truth.detections:
        if all(d.label != "person" for d in sample.ground_truth.detections):
            negative_samples.append(sample)

# Remove other labels from positive samples
for sample in positive_samples:
    sample["ground_truth"].detections = [d for d in sample["ground_truth"].detections if d.label == "person"]

# Ensure negative samples have no labels
for sample in negative_samples:
    sample["ground_truth"].detections = []

# Combine positive and negative samples
combined_samples = positive_samples + negative_samples

# Create a new dataset with combined samples
combined_dataset = fo.Dataset()
for sample in combined_samples:
    combined_dataset.add_sample(sample)

# Count labels in the newly labelled dataset
try:
    label_counts = combined_dataset.count_values("ground_truth.detections.label")
    print("Newly labelled dataset label counts:", label_counts)
except ValueError as e:
    print("Error counting labels:", e)

# Split the dataset into training and testing sets
train_dataset = combined_dataset.take(400)  # 80% of 500 samples
test_dataset = combined_dataset.exclude([s.id for s in train_dataset])  # Remaining 20%

# Perform necessary preprocessing steps
def preprocess_sample(sample):
    # Load image
    img = Image.open(sample["filepath"])
    # Resize image
    img = img.resize((320, 320))
    # Normalize pixel values
    img = np.array(img) / 255.0
    # Save the processed image back to the same path
    img = Image.fromarray((img * 255).astype(np.uint8))
    img.save(sample["filepath"])
    return sample

# Apply preprocessing to each sample in the training and testing views
for sample in train_dataset:
    preprocess_sample(sample)
for sample in test_dataset:
    preprocess_sample(sample)

# Save the processed datasets
train_dataset.save()
test_dataset.save()

# Print the number of samples in each split
print("Number of training samples:", len(train_dataset))
print("Number of testing samples:", len(test_dataset))

# Export the datasets to a format suitable for HIMAX training
train_export_dir = os.path.join(dataset_dir, "train_export")
test_export_dir = os.path.join(dataset_dir, "test_export")

train_dataset.export(export_dir=train_export_dir, dataset_type=fo.types.COCODetectionDataset, label_field="ground_truth")
test_dataset.export(export_dir=test_export_dir, dataset_type=fo.types.COCODetectionDataset, label_field="ground_truth")

print("Training and testing datasets exported successfully.")
