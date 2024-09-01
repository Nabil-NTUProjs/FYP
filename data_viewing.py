import fiftyone as fo
import os

# Define the dataset directories
train_data_dir = "C:/Users/xxnab/OneDrive/Documents/GitHub/FYP/dataset/train_export/data"
train_labels_path = "C:/Users/xxnab/OneDrive/Documents/GitHub/FYP/dataset/train_export/labels.json"
test_data_dir = "C:/Users/xxnab/OneDrive/Documents/GitHub/FYP/dataset/test_export/data"
test_labels_path = "C:/Users/xxnab/OneDrive/Documents/GitHub/FYP/dataset/test_export/labels.json"

# Load the training dataset from the directory
train_dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=train_data_dir,
    labels_path=train_labels_path
)

# Load the testing dataset from the directory
test_dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=test_data_dir,
    labels_path=test_labels_path
)

# Combine training and testing datasets
combined_dataset = train_dataset.clone()
combined_dataset.merge_samples(test_dataset)

# Print the available fields in the dataset
print("Dataset fields:", combined_dataset.get_field_schema())

# Print the available classes and label types
print("Available classes:", combined_dataset.default_classes)

# Debugging: Print a sample to inspect its structure
print("Sample structure:", combined_dataset.first())

# Count labels in the dataset
try:
    label_counts = combined_dataset.count_values("detections.detections.label")
    print("Dataset label counts:", label_counts)
except ValueError as e:
    print("Error counting labels:", e)

# Filter positive samples (with "person" class)
positive_samples = []
for sample in combined_dataset:
    if sample.detections is not None and sample.detections.detections:
        if any(d.label == "person" for d in sample.detections.detections):
            positive_samples.append(sample)

# Filter negative samples (without "person" class)
negative_samples = []
for sample in combined_dataset:
    if sample.detections is not None and sample.detections.detections:
        if all(d.label != "person" for d in sample.detections.detections):
            negative_samples.append(sample)

# Print the number of positive and negative samples
print("Number of positive samples:", len(positive_samples))
print("Number of negative samples:", len(negative_samples))

