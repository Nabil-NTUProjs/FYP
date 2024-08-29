import fiftyone as fo
import fiftyone.zoo as foz
import cv2
import os

# Load the COCO dataset
dataset = foz.load_zoo_dataset("coco-2017", split="train")

# Filter the dataset to include only images containing people
dataset = dataset.filter_labels("detections", fo.ViewField("label") == "person")

# Himax WE-I image dimensions
target_width = 320
target_height = 320

# Output directory
output_dir = "/"
os.makedirs(output_dir, exist_ok=True)

# Process each sample
for sample in dataset:
    img = cv2.imread(sample.filepath)
    if img is None:
        continue
    
    # Resize and rescale image
    resized_img = cv2.resize(img, (target_width, target_height))
    rescaled_img = resized_img / 255.0  # Normalize to [0, 1]
    
    # Save processed image
    output_path = os.path.join(output_dir, os.path.basename(sample.filepath))
    cv2.imwrite(output_path, (rescaled_img * 255).astype('uint8'))

print("Processing complete.")
