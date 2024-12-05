import os
import random

# Define your image directory (change this to match your actual path)
image_dir = './coco_person_yolo/'

# Get a list of all images in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

# Shuffle the images (optional, for random splitting)
random.shuffle(image_files)

# Define the split ratio (80% for training, 20% for validation)
train_ratio = 0.8
split_idx = int(len(image_files) * train_ratio)

# Split the dataset into training and validation sets
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

# Write the training file paths to train.txt
with open('train.txt', 'w') as f:
    for img in train_files:
        f.write(os.path.join(image_dir, img) + '\n')

# Write the validation file paths to val.txt
with open('val.txt', 'w') as f:
    for img in val_files:
        f.write(os.path.join(image_dir, img) + '\n')

print("train.txt and val.txt have been created successfully!")
