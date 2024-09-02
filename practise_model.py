import fiftyone.zoo as foz
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt

# Load 200 samples from the COCO-2017 dataset using FiftyOne
dataset = foz.load_zoo_dataset("coco-2017", split="train", max_samples=200)

# Define a function to preprocess the images and labels
def preprocess_data(sample):
    image = sample.filepath
    label = None
    if hasattr(sample, 'detections') and sample.detections is not None:
        if len(sample.detections.detections) > 0:
            label = sample.detections.detections[0].label
    return image, label

# Print details of the samples before filtering
print("Sample details before filtering:")
for sample in dataset:
    image, label = preprocess_data(sample)
    print(f"Image: {image}, Label: {label}")

# Create a list of preprocessed images and labels, filtering out samples without labels
data = [preprocess_data(sample) for sample in dataset if preprocess_data(sample)[1] is not None]

# Debugging: Print the number of samples after filtering
print(f"Number of samples after filtering: {len(data)}")

# Split the data into training and validation sets
train_data = data[:int(0.8 * len(data))]
val_data = data[int(0.8 * len(data)):]

# Encode the labels
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform([label for image, label in train_data])
val_labels = label_encoder.transform([label for image, label in val_data])

# Define a function to load and preprocess the images
def load_and_preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image

# Create a dataset of preprocessed images and labels for training and validation
train_images = [load_and_preprocess_image(image) for image, label in train_data]
val_images = [load_and_preprocess_image(image) for image, label in val_data]

# Display a few preprocessed images and their labels
def display_samples(images, labels, label_encoder, num_samples=5):
    plt.figure(figsize=(10, 10))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(label_encoder.inverse_transform([labels[i]])[0])
        plt.axis("off")
    plt.show()

# Display training samples
if len(train_images) > 0:
    display_samples(train_images, train_labels, label_encoder)
else:
    print("No training samples to display.")

# Display validation samples
if len(val_images) > 0:
    display_samples(val_images, val_labels, label_encoder)
else:
    print("No validation samples to display.")
