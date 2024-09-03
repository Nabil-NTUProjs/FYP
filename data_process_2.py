import os
import fiftyone as fo
import fiftyone.zoo as foz
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split

# Parameters
img_size = (224, 224)
batch_size = 32  # Adjusted for smaller dataset

# Load the COCO dataset using FiftyOne
dataset = foz.load_zoo_dataset("coco-2017", split="train")

# Get image paths and labels
image_paths = [sample.filepath for sample in dataset]
labels = [
    sample.ground_truth.detections[0].label if sample.ground_truth and sample.ground_truth.detections else "background"
    for sample in dataset
]

# Encode labels as integers
label_to_index = {label: idx for idx, label in enumerate(set(labels))}
labels = [label_to_index[label] for label in labels]

# Select only 200 samples
selected_indices = list(range(200))
selected_image_paths = [image_paths[i] for i in selected_indices]
selected_labels = [labels[i] for i in selected_indices]

# Split the dataset
train_paths, val_paths, train_labels, val_labels = train_test_split(
    selected_image_paths, selected_labels, test_size=0.2, random_state=42
)

# Function to load and preprocess images
def load_image(image_path, label):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img = img_to_array(img) / 255.0
    return img, label

# Create TensorFlow datasets
def create_tf_dataset(image_paths, labels):
    def generator():
        for image_path, label in zip(image_paths, labels):
            img, lbl = load_image(image_path, label)
            yield img, lbl

    dataset = tf.data.Dataset.from_generator(generator, output_signature=(
        tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    ))
    dataset = dataset.batch(batch_size).shuffle(buffer_size=len(image_paths))
    return dataset

train_dataset = create_tf_dataset(train_paths, train_labels)
val_dataset = create_tf_dataset(val_paths, val_labels)

# Debugging: Check the shape of the first batch
for images, labels in train_dataset.take(1):
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)

# Now you can use train_dataset and val_dataset for training your Keras model
