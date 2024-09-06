import os
import json
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Define paths
dataset_dir = r"C:\Users\xxnab\OneDrive\Documents\GitHub\FYP\dataset"
train_dir = os.path.join(dataset_dir, "train_export", "data")
test_dir = os.path.join(dataset_dir, "test_export", "data")
train_labels_path = os.path.join(dataset_dir, "train_export", "labels.json")
test_labels_path = os.path.join(dataset_dir, "test_export", "labels.json")

# Parameters
img_size = (224, 224)  # Changed to 224x224
batch_size = 32
epochs = 10

# Load labels
def load_labels(labels_path):
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    # Filter out metadata
    labels = {k: v for k, v in labels.items() if isinstance(v, int)}
    return labels

train_labels = load_labels(train_labels_path)
test_labels = load_labels(test_labels_path)

# Create a dataset from the images and labels
def create_dataset(data_dir, labels, img_size, batch_size):
    def load_image(filename, label):
        img = load_img(os.path.join(data_dir, filename), target_size=img_size)
        img = img_to_array(img) / 255.0
        img = tf.ensure_shape(img, (*img_size, 3))
        label = tf.ensure_shape(label, [])
        return img, label

    filenames = list(labels.keys())
    labels = list(labels.values())
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(lambda x, y: tf.py_function(load_image, [x, y], [tf.float32, tf.int32]))
    dataset = dataset.map(lambda x, y: (x, tf.cast(y, tf.int32)))  # Ensure the return value is a tuple
    dataset = dataset.batch(batch_size)
    return dataset

train_dataset = create_dataset(train_dir, train_labels, img_size, batch_size)
test_dataset = create_dataset(test_dir, test_labels, img_size, batch_size)

# Debugging: Check the shape of the first batch
for images, labels in train_dataset.take(1):
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)

# Load the pretrained model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*img_size, 3))

# Freeze the base model
base_model.trainable = False

# Add custom layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(set(train_labels.values())), activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=test_dataset
)

# Save the model
model.save(os.path.join(dataset_dir, "trained_model.h5"))

print("Model trained and saved successfully.")
