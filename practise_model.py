import fiftyone as fo
import fiftyone.zoo as foz
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

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

# Create a list of preprocessed images and labels, filtering out samples without labels
data = [preprocess_data(sample) for sample in dataset if preprocess_data(sample)[1] is not None]

# Debugging: Print the number of samples after filtering
print(f"Number of samples after filtering: {len(data)}")

# Split the data into training and validation sets
train_data = data[:int(0.8 * len(data))]
val_data = data[int(0.8 * len(data)):]

# Define a function to load and preprocess the images
def load_and_preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image

# Create a dataset of preprocessed images and labels for training and validation
train_images = [load_and_preprocess_image(image) for image, label in train_data]
train_labels = [label for image, label in train_data]
val_images = [load_and_preprocess_image(image) for image, label in val_data]
val_labels = [label for image, label in val_data]

# Debugging: Print shapes of the images and labels
print(f"Number of training images: {len(train_images)}, Number of training labels: {len(train_labels)}")
print(f"Number of validation images: {len(val_images)}, Number of validation labels: {len(val_labels)}")

# Convert the lists to TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))

# Batch the datasets
train_dataset = train_dataset.batch(32)
val_dataset = val_dataset.batch(32)

# Debugging: Print shapes of the datasets
for images, labels in train_dataset.take(1):
    print(f"Train images shape: {images.shape}, Train labels shape: {labels.shape}")
for images, labels in val_dataset.take(1):
    print(f"Validation images shape: {images.shape}, Validation labels shape: {labels.shape}")

# Define the MobileNetV2 model with alpha=1.0
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    alpha=1.0,
    include_top=False,
    weights='imagenet'
)

# Add a global average pooling layer and a dense layer for classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(80, activation="softmax")(x)  # Number of classes in the COCO dataset

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)

print("Model training completed successfully.")
