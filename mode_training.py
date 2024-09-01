import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths
dataset_dir = r"C:\Users\xxnab\OneDrive\Documents\GitHub\FYP\dataset"
train_dir = os.path.join(dataset_dir, "train_export", "data")
test_dir = os.path.join(dataset_dir, "test_export", "data")

# Parameters
img_size = (320, 320)
batch_size = 32
epochs = 10

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, zoom_range=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Load the pretrained model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*img_size, 3))

# Freeze the base model
base_model.trainable = False

# Add custom layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator
)

# Save the model
model.save(os.path.join(dataset_dir, "trained_model.h5"))

print("Model trained and saved successfully.")

