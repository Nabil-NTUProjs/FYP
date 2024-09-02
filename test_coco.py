import fiftyone.zoo as foz
from PIL import Image

# Load 200 samples from the COCO-2017 dataset using FiftyOne
dataset = foz.load_zoo_dataset("coco-2017", split="train", max_samples=200)

# Print the first few samples to inspect the dataset
print("Dataset samples:")
for sample in dataset[:5]:
    print(sample)

# Display the first image using PIL
first_sample = dataset.first()
image_path = first_sample.filepath
image = Image.open(image_path)
image.show()

# Print the label of the first sample
if hasattr(first_sample, 'detections') and first_sample.detections is not None:
    if len(first_sample.detections.detections) > 0:
        label = first_sample.detections.detections[0].label
        print(f"Label: {label}")
