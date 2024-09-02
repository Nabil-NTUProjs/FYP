import fiftyone as fo
import fiftyone.zoo as foz

# Load 200 samples from the COCO-2017 dataset using FiftyOne
dataset = foz.load_zoo_dataset("coco-2017", split="train", max_samples=200)

# Print the first few samples to inspect the dataset
print("Dataset samples:")
for sample in dataset[:5]:
    print(sample)

# Visualize a sample from the dataset
sample = dataset.first()
sample.view()
