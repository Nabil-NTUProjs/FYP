import os
import requests
import shutil
import zipfile

# Define the COCO dataset URLs
coco_images_url = 'http://images.cocodataset.org/zips/train2017.zip'
coco_val_images_url = 'http://images.cocodataset.org/zips/val2017.zip'
coco_annotations_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'

# Define download paths
data_dir = './coco2017/'
images_zip_path = os.path.join(data_dir, 'train2017.zip')
val_images_zip_path = os.path.join(data_dir, 'val2017.zip')
annotations_zip_path = os.path.join(data_dir, 'annotations_trainval2017.zip')

# Create directory to store dataset
os.makedirs(data_dir, exist_ok=True)

# Download training images
if not os.path.exists(os.path.join(data_dir, 'train2017')):
    print("Downloading COCO train images...")
    r = requests.get(coco_images_url, stream=True)
    with open(images_zip_path, 'wb') as f:
        shutil.copyfileobj(r.raw, f)
    
    # Unzip images
    with zipfile.ZipFile(images_zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print("COCO train images downloaded and extracted.")

# Download validation images
if not os.path.exists(os.path.join(data_dir, 'val2017')):
    print("Downloading COCO validation images...")
    r = requests.get(coco_val_images_url, stream=True)
    with open(val_images_zip_path, 'wb') as f:
        shutil.copyfileobj(r.raw, f)
    
    # Unzip images
    with zipfile.ZipFile(val_images_zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print("COCO validation images downloaded and extracted.")

# Download annotations
if not os.path.exists(os.path.join(data_dir, 'annotations')):
    print("Downloading COCO annotations...")
    r = requests.get(coco_annotations_url, stream=True)
    with open(annotations_zip_path, 'wb') as f:
        shutil.copyfileobj(r.raw, f)
    
    # Unzip annotations
    with zipfile.ZipFile(annotations_zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print("COCO annotations downloaded and extracted.")
