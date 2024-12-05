from pycocotools.coco import COCO
import os
import shutil

# Define paths
data_dir = './coco2017/'
output_dir = './coco_person_yolo/'
os.makedirs(output_dir, exist_ok=True)

# Initialize COCO API for instance annotations
coco = COCO(os.path.join(data_dir, 'annotations/instances_train2017.json'))

# Get all images containing the "person" class
person_category_id = coco.getCatIds(catNms=['person'])[0]
person_image_ids = coco.getImgIds(catIds=[person_category_id])

# Copy person images to a new directory
for img_id in person_image_ids:
    img_info = coco.loadImgs(img_id)[0]
    shutil.copyfile(os.path.join(data_dir, 'train2017', img_info['file_name']),
                    os.path.join(output_dir, img_info['file_name']))

# Convert person annotations to YOLO format
def convert_bbox_to_yolo(bbox, img_width, img_height):
    x_center = (bbox[0] + bbox[2] / 2) / img_width
    y_center = (bbox[1] + bbox[3] / 2) / img_height
    width = bbox[2] / img_width
    height = bbox[3] / img_height
    return [x_center, y_center, width, height]

# Write YOLO annotations for person images
for img_id in person_image_ids:
    img_info = coco.loadImgs(img_id)[0]
    img_width = img_info['width']
    img_height = img_info['height']
    
    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[person_category_id])
    anns = coco.loadAnns(ann_ids)
    
    with open(os.path.join(output_dir, f"{img_info['file_name'].replace('.jpg', '.txt')}"), 'w') as f:
        for ann in anns:
            bbox_yolo = convert_bbox_to_yolo(ann['bbox'], img_width, img_height)
            f.write(f"0 {bbox_yolo[0]} {bbox_yolo[1]} {bbox_yolo[2]} {bbox_yolo[3]}\n")
