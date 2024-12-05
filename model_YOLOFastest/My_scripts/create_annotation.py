from pycocotools.coco import COCO
import os

# Define paths
data_dir = './coco2017/'
output_file = './coco_person_yolo/annotations.txt'
os.makedirs('./coco_person_yolo/', exist_ok=True)

# Initialize COCO API for instance annotations
coco = COCO(os.path.join(data_dir, 'annotations/instances_train2017.json'))

# Get all images containing the "person" class
person_category_id = coco.getCatIds(catNms=['person'])[0]
person_image_ids = coco.getImgIds(catIds=[person_category_id])

# Convert person annotations to YOLO format
def convert_bbox_to_yolo(bbox, img_width, img_height):
    x_center = (bbox[0] + bbox[2] / 2) / img_width
    y_center = (bbox[1] + bbox[3] / 2) / img_height
    width = bbox[2] / img_width
    height = bbox[3] / img_height
    return [x_center, y_center, width, height]

# Write all annotations to a single file
with open(output_file, 'w') as f_out:
    for img_id in person_image_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_width = img_info['width']
        img_height = img_info['height']
        img_path = os.path.join('./coco_person_yolo/', img_info['file_name'])
        
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[person_category_id])
        anns = coco.loadAnns(ann_ids)

        # For each annotation in the image
        for ann in anns:
            bbox_yolo = convert_bbox_to_yolo(ann['bbox'], img_width, img_height)
            # Write to annotation file in required format
            f_out.write(f"{img_path} 0 {bbox_yolo[0]} {bbox_yolo[1]} {bbox_yolo[2]} {bbox_yolo[3]}\n")
