import cv2
import json
import supervision as sv
from typing import List
import os
from .utils.environment import setup_environment
import matplotlib.pyplot as plt

def detect_objects(image_path: str, annotation_path: str, num_images: int = 8):
    """
    Detect objects in images using PaliGemma model.
    
    Args:
        image_path: Path to the image directory
        annotation_path: Path to the annotation file
        num_images: Number of images to process
    
    Returns:
        detections: Object detections with bounding boxes
    """
    setup_environment()
    
    # Read available lines from annotation file
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
    
    # Adjust num_images if we have fewer images than requested
    num_images = min(num_images, len(lines))
    if num_images == 0:
        raise ValueError("No images found in annotation file")
    
    # Process each line
    images = []
    first = json.loads(lines[0])
    CLASSES = first.get('prefix').replace("detect ", "").split(" ; ")
    
    for line in lines[:num_images]:
        data = json.loads(line)
        image = cv2.imread(os.path.join(image_path, data.get('image')))
        if image is None:
            print(f"Warning: Could not read image {data.get('image')}")
            continue
            
        (h, w, _) = image.shape
        detections = sv.Detections.from_lmm(
            lmm='paligemma',
            result=data.get('suffix'),
            resolution_wh=(w, h),
            classes=CLASSES)
        
        image = sv.BoxAnnotator(thickness=4).annotate(image, detections)
        image = sv.LabelAnnotator(text_scale=2, text_thickness=4).annotate(image, detections)
        images.append(image)
    
    if images:
        if num_images == 1:
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
        else:
            rows = (num_images + 1) // 2
            sv.plot_images_grid(images, (rows, min(2, num_images)))
        return detections
    else:
        raise RuntimeError("No valid images were processed") 