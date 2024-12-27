import torch
import numpy as np
from PIL import Image
import requests
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
import cv2
import json
import supervision as sv
from typing import List
import os
import sys

def read_n_lines(file_path: str, n: int) -> List[str]:
    with open(file_path, 'r') as file:
        lines = [next(file).strip() for _ in range(n)]
    return lines

def visualize_detections():
    images = []
    lines = read_n_lines(f"/root/model/p-1.v1i.paligemma/dataset/_annotations.train1.jsonl", 8)
    first = json.loads(lines[0])

    CLASSES = first.get('prefix').replace("detect ", "").split(" ; ")

    for line in lines:
        data = json.loads(line)
        image = cv2.imread(f"/root/model/p-1.v1i.paligemma/dataset/{data.get('image')}")
        (h, w, _) = image.shape
        detections = sv.Detections.from_lmm(
            lmm='paligemma',
            result=data.get('suffix'),
            resolution_wh=(w, h),
            classes=CLASSES)
        
        image = sv.BoundingBoxAnnotator(thickness=4).annotate(image, detections)
        image = sv.LabelAnnotator(text_scale=2, text_thickness=4).annotate(image, detections)
        images.append(image)

    sv.plot_images_grid(images, (4, 2))
    return detections

def setup_environment():
    # Get current working directory
    current_dir = os.getcwd()
    
    # Build big_vision_repo path
    big_vision_path = os.path.join(current_dir, "big_vision_repo")
    
    # Add big_vision_repo to PYTHONPATH
    os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + os.pathsep + big_vision_path
    
    # Ensure Python knows big_vision_repo location
    sys.path.append(big_vision_path)
    
    # Clone big_vision repository if not exists
    if not os.path.exists("big_vision_repo"):
        os.system("git clone --quiet --branch=main --depth=1 https://github.com/google-research/big_vision big_vision_repo")

def main():
    # Setup environment
    setup_environment()
    
    # Visualize detections
    detections = visualize_detections()
    print(dir(detections))

if __name__ == "__main__":
    main() 