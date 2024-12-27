import os
from paligemma_multitask.object_detection import detect_objects

def main():
    # Use correct dataset path
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    IMAGE_PATH = os.path.join(BASE_DIR, "dataset")
    ANNOTATION_PATH = os.path.join(BASE_DIR, "dataset", "p-1.v1i.paligemma-multimodal", "dataset", "_annotations.dataset.jsonl")
    
    print(f"Using dataset path: {IMAGE_PATH}")
    print(f"Using annotation path: {ANNOTATION_PATH}")
    
    print("Running object detection demo...")
    detections = detect_objects(
        image_path=IMAGE_PATH,
        annotation_path=ANNOTATION_PATH,
        num_images=1  # Only use one image for testing
    )
    print("Detection complete!")
    print(f"Found {len(detections)} objects")

if __name__ == "__main__":
    main() 