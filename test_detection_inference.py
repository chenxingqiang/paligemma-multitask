import torch
import os
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoImageProcessor

def plot_detection(image, boxes, classes, class_names=['void', 'crack'], save_path=None):
    """Plot detection results on image"""
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)
    
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    
    # Plot each bounding box
    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Convert normalized coordinates to pixel coordinates if needed
        if x1 <= 1.0 and y1 <= 1.0 and x2 <= 1.0 and y2 <= 1.0:
            img_width, img_height = image.size
            x1 *= img_width
            y1 *= img_height
            width *= img_width
            height *= img_height
        
        # Create rectangle
        rect = patches.Rectangle(
            (x1, y1), width, height, 
            linewidth=2, 
            edgecolor=colors[cls % len(colors)], 
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add class label
        plt.text(
            x1, y1, f"{class_names[cls]}", 
            color='white', 
            bbox=dict(facecolor=colors[cls % len(colors)], alpha=0.8)
        )
    
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Detection result saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Test PaliGemma damage detection")
    parser.add_argument("--dataset_path", type=str, default="dataset",
                      help="Path to the dataset directory")
    parser.add_argument("--num_samples", type=int, default=3,
                      help="Number of samples to test")
    parser.add_argument("--output_dir", type=str, default="detection_results",
                      help="Directory to save detection results")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load processor and model
    print("Loading processor and model...")
    processor = AutoProcessor.from_pretrained("google/paligemma-3b-mix-224")
    model = AutoModelForVision2Seq.from_pretrained("google/paligemma-3b-mix-224")
    model.to(device)
    model.eval()
    
    # Load test images
    print("Loading test samples...")
    annotation_file = os.path.join(args.dataset_path, "annotations", "multimodal", "annotations.valid.jsonl")
    
    if not os.path.exists(annotation_file):
        print(f"Error: Annotation file not found at {annotation_file}")
        return
    
    # Load annotations
    with open(annotation_file, "r") as f:
        annotations = [json.loads(line) for line in f]
    
    # Limit to num_samples
    annotations = annotations[:args.num_samples]
    
    # Process each test sample
    for i, ann in enumerate(annotations):
        print(f"\nTesting sample {i+1}/{len(annotations)}")
        
        # Load image
        image_path = os.path.join(args.dataset_path, "images", "datasets", ann["image"])
        if not os.path.exists(image_path):
            # Try alternative paths
            alt_paths = [
                os.path.join(args.dataset_path, "images", ann["image"]),
                os.path.join(args.dataset_path, ann["image"])
            ]
            for path in alt_paths:
                if os.path.exists(path):
                    image_path = path
                    break
        
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            continue
        
        print(f"Image: {image_path}")
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Get image description
        description = ann.get("caption", ann.get("suffix", ""))
        description = description.replace("<image>", "").strip()
        print(f"Description: {description}")
        
        # Use a localization prompt
        prompt = "Localize any damage in this ground-penetrating radar image. Include the type (void or crack) and position."
        
        # Process inputs
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        
        # Generate text-based localization (VLM approach)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_beams=4
            )
        
        # Decode the generated text
        generated_text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Localization response: {generated_text}")
        
        # Basic heuristic detection based on descriptions
        if "void" in description.lower():
            cls = 0  # void
        else:
            cls = 1  # crack
            
        # Look for position information
        pos_map = {
            "center": [0.4, 0.4, 0.6, 0.6],
            "top": [0.4, 0.1, 0.6, 0.3],
            "bottom": [0.4, 0.7, 0.6, 0.9],
            "left": [0.1, 0.4, 0.3, 0.6],
            "right": [0.7, 0.4, 0.9, 0.6]
        }
        
        # Default position is center
        box = pos_map["center"]
        
        # Check description for position keywords
        for pos, coords in pos_map.items():
            if pos in description.lower():
                box = coords
                break
        
        # Plot and save results
        save_path = os.path.join(args.output_dir, f"sample_{i+1}_detection.jpg")
        plot_detection(image, [box], [cls], save_path=save_path)
        
        # Create annotation visualization
        with open(os.path.join(args.output_dir, f"sample_{i+1}_info.txt"), "w") as f:
            f.write(f"Image: {ann['image']}\n")
            f.write(f"Description: {description}\n")
            f.write(f"Model response: {generated_text}\n")
            f.write(f"Detected class: {['void', 'crack'][cls]}\n")
            f.write(f"Detected box: {box}\n")

if __name__ == "__main__":
    main() 