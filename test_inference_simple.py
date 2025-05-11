import torch
import os
import json
import argparse
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

def main():
    parser = argparse.ArgumentParser(description="Test PaliGemma caption generation")
    parser.add_argument("--dataset_path", type=str, default="dataset",
                      help="Path to the dataset directory")
    parser.add_argument("--num_samples", type=int, default=3,
                      help="Number of samples to test")
    args = parser.parse_args()
    
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
        
        # Get the ground truth caption
        ground_truth = ann.get("caption", ann.get("suffix", ""))
        ground_truth = ground_truth.replace("<image>", "").strip()
        print(f"Ground truth: {ground_truth}")
        
        # Process inputs
        prompt = "Describe the image in detail."
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        
        # Generate text
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
        print(f"Generated: {generated_text}")
        
        # Process inputs with a specialized prompt
        specialized_prompt = "This is a ground-penetrating radar image used in civil engineering. Describe in detail what damage is visible in this image, including the type (void or crack), location, and characteristics."
        inputs = processor(images=image, text=specialized_prompt, return_tensors="pt").to(device)
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_beams=4
            )
        
        # Decode the generated text
        generated_text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated with specialized prompt: {generated_text}")
        
        # Try with VQA format for damage type
        vqa_prompt = "What type of damage is shown in this ground-penetrating radar image: void or crack?"
        vqa_inputs = processor(images=image, text=vqa_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            vqa_outputs = model.generate(
                **vqa_inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_beams=4
            )
        
        # Decode VQA response
        vqa_text = processor.tokenizer.decode(vqa_outputs[0], skip_special_tokens=True)
        print(f"VQA Response: {vqa_text}")

if __name__ == "__main__":
    main() 