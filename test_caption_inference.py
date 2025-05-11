import torch
import os
import json
import argparse
from PIL import Image
from transformers import AutoProcessor
from train_caption_simple import CaptioningModule

def main():
    parser = argparse.ArgumentParser(description="Test PaliGemma caption generation")
    parser.add_argument("--model_path", type=str, default="caption_model/checkpoint_epoch_1",
                      help="Path to the trained model checkpoint")
    parser.add_argument("--dataset_path", type=str, default="dataset",
                      help="Path to the dataset directory")
    parser.add_argument("--num_samples", type=int, default=5,
                      help="Number of samples to test")
    args = parser.parse_args()
    
    # Load processor
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained("google/paligemma-3b-mix-224")
    
    # Load base model
    print("Loading base model...")
    from transformers import AutoModelForVision2Seq
    base_model = AutoModelForVision2Seq.from_pretrained("google/paligemma-3b-mix-224")
    
    # Create captioning module
    model = CaptioningModule(base_model, processor.tokenizer)
    
    # Load trained weights
    print(f"Loading trained weights from {args.model_path}...")
    model_path = os.path.join(args.model_path, "model.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print("Model loaded successfully")
    else:
        print(f"Error: Model file not found at {model_path}")
        return
    
    # Set model to evaluation mode
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
        
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor.image_processor(images=image, return_tensors="pt").pixel_values
        
        # Get the ground truth caption
        ground_truth = ann.get("caption", ann.get("suffix", ""))
        ground_truth = ground_truth.replace("<image>", "").strip()
        print(f"Ground truth: {ground_truth}")
        
        # Generate caption
        with torch.no_grad():
            # Prepare inputs for vision tower
            vision_outputs = model.vision_tower(pixel_values)
            image_features = vision_outputs.last_hidden_state
            
            # Use standard PaliGemma generation with image features
            gen_kwargs = {
                "max_new_tokens": 100,
                "num_beams": 4,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True
            }
            
            # Prepare generation inputs
            input_ids = processor.tokenizer(['<image>'], return_tensors="pt").input_ids
            
            # Generate caption using PaliGemma model directly
            outputs = base_model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                **gen_kwargs
            )
            
            # Decode the generated caption
            generated_text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Generated: {generated_text}")

if __name__ == "__main__":
    main() 