import json
import os
import shutil
from pathlib import Path

def load_jsonl(file_path):
    """Load JSONL file and return list of entries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data, file_path):
    """Save list of entries to JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def create_directory(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_dataset(base_dir="dataset/p-1.v1i.paligemma-multimodal/dataset"):
    """Process the dataset files and organize them into train and test directories."""
    # Define paths
    train1_path = os.path.join(base_dir, "_annotations.train1.jsonl")
    train2_path = os.path.join(base_dir, "_annotations.train2.jsonl")
    valid1_path = os.path.join(base_dir, "_annotations.valid1.jsonl")
    valid2_path = os.path.join(base_dir, "_annotations.valid2.jsonl")
    
    # Create output directories
    output_base = "processed_dataset"
    train_dir = os.path.join(output_base, "train")
    test_dir = os.path.join(output_base, "test")
    train_images_dir = os.path.join(train_dir, "images")
    test_images_dir = os.path.join(test_dir, "images")
    
    create_directory(train_dir)
    create_directory(test_dir)
    create_directory(train_images_dir)
    create_directory(test_images_dir)
    
    # Load data
    train1_data = load_jsonl(train1_path)
    train2_data = load_jsonl(train2_path)
    valid1_data = load_jsonl(valid1_path)
    valid2_data = load_jsonl(valid2_path)
    
    # Save annotations to new locations
    save_jsonl(train1_data, os.path.join(train_dir, "annotations_basic.jsonl"))
    save_jsonl(train2_data, os.path.join(train_dir, "annotations_technical.jsonl"))
    save_jsonl(valid1_data, os.path.join(test_dir, "annotations_basic.jsonl"))
    save_jsonl(valid2_data, os.path.join(test_dir, "annotations_technical.jsonl"))
    
    # Create combined annotation files
    save_jsonl(train1_data + train2_data, os.path.join(train_dir, "annotations_combined.jsonl"))
    save_jsonl(valid1_data + valid2_data, os.path.join(test_dir, "annotations_combined.jsonl"))
    
    # Create lists of image filenames for copying
    train_images = set([entry["image"].split("/")[-1] for entry in train1_data + train2_data])
    test_images = set([entry["image"].split("/")[-1] for entry in valid1_data + valid2_data])
    
    # Copy images (if they exist in the dataset directory)
    for img in train_images:
        src_path = os.path.join(base_dir, img)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(train_images_dir, img))
        else:
            print(f"Warning: Image {img} not found in dataset directory")
    
    for img in test_images:
        src_path = os.path.join(base_dir, img)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(test_images_dir, img))
        else:
            print(f"Warning: Image {img} not found in dataset directory")
    
    # Generate statistics
    print(f"Training set: {len(train_images)} images")
    print(f"  - Basic annotations: {len(train1_data)} entries")
    print(f"  - Technical annotations: {len(train2_data)} entries")
    print(f"Testing set: {len(test_images)} images")
    print(f"  - Basic annotations: {len(valid1_data)} entries")
    print(f"  - Technical annotations: {len(valid2_data)} entries")
    
    # Create a metadata file with dataset information
    metadata = {
        "dataset_name": "Ground-Penetrating Radar Damage Detection",
        "description": "Dataset for detecting voids and cracks in ground-penetrating radar images",
        "statistics": {
            "train_images": len(train_images),
            "train_basic_annotations": len(train1_data),
            "train_technical_annotations": len(train2_data),
            "test_images": len(test_images),
            "test_basic_annotations": len(valid1_data),
            "test_technical_annotations": len(valid2_data)
        },
        "annotation_types": {
            "basic": "General descriptions of damage appearance and location",
            "technical": "Detailed technical descriptions including amplitude, attenuation, and distribution range"
        }
    }
    
    with open(os.path.join(output_base, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Dataset processed and organized in '{output_base}' directory")

if __name__ == "__main__":
    process_dataset() 