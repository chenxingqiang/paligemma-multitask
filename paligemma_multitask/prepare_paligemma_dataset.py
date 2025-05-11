import json
import os
import random
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

def prepare_paligemma_dataset(processed_dir="processed_dataset", output_dir="paligemma_dataset"):
    """
    Prepare the dataset for fine-tuning with PaLI-GEMMA.
    
    This function:
    1. Creates formatted datasets for different training scenarios
    2. Generates prompts for void and crack detection
    3. Creates different versions of the dataset with various prompt formats
    """
    # Create output directory
    create_directory(output_dir)
    
    # Load processed data
    train_basic = load_jsonl(os.path.join(processed_dir, "train", "annotations_basic.jsonl"))
    train_technical = load_jsonl(os.path.join(processed_dir, "train", "annotations_technical.jsonl"))
    test_basic = load_jsonl(os.path.join(processed_dir, "test", "annotations_basic.jsonl"))
    test_technical = load_jsonl(os.path.join(processed_dir, "test", "annotations_technical.jsonl"))
    
    # Define prompt templates
    prompt_templates = {
        "detect": "detect {damage_type}",
        "describe": "describe the {damage_type} in this image",
        "analyze": "analyze the ground-penetrating radar image for {damage_type}",
        "locate": "locate and describe the {damage_type} in this radar image",
        "technical": "provide technical details about the {damage_type} in this radar image"
    }
    
    # Create different dataset versions
    
    # 1. Basic detection dataset (just detect void/crack)
    basic_detection_train = []
    basic_detection_test = []
    
    for entry in train_basic + train_technical:
        # Determine if the entry contains void, crack, or both
        text = entry["suffix"].lower()
        has_void = "void" in text
        has_crack = "crack" in text
        
        if has_void:
            basic_detection_train.append({
                "image": os.path.join("train", "images", entry["image"]),
                "prefix": "detect void",
                "suffix": "void" if has_void else "no void"
            })
        
        if has_crack:
            basic_detection_train.append({
                "image": os.path.join("train", "images", entry["image"]),
                "prefix": "detect crack",
                "suffix": "crack" if has_crack else "no crack"
            })
    
    for entry in test_basic + test_technical:
        text = entry["suffix"].lower()
        has_void = "void" in text
        has_crack = "crack" in text
        
        if has_void:
            basic_detection_test.append({
                "image": os.path.join("test", "images", entry["image"]),
                "prefix": "detect void",
                "suffix": "void" if has_void else "no void"
            })
        
        if has_crack:
            basic_detection_test.append({
                "image": os.path.join("test", "images", entry["image"]),
                "prefix": "detect crack",
                "suffix": "crack" if has_crack else "no crack"
            })
    
    # 2. Descriptive dataset (use the full descriptions)
    descriptive_train = []
    descriptive_test = []
    
    for entry in train_basic:
        text = entry["suffix"].lower()
        prompt_type = random.choice(list(prompt_templates.keys()))
        
        if "void" in text:
            descriptive_train.append({
                "image": os.path.join("train", "images", entry["image"]),
                "prefix": prompt_templates[prompt_type].format(damage_type="void"),
                "suffix": entry["suffix"]
            })
        
        if "crack" in text:
            descriptive_train.append({
                "image": os.path.join("train", "images", entry["image"]),
                "prefix": prompt_templates[prompt_type].format(damage_type="crack"),
                "suffix": entry["suffix"]
            })
    
    for entry in test_basic:
        text = entry["suffix"].lower()
        prompt_type = random.choice(list(prompt_templates.keys()))
        
        if "void" in text:
            descriptive_test.append({
                "image": os.path.join("test", "images", entry["image"]),
                "prefix": prompt_templates[prompt_type].format(damage_type="void"),
                "suffix": entry["suffix"]
            })
        
        if "crack" in text:
            descriptive_test.append({
                "image": os.path.join("test", "images", entry["image"]),
                "prefix": prompt_templates[prompt_type].format(damage_type="crack"),
                "suffix": entry["suffix"]
            })
    
    # 3. Technical dataset (use the technical descriptions)
    technical_train = []
    technical_test = []
    
    for entry in train_technical:
        text = entry["suffix"].lower()
        
        if "void" in text:
            technical_train.append({
                "image": os.path.join("train", "images", entry["image"]),
                "prefix": "provide technical details about the void in this radar image",
                "suffix": entry["suffix"]
            })
        
        if "crack" in text:
            technical_train.append({
                "image": os.path.join("train", "images", entry["image"]),
                "prefix": "provide technical details about the crack in this radar image",
                "suffix": entry["suffix"]
            })
    
    for entry in test_technical:
        text = entry["suffix"].lower()
        
        if "void" in text:
            technical_test.append({
                "image": os.path.join("test", "images", entry["image"]),
                "prefix": "provide technical details about the void in this radar image",
                "suffix": entry["suffix"]
            })
        
        if "crack" in text:
            technical_test.append({
                "image": os.path.join("test", "images", entry["image"]),
                "prefix": "provide technical details about the crack in this radar image",
                "suffix": entry["suffix"]
            })
    
    # 4. Combined dataset (mix of all types)
    combined_train = basic_detection_train + descriptive_train + technical_train
    combined_test = basic_detection_test + descriptive_test + technical_test
    
    # Shuffle the combined datasets
    random.shuffle(combined_train)
    random.shuffle(combined_test)
    
    # Save all dataset versions
    save_jsonl(basic_detection_train, os.path.join(output_dir, "basic_detection_train.jsonl"))
    save_jsonl(basic_detection_test, os.path.join(output_dir, "basic_detection_test.jsonl"))
    save_jsonl(descriptive_train, os.path.join(output_dir, "descriptive_train.jsonl"))
    save_jsonl(descriptive_test, os.path.join(output_dir, "descriptive_test.jsonl"))
    save_jsonl(technical_train, os.path.join(output_dir, "technical_train.jsonl"))
    save_jsonl(technical_test, os.path.join(output_dir, "technical_test.jsonl"))
    save_jsonl(combined_train, os.path.join(output_dir, "combined_train.jsonl"))
    save_jsonl(combined_test, os.path.join(output_dir, "combined_test.jsonl"))
    
    # Create metadata
    metadata = {
        "dataset_name": "PaLI-GEMMA Ground-Penetrating Radar Dataset",
        "description": "Dataset prepared for fine-tuning PaLI-GEMMA on ground-penetrating radar damage detection",
        "versions": {
            "basic_detection": "Simple detection of void or crack presence",
            "descriptive": "General descriptions of damage appearance and location",
            "technical": "Detailed technical descriptions of damage characteristics",
            "combined": "Mix of all prompt types and description formats"
        },
        "statistics": {
            "basic_detection_train": len(basic_detection_train),
            "basic_detection_test": len(basic_detection_test),
            "descriptive_train": len(descriptive_train),
            "descriptive_test": len(descriptive_test),
            "technical_train": len(technical_train),
            "technical_test": len(technical_test),
            "combined_train": len(combined_train),
            "combined_test": len(combined_test)
        },
        "prompt_templates": prompt_templates
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"PaLI-GEMMA dataset prepared in '{output_dir}' directory")
    print(f"Basic detection dataset: {len(basic_detection_train)} train, {len(basic_detection_test)} test")
    print(f"Descriptive dataset: {len(descriptive_train)} train, {len(descriptive_test)} test")
    print(f"Technical dataset: {len(technical_train)} train, {len(technical_test)} test")
    print(f"Combined dataset: {len(combined_train)} train, {len(combined_test)} test")

if __name__ == "__main__":
    prepare_paligemma_dataset() 