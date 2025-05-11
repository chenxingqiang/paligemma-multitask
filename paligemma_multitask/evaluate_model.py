import os
import json
import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_jsonl(file_path):
    """Load JSONL file and return list of entries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def evaluate_basic_detection(model, processor, test_file, base_dir=".", device="cuda"):
    """
    Evaluate the model on basic detection tasks (void/crack detection).
    
    Args:
        model: The fine-tuned PaLI-GEMMA model
        processor: The PaLI-GEMMA processor
        test_file: Path to the test JSONL file
        base_dir: Base directory for image paths
        device: Device to run inference on
    
    Returns:
        Dictionary containing evaluation metrics
    """
    # Load test data
    test_data = load_jsonl(test_file)
    
    # Prepare for evaluation
    predictions = []
    ground_truth = []
    
    # Process each test example
    for item in tqdm(test_data, desc="Evaluating"):
        # Load and process image
        image_path = os.path.join(base_dir, item["image"])
        image = Image.open(image_path).convert("RGB")
        
        # Prepare text prompt
        prompt = item["prefix"]
        
        # Process inputs
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        
        # Generate prediction
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=50,
                num_beams=5,
                early_stopping=True
            )
        
        # Decode prediction
        prediction = processor.decode(output[0], skip_special_tokens=True)
        
        # Store prediction and ground truth
        predictions.append(prediction.lower().strip())
        ground_truth.append(item["suffix"].lower().strip())
    
    # Calculate metrics
    accuracy = accuracy_score([1 if gt in ["void", "crack"] else 0 for gt in ground_truth],
                             [1 if pred in ["void", "crack"] else 0 for pred in predictions])
    
    # For void detection
    void_gt = [1 if gt == "void" else 0 for gt in ground_truth]
    void_pred = [1 if pred == "void" else 0 for pred in predictions]
    void_precision = precision_score(void_gt, void_pred, zero_division=0)
    void_recall = recall_score(void_gt, void_pred, zero_division=0)
    void_f1 = f1_score(void_gt, void_pred, zero_division=0)
    
    # For crack detection
    crack_gt = [1 if gt == "crack" else 0 for gt in ground_truth]
    crack_pred = [1 if pred == "crack" else 0 for pred in predictions]
    crack_precision = precision_score(crack_gt, crack_pred, zero_division=0)
    crack_recall = recall_score(crack_gt, crack_pred, zero_division=0)
    crack_f1 = f1_score(crack_gt, crack_pred, zero_division=0)
    
    # Create confusion matrix
    labels = ["void", "crack", "other"]
    gt_labels = ["void" if gt == "void" else "crack" if gt == "crack" else "other" for gt in ground_truth]
    pred_labels = ["void" if pred == "void" else "crack" if pred == "crack" else "other" for pred in predictions]
    
    cm = confusion_matrix(
        [labels.index(l) for l in gt_labels], 
        [labels.index(l) for l in pred_labels],
        labels=range(len(labels))
    )
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    
    # Return metrics
    return {
        "accuracy": accuracy,
        "void_precision": void_precision,
        "void_recall": void_recall,
        "void_f1": void_f1,
        "crack_precision": crack_precision,
        "crack_recall": crack_recall,
        "crack_f1": crack_f1,
        "confusion_matrix": cm.tolist(),
        "predictions": predictions,
        "ground_truth": ground_truth
    }

def evaluate_descriptive(model, processor, test_file, base_dir=".", device="cuda"):
    """
    Evaluate the model on descriptive tasks (generating descriptions).
    
    Args:
        model: The fine-tuned PaLI-GEMMA model
        processor: The PaLI-GEMMA processor
        test_file: Path to the test JSONL file
        base_dir: Base directory for image paths
        device: Device to run inference on
    
    Returns:
        Dictionary containing evaluation metrics and sample predictions
    """
    # Load test data
    test_data = load_jsonl(test_file)
    
    # Select a subset of examples for detailed evaluation
    sample_size = min(20, len(test_data))
    sample_indices = np.random.choice(len(test_data), sample_size, replace=False)
    samples = [test_data[i] for i in sample_indices]
    
    # Process each sample
    results = []
    
    for item in tqdm(samples, desc="Evaluating samples"):
        # Load and process image
        image_path = os.path.join(base_dir, item["image"])
        image = Image.open(image_path).convert("RGB")
        
        # Prepare text prompt
        prompt = item["prefix"]
        
        # Process inputs
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        
        # Generate prediction
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=512,
                num_beams=5,
                early_stopping=True
            )
        
        # Decode prediction
        prediction = processor.decode(output[0], skip_special_tokens=True)
        
        # Store result
        results.append({
            "image": item["image"],
            "prompt": prompt,
            "prediction": prediction,
            "ground_truth": item["suffix"]
        })
    
    return {
        "sample_results": results
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned PaLI-GEMMA model")
    
    # Model and data arguments
    parser.add_argument("--model_path", type=str, default="paligemma_finetuned/final_model",
                        help="Path to the fine-tuned model")
    parser.add_argument("--data_dir", type=str, default="paligemma_dataset",
                        help="Directory containing the dataset files")
    parser.add_argument("--base_dir", type=str, default="processed_dataset",
                        help="Base directory for image paths")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Output directory for evaluation results")
    parser.add_argument("--dataset_version", type=str, default="combined",
                        choices=["basic_detection", "descriptive", "technical", "combined"],
                        help="Dataset version to use for evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run evaluation on")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and processor
    model = AutoModelForVision2Seq.from_pretrained(args.model_path).to(args.device)
    processor = AutoProcessor.from_pretrained(args.model_path)
    
    # Set model to evaluation mode
    model.eval()
    
    # Evaluate on basic detection dataset
    basic_detection_results = evaluate_basic_detection(
        model,
        processor,
        os.path.join(args.data_dir, "basic_detection_test.jsonl"),
        base_dir=args.base_dir,
        device=args.device
    )
    
    # Save basic detection results
    with open(os.path.join(args.output_dir, "basic_detection_results.json"), "w", encoding="utf-8") as f:
        json.dump(basic_detection_results, f, indent=2, ensure_ascii=False)
    
    # Evaluate on descriptive dataset
    descriptive_results = evaluate_descriptive(
        model,
        processor,
        os.path.join(args.data_dir, f"{args.dataset_version}_test.jsonl"),
        base_dir=args.base_dir,
        device=args.device
    )
    
    # Save descriptive results
    with open(os.path.join(args.output_dir, f"{args.dataset_version}_descriptive_results.json"), "w", encoding="utf-8") as f:
        json.dump(descriptive_results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("Evaluation complete!")
    print(f"Basic detection accuracy: {basic_detection_results['accuracy']:.4f}")
    print(f"Void detection F1 score: {basic_detection_results['void_f1']:.4f}")
    print(f"Crack detection F1 score: {basic_detection_results['crack_f1']:.4f}")
    print(f"Detailed results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 