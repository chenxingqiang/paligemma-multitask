import os
import json
import argparse
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor, 
    AutoModelForVision2Seq,
    default_data_collator
)
from trl import GRPOConfig, GRPOTrainer
from PIL import Image
import numpy as np
from datasets import Dataset as HFDataset

class GPRDamageDataset(Dataset):
    """Dataset for Ground-Penetrating Radar Damage Detection."""
    
    def __init__(self, jsonl_file, processor=None, base_dir=".", max_length=512):
        """
        Initialize the dataset.
        
        Args:
            jsonl_file: Path to the JSONL file containing the dataset
            processor: PaLI-GEMMA processor (optional for GRPO)
            base_dir: Base directory for image paths
            max_length: Maximum length for text tokenization
        """
        self.data = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        self.processor = processor
        self.base_dir = base_dir
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        image_path = os.path.join(self.base_dir, item["image"])
        image = Image.open(image_path).convert("RGB")
        
        # Prepare text
        prompt = item["prefix"]
        target = item["suffix"]
        
        # For GRPO, we return a dict with the prompt and image
        return {
            "prompt": prompt,
            "image": image,
            "target": target
        }

def convert_to_hf_dataset(dataset):
    """Convert our custom dataset to a HuggingFace dataset for GRPO."""
    
    # Extract data from our custom dataset
    prompts = []
    images = []
    targets = []
    
    for i in range(len(dataset)):
        item = dataset[i]
        prompts.append(item["prompt"])
        images.append(item["image"])
        targets.append(item["target"])
    
    # Create a HuggingFace dataset
    hf_dataset = HFDataset.from_dict({
        "prompt": prompts,
        "image": images,
        "target": targets
    })
    
    return hf_dataset

def reward_technical_detail(completions, **kwargs):
    """
    Reward function that gives higher scores to completions with technical details.
    
    This function rewards completions that:
    1. Mention technical terms like "amplitude", "attenuation", "distribution range"
    2. Include specific measurements or characteristics
    3. Have a structured format with clear sections
    """
    rewards = []
    
    technical_terms = [
        "amplitude", "attenuation", "distribution range", "hyperbolic", 
        "multiple reflections", "downward-opening", "symmetrical", "irregular"
    ]
    
    for completion in completions:
        score = 0
        
        # Check for technical terms
        for term in technical_terms:
            if term in completion.lower():
                score += 1
        
        # Check for structured format (sections)
        if "characterized by" in completion.lower():
            score += 2
            
        # Check for specific measurements
        if "strong" in completion.lower() or "weak" in completion.lower():
            score += 1
        if "small" in completion.lower() or "large" in completion.lower():
            score += 1
            
        # Normalize score (0-10 range)
        score = min(10, score)
        rewards.append(float(score))
    
    return rewards

def reward_location_accuracy(completions, **kwargs):
    """
    Reward function that gives higher scores to completions with accurate location descriptions.
    
    This function rewards completions that:
    1. Clearly specify the location of damage (e.g., "center", "upper left")
    2. Use consistent terminology for locations
    """
    rewards = []
    
    location_terms = [
        "center", "upper left", "upper right", "lower left", "lower right",
        "top left", "top right", "bottom left", "bottom right",
        "left of center", "right of center", "above center", "below center"
    ]
    
    for completion in completions:
        score = 0
        
        # Check for location terms
        for term in location_terms:
            if term in completion.lower():
                score += 2
                break
        
        # Check for specific location descriptions
        if "located in" in completion.lower() or "located at" in completion.lower():
            score += 2
            
        # Normalize score (0-5 range)
        score = min(5, score)
        rewards.append(float(score))
    
    return rewards

def reward_reporter_format(completions, **kwargs):
    """
    Reward function that gives higher scores to completions with a reporter-style format.
    
    This function rewards completions that:
    1. Have a clear, concise reporting style
    2. Present information in a structured manner
    3. Avoid unnecessary words and focus on facts
    """
    rewards = []
    
    for completion in completions:
        score = 0
        
        # Check for concise reporting style (avoid "I think", "I believe", etc.)
        if "I " not in completion.lower() and "we " not in completion.lower():
            score += 2
            
        # Check for factual statements
        if "is " in completion.lower() or "are " in completion.lower():
            score += 1
            
        # Check for structured information
        if "." in completion and len(completion.split(".")) >= 3:
            score += 2
            
        # Normalize score (0-5 range)
        score = min(5, score)
        rewards.append(float(score))
    
    return rewards

def process_images_for_grpo(examples, processor):
    """Process images for GRPO training."""
    examples["images"] = [processor.image_processor(img) for img in examples["image"]]
    return examples

def train(args):
    """Train PaLI-GEMMA on the GPR damage dataset using GRPO."""
    
    # Load processor and model
    processor = AutoProcessor.from_pretrained(args.model_name)
    
    # Prepare datasets
    train_dataset = GPRDamageDataset(
        os.path.join(args.data_dir, f"{args.dataset_version}_train.jsonl"),
        processor=processor,
        base_dir=args.base_dir,
        max_length=args.max_length
    )
    
    eval_dataset = GPRDamageDataset(
        os.path.join(args.data_dir, f"{args.dataset_version}_test.jsonl"),
        processor=processor,
        base_dir=args.base_dir,
        max_length=args.max_length
    )
    
    # Convert to HuggingFace datasets for GRPO
    train_hf_dataset = convert_to_hf_dataset(train_dataset)
    eval_hf_dataset = convert_to_hf_dataset(eval_dataset)
    
    # Process images
    train_hf_dataset = train_hf_dataset.map(
        lambda examples: process_images_for_grpo(examples, processor),
        batched=True
    )
    eval_hf_dataset = eval_hf_dataset.map(
        lambda examples: process_images_for_grpo(examples, processor),
        batched=True
    )
    
    # Set up GRPO training arguments
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
        hub_token=args.hub_token if args.push_to_hub else None,
        fp16=args.fp16,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=10,
        report_to=["tensorboard"],
        # GRPO specific parameters
        beta=args.beta,
        num_iterations=args.num_iterations,
        epsilon=args.epsilon,
        reward_weights=args.reward_weights,
        num_generations=args.num_generations,
        temperature=args.temperature,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        use_vllm=args.use_vllm,
        log_completions=True
    )
    
    # Define reward functions
    reward_functions = [
        reward_technical_detail,
        reward_location_accuracy,
        reward_reporter_format
    ]
    
    # Initialize GRPO trainer
    trainer = GRPOTrainer(
        model=args.model_name,
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=train_hf_dataset,
        eval_dataset=eval_hf_dataset,
        processing_class=processor
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    processor.save_pretrained(os.path.join(args.output_dir, "final_model"))
    
    # Push to hub if requested
    if args.push_to_hub:
        trainer.push_to_hub()

def main():
    parser = argparse.ArgumentParser(description="Fine-tune PaLI-GEMMA with GRPO on GPR damage dataset")
    
    # Model and data arguments
    parser.add_argument("--model_name", type=str, default="google/paligemma-3b-mix-224", 
                        help="Model name or path")
    parser.add_argument("--data_dir", type=str, default="paligemma_dataset",
                        help="Directory containing the dataset files")
    parser.add_argument("--base_dir", type=str, default="processed_dataset",
                        help="Base directory for image paths")
    parser.add_argument("--output_dir", type=str, default="paligemma_grpo_finetuned",
                        help="Output directory for the fine-tuned model")
    parser.add_argument("--dataset_version", type=str, default="combined",
                        choices=["basic_detection", "descriptive", "technical", "combined"],
                        help="Dataset version to use for training")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=1e-6,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")
    
    # GRPO specific arguments
    parser.add_argument("--beta", type=float, default=0.04,
                        help="KL coefficient for GRPO")
    parser.add_argument("--num_iterations", type=int, default=1,
                        help="Number of iterations per batch for GRPO")
    parser.add_argument("--epsilon", type=float, default=0.2,
                        help="Epsilon value for clipping in GRPO")
    parser.add_argument("--reward_weights", type=float, nargs="+", default=None,
                        help="Weights for each reward function")
    parser.add_argument("--num_generations", type=int, default=8,
                        help="Number of generations per prompt to sample")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Temperature for sampling")
    parser.add_argument("--max_prompt_length", type=int, default=512,
                        help="Maximum length of the prompt")
    parser.add_argument("--max_completion_length", type=int, default=256,
                        help="Maximum length of the generated completion")
    parser.add_argument("--use_vllm", action="store_true",
                        help="Use vLLM for generation acceleration")
    
    # HuggingFace Hub arguments
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push model to HuggingFace Hub")
    parser.add_argument("--hub_model_id", type=str, default=None,
                        help="Model ID for HuggingFace Hub")
    parser.add_argument("--hub_token", type=str, default=None,
                        help="HuggingFace Hub token")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train the model
    train(args)

if __name__ == "__main__":
    main() 