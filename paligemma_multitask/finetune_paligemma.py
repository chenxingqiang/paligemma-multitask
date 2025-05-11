import os
import json
import argparse
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor, 
    AutoModelForVision2Seq, 
    Trainer, 
    TrainingArguments,
    default_data_collator
)
from PIL import Image
import numpy as np
from datasets import load_dataset

class GPRDamageDataset(Dataset):
    """Dataset for Ground-Penetrating Radar Damage Detection."""
    
    def __init__(self, jsonl_file, processor, base_dir=".", max_length=512):
        """
        Initialize the dataset.
        
        Args:
            jsonl_file: Path to the JSONL file containing the dataset
            processor: PaLI-GEMMA processor
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
        
        # Load and process image
        image_path = os.path.join(self.base_dir, item["image"])
        image = Image.open(image_path).convert("RGB")
        
        # Prepare text
        prefix = item["prefix"]
        suffix = item["suffix"]
        
        # Process inputs
        encoding = self.processor(
            images=image,
            text=prefix,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Process labels (target text)
        with self.processor.as_target_processor():
            labels = self.processor(
                text=suffix,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            ).input_ids
        
        # Remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["labels"] = labels.squeeze()
        
        return encoding

def train(args):
    """Train PaLI-GEMMA on the GPR damage dataset."""
    
    # Load processor and model
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModelForVision2Seq.from_pretrained(args.model_name)
    
    # Prepare datasets
    train_dataset = GPRDamageDataset(
        os.path.join(args.data_dir, f"{args.dataset_version}_train.jsonl"),
        processor,
        base_dir=args.base_dir,
        max_length=args.max_length
    )
    
    eval_dataset = GPRDamageDataset(
        os.path.join(args.data_dir, f"{args.dataset_version}_test.jsonl"),
        processor,
        base_dir=args.base_dir,
        max_length=args.max_length
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
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
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
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
    parser = argparse.ArgumentParser(description="Fine-tune PaLI-GEMMA on GPR damage dataset")
    
    # Model and data arguments
    parser.add_argument("--model_name", type=str, default="google/paligemma-3b-mix-224", 
                        help="Model name or path")
    parser.add_argument("--data_dir", type=str, default="paligemma_dataset",
                        help="Directory containing the dataset files")
    parser.add_argument("--base_dir", type=str, default="processed_dataset",
                        help="Base directory for image paths")
    parser.add_argument("--output_dir", type=str, default="paligemma_finetuned",
                        help="Output directory for the fine-tuned model")
    parser.add_argument("--dataset_version", type=str, default="combined",
                        choices=["basic_detection", "descriptive", "technical", "combined"],
                        help="Dataset version to use for training")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")
    
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