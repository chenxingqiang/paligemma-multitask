import os
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, AutoModelForVision2Seq, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from PIL import Image
import json

class SimpleCaptionDataset(Dataset):
    def __init__(self, dataset_path, processor, split="train"):
        self.dataset_path = dataset_path
        self.processor = processor
        self.split = split
        self.image_processor = processor.image_processor
        self.tokenizer = processor.tokenizer
        
        # Load annotations from multimodal directory
        annotations_dir = os.path.join(dataset_path, "annotations", "multimodal")
        if split == "train":
            annotation_file = os.path.join(annotations_dir, "annotations.train.jsonl")
        else:
            annotation_file = os.path.join(annotations_dir, "annotations.valid.jsonl")
        
        # Check if the file exists
        if not os.path.exists(annotation_file):
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        
        # Load annotations
        self.annotations = []
        with open(annotation_file, "r") as f:
            for line in f:
                self.annotations.append(json.loads(line))
                
        # In debug mode, limit the dataset size
        if os.getenv("DEBUG", "false").lower() == "true":
            debug_size = min(10, len(self.annotations))
            self.annotations = self.annotations[:debug_size]
            print(f"DEBUG mode: Using {debug_size} samples")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        # Load image
        image_path = os.path.join(self.dataset_path, "images", "datasets", ann["image"])
        if not os.path.exists(image_path):
            # Try alternative paths
            alt_paths = [
                os.path.join(self.dataset_path, "images", ann["image"]),
                os.path.join(self.dataset_path, ann["image"])
            ]
            for path in alt_paths:
                if os.path.exists(path):
                    image_path = path
                    break
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert("RGB")
        
        # Get text
        caption = ann.get("caption", ann.get("suffix", ""))
        caption = caption.replace("<image>", "").strip()
        
        # Process image and text
        inputs = self.processor(
            images=image,
            text=caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=384
        )
        
        # Prepare the input dictionary
        result = {
            "pixel_values": inputs.pixel_values.squeeze(),
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": inputs.input_ids.squeeze().clone()
        }
        
        # Set padding tokens to -100 to ignore them in the loss
        result["labels"][result["labels"] == self.tokenizer.pad_token_id] = -100
        
        return result

def train_caption_model():
    parser = argparse.ArgumentParser(description="Train PaliGemma for image captioning")
    parser.add_argument("--dataset_path", type=str, default="dataset", help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="caption_model", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for efficient fine-tuning")
    args = parser.parse_args()
    
    # Set debug environment variable if needed
    if args.debug:
        os.environ["DEBUG"] = "true"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize processor and model
    print("Loading processor and model...")
    processor = AutoProcessor.from_pretrained("google/paligemma-3b-mix-224")
    model = AutoModelForVision2Seq.from_pretrained("google/paligemma-3b-mix-224")
    
    # Apply LoRA if requested
    if args.use_lora:
        print("Applying LoRA for efficient fine-tuning...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = SimpleCaptionDataset(args.dataset_path, processor, split="train")
    val_dataset = SimpleCaptionDataset(args.dataset_path, processor, split="valid")
    
    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        save_total_limit=2,
        evaluation_strategy="steps",
        fp16=torch.cuda.is_available() and not args.debug,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        weight_decay=0.01,
        max_grad_norm=1.0,
        report_to="none",
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    model.save_pretrained(os.path.join(args.output_dir, "final_model"))
    processor.save_pretrained(os.path.join(args.output_dir, "processor"))
    
    print("Training complete!")

if __name__ == "__main__":
    train_caption_model() 