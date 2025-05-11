import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoProcessor, AutoModelForVision2Seq, get_linear_schedule_with_warmup
from tqdm import tqdm
from PIL import Image
import json

class SimpleCaptionDataset(Dataset):
    def __init__(self, dataset_path, processor, split="train"):
        self.dataset_path = dataset_path
        self.processor = processor
        self.split = split
        
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
            print(f"DEBUG mode: Using {debug_size} samples for {split}")
    
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
        
        # Process image separately
        image_features = self.processor.image_processor(images=image, return_tensors="pt").pixel_values[0]
        
        # Process text separately 
        inputs = self.processor.tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length", 
            truncation=True,
            max_length=384
        )
        
        # Create result dict
        result = {
            "pixel_values": image_features,
            "input_ids": inputs.input_ids[0],
            "attention_mask": inputs.attention_mask[0],
            "caption": caption,
        }
        
        return result

def collate_fn(batch):
    """
    Custom collate function to handle variable length inputs
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    
    # Pad input_ids and attention_mask
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    
    # Get max length
    max_length = max(input_id.size(0) for input_id in input_ids)
    
    # Pad sequences
    padded_input_ids = []
    padded_attention_mask = []
    pad_token_id = 0  # Assume 0 is pad token ID
    
    for input_id, mask in zip(input_ids, attention_mask):
        if input_id.size(0) < max_length:
            # Pad input_ids
            padding = torch.full((max_length - input_id.size(0),), pad_token_id, dtype=input_id.dtype)
            padded_input_ids.append(torch.cat([input_id, padding], dim=0))
            
            # Pad attention_mask
            mask_padding = torch.zeros(max_length - mask.size(0), dtype=mask.dtype)
            padded_attention_mask.append(torch.cat([mask, mask_padding], dim=0))
        else:
            padded_input_ids.append(input_id)
            padded_attention_mask.append(mask)
    
    # Stack
    padded_input_ids = torch.stack(padded_input_ids)
    padded_attention_mask = torch.stack(padded_attention_mask)
    
    # Also collect captions
    captions = [item["caption"] for item in batch]
    
    return {
        "pixel_values": pixel_values,
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "captions": captions,
    }

# Simple captioning module
class CaptioningModule(nn.Module):
    def __init__(self, base_model, tokenizer):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        
        # Get vision tower and language model
        self.vision_tower = base_model.vision_tower
        self.language_model = base_model.language_model
        
        # Get multi-modal projector for combining vision and language
        self.multi_modal_projector = base_model.multi_modal_projector
        
        # Get embeddings layer from language model
        self.embed_tokens = self.language_model.model.embed_tokens
        
        # Special tokens
        self.pad_token_id = tokenizer.pad_token_id
        self.image_token_id = 257152  # From config.image_token_index
        
    def forward(self, pixel_values, input_ids, attention_mask=None):
        batch_size = pixel_values.shape[0]
        
        # Get image features from vision tower
        vision_outputs = self.vision_tower(pixel_values)
        image_features = vision_outputs.last_hidden_state
        
        # Use language model directly without image tokens
        # This is a simplified approach that just focuses on caption training
        
        # Create labels by shifting input_ids right
        labels = input_ids.clone()
        labels = torch.cat([torch.ones((batch_size, 1), dtype=torch.long, device=labels.device) * -100,
                            labels[:, :-1]], dim=1)
        # Mask out padding tokens
        labels[labels == self.pad_token_id] = -100
        
        # Forward pass through language model with text inputs
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs

def train_caption_model():
    parser = argparse.ArgumentParser(description="Train PaliGemma for image captioning")
    parser.add_argument("--dataset_path", type=str, default="dataset", help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="caption_model", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--freeze_vision", action="store_true", default=True, help="Freeze vision tower")
    args = parser.parse_args()
    
    # Set debug environment variable if needed
    if args.debug:
        os.environ["DEBUG"] = "true"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize processor and model
    print("Loading processor and model...")
    processor = AutoProcessor.from_pretrained("google/paligemma-3b-mix-224")
    base_model = AutoModelForVision2Seq.from_pretrained("google/paligemma-3b-mix-224")
    
    # Create our captioning module
    model = CaptioningModule(base_model, processor.tokenizer)
    
    # Move model to device
    model = model.to(device)
    
    # Set modules to train - freeze vision tower if specified
    if args.freeze_vision:
        print("Freezing vision tower parameters...")
        for param in model.vision_tower.parameters():
            param.requires_grad = False
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.2%})")
    
    # Create datasets and data loaders
    print("Creating datasets...")
    train_dataset = SimpleCaptionDataset(args.dataset_path, processor, split="train")
    val_dataset = SimpleCaptionDataset(args.dataset_path, processor, split="valid")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
    
    # Set up optimizer and scheduler
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    
    # Enable gradient checkpointing to save memory
    if hasattr(model.language_model, "gradient_checkpointing_enable"):
        print("Enabling gradient checkpointing...")
        model.language_model.gradient_checkpointing_enable()
    
    # Training loop
    print("Starting training...")
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Training
        model.train()
        train_loss = 0
        train_steps = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            # Move batch to device
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            # Track loss
            train_loss += loss.item()
            train_steps += 1
            
            # Print every 5 steps
            if train_steps % 5 == 0:
                print(f"Step {train_steps}, Loss: {loss.item():.6f}")
                
            # Free up memory
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Calculate average training loss
        avg_train_loss = train_loss / train_steps
        print(f"Average training loss: {avg_train_loss:.6f}")
        
        # Validation
        model.eval()
        val_loss = 0
        val_steps = 0
        
        print("Validating...")
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                
                # Forward pass
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                loss = outputs.loss
                
                # Track loss
                val_loss += loss.item()
                val_steps += 1
        
        # Calculate average validation loss
        avg_val_loss = val_loss / val_steps
        print(f"Average validation loss: {avg_val_loss:.6f}")
        
        # Save checkpoint after each epoch
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}")
        os.makedirs(checkpoint_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(checkpoint_path, "model.pt"))
        print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save the final model
    final_model_path = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_model_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_model_path, "model.pt"))
    processor.save_pretrained(final_model_path)
    
    print("Training complete!")

if __name__ == "__main__":
    train_caption_model() 