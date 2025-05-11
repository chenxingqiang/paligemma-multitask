import os
import argparse
import torch
from transformers import AutoProcessor

from paligemma_multitask.model import create_model
from paligemma_multitask.data import create_data_loaders
from paligemma_multitask.training import PaliGemmaTrainer

def main():
    parser = argparse.ArgumentParser(description="Fine-tune PaLI-GEMMA with multitask capabilities")
    
    # Model and data arguments
    parser.add_argument("--model_name", type=str, default="google/paligemma-3b-mix-224", 
                        help="Model name or path")
    parser.add_argument("--dataset_path", type=str, default="dataset",
                        help="Path to the dataset directory")
    parser.add_argument("--annotation_type", type=str, default="multimodal", choices=["multimodal", "original"],
                        help="Type of annotations to use (multimodal or original)")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="Output directory for the fine-tuned model")
    parser.add_argument("--debug", action="store_true",
                        help="Run in debug mode with limited data")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for clipping")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of data loader workers")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")
    parser.add_argument("--caption_loss_weight", type=float, default=0.1,
                        help="Weight for the caption generation loss")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Disable fp16 when on CPU
    if device == "cpu" and args.fp16:
        print("Warning: Mixed precision (fp16) is not available on CPU. Disabling fp16.")
        args.fp16 = False
    
    # Load processor
    print(f"Loading processor from {args.model_name}...")
    processor = AutoProcessor.from_pretrained(args.model_name)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        dataset_path=args.dataset_path,
        processor=processor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        debug_mode=args.debug,
        annotation_type=args.annotation_type
    )
    
    # Create model
    print(f"Creating model from {args.model_name}...")
    # Disable LoRA in debug mode for simpler training
    use_lora = False if args.debug else True
    # Pass fp16 flag to create_model
    model = create_model(args.model_name, num_classes=2, apply_lora=use_lora, debug_mode=args.debug)
    
    # Create trainer
    trainer = PaliGemmaTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_grad_norm=args.max_grad_norm,
        device=device,
        caption_loss_weight=args.caption_loss_weight
    )
    
    # Train model
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    final_model_path = os.path.join(args.output_dir, "final_model.pt")
    trainer.save_model(final_model_path)
    print(f"Final model saved to {final_model_path}")

if __name__ == "__main__":
    main() 