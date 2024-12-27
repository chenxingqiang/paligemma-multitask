import os
import jax
import numpy as np
import sentencepiece
from paligemma_multitask.config import TrainingConfig
from paligemma_multitask.training import JointTrainer
import paligemma
from paligemma_multitask.utils.environment import setup_environment

def main():
    # Setup environment
    setup_environment()
    
    # Configuration
    config = TrainingConfig(
        train_examples=170,  # Adjust based on your dataset size
        epochs=30,
        batch_size=16,
        learning_rate=0.01,
        detection_weight=0.5,
        multimodal_weight=0.5
    )
    
    # Initialize model and tokenizer
    model = paligemma.Model(**config.model_config())
    tokenizer = sentencepiece.SentencePieceProcessor()
    tokenizer.load("model/paligemma_tokenizer.model")
    
    # Create trainer
    trainer = JointTrainer(model, tokenizer, config)
    
    # Setup data iterators
    from paligemma_multitask.data import create_data_iterators
    train_iterator, val_iterator = create_data_iterators(
        dataset_path="dataset/p-1.v1i.paligemma-multimodal/dataset",
        batch_size=config.batch_size,
        seqlen=config.seqlen
    )
    
    # Train model
    print("Starting joint training...")
    params, metrics = trainer.train(train_iterator, val_iterator)
    
    # Save model
    print("Training complete. Saving model...")
    output_dir = "model/joint_finetuned"
    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        os.path.join(output_dir, "model_weights.npz"),
        **{k: v for k, v in params.items()}
    )
    
    print("Model saved successfully!")

if __name__ == "__main__":
    main() 