from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Basic training parameters
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 10
    
    # Loss weights
    detection_weight: float = 1.0
    multimodal_weight: float = 1.0
    
    # Model parameters
    num_classes: int = 10
    max_objects: int = 10
    hidden_size: int = 768
    seqlen: int = 128
    
    # Optimizer parameters
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Evaluation parameters
    eval_steps: int = 100
    save_steps: int = 1000
    
    # Early stopping
    patience: int = 3
    min_delta: float = 0.001
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.num_epochs > 0, "Number of epochs must be positive"
        assert self.detection_weight >= 0, "Detection weight must be non-negative"
        assert self.multimodal_weight >= 0, "Multimodal weight must be non-negative"
        assert self.num_classes > 0, "Number of classes must be positive"
        assert self.max_objects > 0, "Maximum number of objects must be positive"
        assert self.hidden_size > 0, "Hidden size must be positive"
        assert self.seqlen > 0, "Sequence length must be positive"
        assert self.weight_decay >= 0, "Weight decay must be non-negative"
        assert self.warmup_steps >= 0, "Warmup steps must be non-negative"
        assert self.max_grad_norm > 0, "Maximum gradient norm must be positive"
        assert self.eval_steps > 0, "Evaluation steps must be positive"
        assert self.save_steps > 0, "Save steps must be positive"
        assert self.patience > 0, "Patience must be positive"
        assert self.min_delta > 0, "Minimum delta must be positive" 