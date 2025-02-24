import os
import json
import torch
import torch.nn as nn
from transformers import AutoModelForVision2Seq
from peft import get_peft_model, LoraConfig

class PaliGemmaMultitaskModel(nn.Module):
    def __init__(self, model_id, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        
        # Load base model
        self.base_model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Get device from base model
        self.device = next(self.base_model.parameters()).device
        
        # Expose config and generation config
        self.config = self.base_model.config
        self.generation_config = self.base_model.generation_config
        
        # Get hidden size
        hidden_size = self.base_model.config.hidden_size
        
        # Add detection head
        self.detection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, dtype=torch.float16),
            nn.ReLU(),
            nn.Linear(hidden_size, 4, dtype=torch.float16)  # [x1, y1, x2, y2]
        ).to(self.device)
        
        # Add classification head
        self.class_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, dtype=torch.float16),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes, dtype=torch.float16)
        ).to(self.device)
    
    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, **kwargs):
        # Forward pass
        outputs = self.base_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True
        )
        
        # Get features
        features = outputs.hidden_states[-1]  # Use all token features
        batch_size = features.size(0)
        
        # Generate detection boxes
        boxes = self.detection_head(features[:, 0])  # Use [CLS] token
        boxes = torch.sigmoid(boxes)  # Normalize to [0, 1]
        
        # Generate class predictions
        class_logits = self.class_head(features[:, 0])  # Use [CLS] token
        
        return {
            'boxes': boxes,
            'class_logits': class_logits
        }
    
    def prepare_inputs_for_generation(self, *args, **kwargs):
        """Support text generation"""
        return self.base_model.prepare_inputs_for_generation(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        """Support text generation"""
        return self.base_model.generate(*args, **kwargs)

    def save_pretrained(self, save_directory):
        """Save model weights and config"""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save base model
        self.base_model.save_pretrained(save_directory)
        
        # Save detection and classification heads
        torch.save({
            'detection_head': self.detection_head.state_dict(),
            'class_head': self.class_head.state_dict()
        }, os.path.join(save_directory, "custom_heads.bin"))
        
        # Save config
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump({"num_classes": self.num_classes}, f)
        
        print(f"Model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, save_directory):
        """Load model from saved directory"""
        # Load config
        with open(os.path.join(save_directory, "config.json"), "r") as f:
            config = json.load(f)
        num_classes = config["num_classes"]
        
        # Initialize model
        model = cls(model_id=save_directory, num_classes=num_classes)
        
        # Load base model
        model.base_model = AutoModelForVision2Seq.from_pretrained(save_directory)
        
        # Load detection and classification heads
        heads_weights = torch.load(os.path.join(save_directory, "custom_heads.bin"))
        model.detection_head.load_state_dict(heads_weights['detection_head'])
        model.class_head.load_state_dict(heads_weights['class_head'])
        
        return model


def create_model(model_id, num_classes=2):
    """Create and configure model"""
    # Create base model
    model = PaliGemmaMultitaskModel(model_id, num_classes)
    
    # LoRA config
    peft_config = LoraConfig(
        r=8,  # LoRA rank
        lora_alpha=32,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    
    return model 
