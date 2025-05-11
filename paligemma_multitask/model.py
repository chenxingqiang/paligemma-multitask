import os
import json
import torch
import torch.nn as nn
from transformers import AutoModelForVision2Seq
from peft import get_peft_model, LoraConfig

class PaliGemmaMultitaskModel(nn.Module):
    def __init__(self, model_id, num_classes=2, use_fp16=True):
        super().__init__()
        self.num_classes = num_classes
        
        # Load base model
        self.base_model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if use_fp16 else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Get device from base model
        self.device = next(self.base_model.parameters()).device
        
        # Expose config and generation config
        self.config = self.base_model.config
        self.generation_config = self.base_model.generation_config
        
        # Object detection head
        hidden_size = self.base_model.config.hidden_size
        self.detection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4 + self.num_classes)  # 4 for box coordinates, num_classes for class logits
        )
        
    def forward(
        self, 
        input_ids=None, 
        attention_mask=None, 
        pixel_values=None, 
        boxes=None, 
        classes=None, 
        labels=None, 
        inputs_embeds=None,
        caption_loss_weight=0.1,
        **kwargs
    ):
        # 确保输入标签的尺寸与input_ids匹配
        if labels is not None and input_ids is not None:
            if labels.size() != input_ids.size():
                # 调整labels大小以匹配input_ids
                if labels.dim() == 1:
                    # 如果labels是一维的，但input_ids是二维的
                    labels = labels.unsqueeze(0)
                
                # 处理batch大小和序列长度不匹配的情况
                if labels.size(0) != input_ids.size(0):
                    # 重复或截断batch维度
                    if labels.size(0) < input_ids.size(0):
                        labels = labels.repeat(input_ids.size(0), 1)
                    else:
                        labels = labels[:input_ids.size(0), :]
                
                # 处理序列长度不匹配的情况
                if labels.size(1) != input_ids.size(1):
                    if labels.size(1) < input_ids.size(1):
                        # 填充序列长度
                        padding = torch.full(
                            (labels.size(0), input_ids.size(1) - labels.size(1)),
                            -100,  # 使用-100作为填充值，模型会忽略这部分
                            dtype=labels.dtype,
                            device=labels.device
                        )
                        labels = torch.cat([labels, padding], dim=1)
                    else:
                        # 截断序列长度
                        labels = labels[:, :input_ids.size(1)]
        
        # 运行基础模型
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs
        )
        
        # 获取最后一个隐藏状态
        last_hidden_state = outputs.last_hidden_state
        
        # 第一个token通常对应于整个序列的表示
        cls_token = last_hidden_state[:, 0, :]
        
        # 应用检测头
        detection_output = self.detection_head(cls_token)
        
        # 分离边界框和类别预测
        box_preds = detection_output[:, :4]
        class_logits = detection_output[:, 4:]
        
        # 计算检测损失（如果提供了ground truth）
        detection_loss = 0
        if boxes is not None and classes is not None:
            # 边界框损失 (L1 loss)
            box_loss = nn.functional.l1_loss(box_preds, boxes[:, 0, :]) 
            
            # 分类损失 (CrossEntropyLoss)
            # 移除填充的类别 (-1)
            valid_mask = (classes[:, 0] != -1)
            valid_classes = classes[valid_mask, 0]
            valid_logits = class_logits[valid_mask]
            
            if valid_classes.numel() > 0:
                class_loss = nn.functional.cross_entropy(valid_logits, valid_classes)
            else:
                class_loss = torch.tensor(0.0, device=self.device)
            
            detection_loss = box_loss + class_loss
        
        # 获取语言建模损失（如果有）
        language_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=self.device)
        
        # 初始化caption_loss
        caption_loss = torch.tensor(0.0, device=self.device)
        
        # 如果提供了标签，计算caption_loss
        if labels is not None:
            # caption_loss就是language_loss的一部分，使用权重控制其影响
            caption_loss = language_loss * caption_loss_weight
            # 调整language_loss的权重为1.0 - caption_loss_weight
            language_loss = language_loss * (1.0 - caption_loss_weight)
        
        # 总损失是语言建模损失和检测损失的组合
        if labels is not None:
            total_loss = language_loss + detection_loss + caption_loss
        else:
            total_loss = detection_loss
        
        # 返回结果字典
        result = {
            "loss": total_loss,
            "language_loss": language_loss,
            "detection_loss": detection_loss,
            "caption_loss": caption_loss,
            "box_preds": box_preds,
            "class_logits": class_logits,
            "logits": outputs.logits if hasattr(outputs, "logits") else None
        }
        
        return result
    
    def prepare_inputs_for_generation(self, *args, **kwargs):
        """支持文本生成所需的方法"""
        return self.base_model.prepare_inputs_for_generation(*args, **kwargs)
    
    def generate(self, input_ids=None, attention_mask=None, pixel_values=None, **kwargs):
        """使用基础模型生成文本"""
        return self.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            **kwargs
        )

    def save_pretrained(self, save_directory):
        """Save model weights and config"""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save base model
        self.base_model.save_pretrained(save_directory)
        
        # Save detection and classification heads
        torch.save({
            'detection_head': self.detection_head.state_dict()
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
        
        return model


def create_model(model_id, num_classes=2, apply_lora=True, debug_mode=False):
    """创建并配置模型"""
    # 使用fp32在CPU上，fp16在GPU上
    use_fp16 = torch.cuda.is_available() and not debug_mode
    
    if debug_mode:
        # 调试模式下使用简化设置，无LoRA，使用fp32
        model = PaliGemmaMultitaskModel(
            model_id, 
            num_classes=num_classes,
            use_fp16=False  # 调试模式下使用fp32
        )
        print("Debug mode enabled: Using simplified model without LoRA, FP32 precision")
        return model
    
    # 标准模式
    model = PaliGemmaMultitaskModel(model_id, num_classes, use_fp16=use_fp16)
    
    # 如果需要，应用LoRA
    if apply_lora:
        # 配置LoRA
        lora_config = LoraConfig(
            r=16,  # LoRA注意力尺寸
            lora_alpha=32,  # LoRA的alpha参数
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 目标模块
            lora_dropout=0.05,  # LoRA的丢弃率
            bias="none",  # 偏置参数
            task_type="CAUSAL_LM"  # 任务类型
        )
        
        # 将LoRA应用于模型
        model = get_peft_model(model, lora_config)
        
        # 打印LoRA参数信息
        model.print_trainable_parameters()
    
    return model 
