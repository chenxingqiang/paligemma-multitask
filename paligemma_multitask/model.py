import torch
import torch.nn as nn
from transformers import AutoModelForVision2Seq
from peft import get_peft_model, LoraConfig

class PaliGemmaMultitaskModel(nn.Module):
    def __init__(self, model_id, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        
        # 加载基础模型
        self.base_model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 暴露配置
        self.config = self.base_model.config
        
        # 获取隐藏层维度
        hidden_size = self.base_model.config.hidden_size
        
        # 添加检测头
        self.detection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, dtype=torch.float16),
            nn.ReLU(),
            nn.Linear(hidden_size, 4, dtype=torch.float16)  # [x1, y1, x2, y2]
        )
        
        # 添加分类头
        self.class_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, dtype=torch.float16),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes, dtype=torch.float16)
        )
    
    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, **kwargs):
        # 前向传播
        outputs = self.base_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True
        )
        
        # 获取特征
        features = outputs.hidden_states[-1]  # 使用所有token的特征
        batch_size = features.size(0)
        
        # 生成检测框
        boxes = self.detection_head(features[:, 0])  # 使用 [CLS] token
        boxes = torch.sigmoid(boxes)  # 归一化到 [0, 1]
        
        # 生成类别预测 - 为每个边界框生成预测
        class_logits = self.class_head(features[:, 0])  # 使用 [CLS] token
        
        return {
            'boxes': boxes,
            'class_logits': class_logits
        }
    
    def prepare_inputs_for_generation(self, *args, **kwargs):
        """支持文本生成"""
        return self.base_model.prepare_inputs_for_generation(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        """支持文本生成"""
        return self.base_model.generate(*args, **kwargs)

def create_model(model_id, num_classes=2):
    """创建并配置模型"""
    # 创建基础模型
    model = PaliGemmaMultitaskModel(model_id, num_classes)
    
    # LoRA 配置
    peft_config = LoraConfig(
        r=8,  # LoRA 秩
        lora_alpha=32,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 应用 LoRA
    model = get_peft_model(model, peft_config)
    
    return model 