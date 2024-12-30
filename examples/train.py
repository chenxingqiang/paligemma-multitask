import argparse
import torch
import os
import json
import shutil
from pathlib import Path
from paligemma_multitask.model import create_model
from paligemma_multitask.data import create_data_loaders
from paligemma_multitask.training import PaliGemmaTrainer
from transformers import AutoProcessor
from huggingface_hub import login, HfApi

def save_model_files(model_path, hub_model_id, processor, merged_model):
    """保存所有模型相关文件到Hub"""
    # 创建临时目录
    tmp_path = Path("tmp_model_files")
    tmp_path.mkdir(exist_ok=True)
    
    # 保存多任务模型配置
    config = {
        "model_type": "paligemma-multitask",
        "tasks": [
            {
                "name": "object_detection",
                "type": "detection",
                "labels": ["void", "crack"],
                "position_map": {
                    "center": [0.5, 0.5],
                    "top": [0.5, 0.25],
                    "bottom": [0.5, 0.75],
                    "left": [0.25, 0.5],
                    "right": [0.75, 0.5],
                    "top left": [0.25, 0.25],
                    "top right": [0.75, 0.25],
                    "bottom left": [0.25, 0.75],
                    "bottom right": [0.75, 0.75]
                }
            },
            {
                "name": "text_generation",
                "type": "language_modeling",
                "model_type": "causal_lm",
                "max_length": 512,
                "generation_config": {
                    "max_length": 512,
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            },
            {
                "name": "visual_question_answering",
                "type": "vqa",
                "model_type": "vision2seq",
                "max_length": 512
            }
        ],
        "framework": "pytorch",
        "base_model": "google/paligemma-3b-mix-224",
        "training_config": {
            "lora_config": {
                "r": 8,
                "lora_alpha": 32,
                "target_modules": [
                    "q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
                "lora_dropout": 0.1
            }
        },
        "processor_config": {
            "type": "paligemma",
            "image_size": 224,
            "max_length": 512
        }
    }
    
    # 保存配置文件
    with open(tmp_path / "config.json", "w", encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # 保存模型和处理器
    merged_model.save_pretrained(tmp_path)
    processor.save_pretrained(tmp_path)
    
    # 保存模型卡片信息
    model_card = f"""---
language:
- en
- zh
tags:
- paligemma
- multitask
- object-detection
- text-generation
- visual-question-answering
datasets:
- custom
license: mit

model-index:
- name: {hub_model_id}
  results:
  - task:
      type: object-detection
    dataset:
      type: custom
      name: Defect Detection
  - task:
      type: text-generation
    dataset:
      type: custom
      name: Text Description
  - task:
      type: visual-question-answering
    dataset:
      type: custom
      name: Visual QA
---

# PaliGemma Multitask Model

This is a multitask model based on PaliGemma that can perform:
1. Object Detection (defect detection)
2. Text Generation (defect description)
3. Visual Question Answering

## Model Description

The model is fine-tuned from google/paligemma-3b-mix-224 using LoRA technique.

## Usage

```python
from transformers import AutoProcessor, AutoModelForVision2Seq

# Load model and processor
processor = AutoProcessor.from_pretrained("{hub_model_id}")
model = AutoModelForVision2Seq.from_pretrained("{hub_model_id}")

# Process image and text
inputs = processor(images=image, text=text, return_tensors="pt")

# Get predictions
outputs = model(**inputs)
```

## Training Procedure

The model was trained on a custom dataset with multiple tasks:
- Object detection for defect localization
- Text generation for defect description
- Visual question answering for interactive analysis

## Limitations and Biases

This model is specifically trained for defect detection and analysis tasks.
"""
    
    # 保存模型卡片
    with open(tmp_path / "README.md", "w", encoding='utf-8') as f:
        f.write(model_card)
    
    # 上传到Hub
    api = HfApi()
    api.upload_folder(
        folder_path=str(tmp_path),
        repo_id=hub_model_id,
        repo_type="model"
    )
    
    # 清理临时文件
    shutil.rmtree(tmp_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="google/paligemma-3b-mix-224")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--hub_model_id", type=str, default='xingqiang/paligemma-multitask-detector', help="Model ID for uploading to Hugging Face Hub")
    args = parser.parse_args()
    
    # 从环境变量获取 Hugging Face token
    hf_token = os.getenv('HF')
    if hf_token:
        print("Found Hugging Face token in environment variables")
        login(token=hf_token)
    else:
        print("Warning: No Hugging Face token found in environment variables (HF)")
    
    # 创建处理器
    processor = AutoProcessor.from_pretrained(args.model_id)
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(
        dataset_path=args.dataset_path,
        processor=processor,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # 创建模型
    model = create_model(model_id=args.model_id)
    
    # 创建训练器
    trainer = PaliGemmaTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs
    )
    
    # 开始训练
    trainer.train()
    
    # 保存最终模型到本地
    trainer.save_model("final_model.pt")
    
    # 合并 LoRA 权重并保存
    merged_model = model.merge_and_unload()
    
    # 如果提供了 hub_model_id，保存到 Hugging Face Hub
    if args.hub_model_id:
        print(f"Pushing model to Hugging Face Hub: {args.hub_model_id}")
        save_model_files("merged_model", args.hub_model_id, processor, merged_model)
        print("Model and configuration files successfully pushed to Hub!")

if __name__ == "__main__":
    main() 