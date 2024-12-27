import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from typing import Dict, Any, Iterator

from .utils.metrics import calculate_detection_metrics
from .config import TrainingConfig

class PaliGemmaTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        learning_rate=2e-4,
        num_epochs=3,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        
        # 移动模型到设备
        self.model = self.model.to(self.device)
        
        # 设置优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )
        
        # 损失函数
        self.detection_loss = nn.MSELoss().to(device=self.device)
        self.class_loss = nn.CrossEntropyLoss().to(device=self.device)
        
    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        
        # 移动数据到设备并转换数据类型
        batch = {
            k: (v.to(device=self.device) if k == 'pixel_values' or k == 'boxes' 
               else v.to(device=self.device)) 
            for k, v in batch.items()
        }
        
        # 前向传播
        outputs = self.model(
            pixel_values=batch['pixel_values'],
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        
        # 计算检测损失 - 只对第一个边界框计算损失
        detection_loss = self.detection_loss(
            outputs['boxes'].float(),
            batch['boxes'][:, 0].float()
        )
        
        # 计算分类损失 - 只对第一个类别计算损失
        class_loss = self.class_loss(
            outputs['class_logits'].float(),
            batch['classes'][:, 0]
        )
        
        # 计算总损失
        total_loss = detection_loss + class_loss
        
        # 反向传播
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'loss': total_loss.item(),
            'detection_loss': detection_loss.item(),
            'class_loss': class_loss.item()
        }
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for batch in self.val_loader:
            # 移动数据到设备并转换数据类型
            batch = {
                k: (v.to(device=self.device) if k == 'pixel_values' or k == 'boxes' 
                   else v.to(device=self.device)) 
                for k, v in batch.items()
            }
            
            # 前向传播
            outputs = self.model(
                pixel_values=batch['pixel_values'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            # 计算检测损失 - 只对第一个边界框计算损失
            detection_loss = self.detection_loss(
                outputs['boxes'].float(),
                batch['boxes'][:, 0].float()
            )
            
            # 计算分类损失 - 只对第一个类别计算损失
            class_loss = self.class_loss(
                outputs['class_logits'].float(),
                batch['classes'][:, 0]
            )
            
            total_loss += (
                detection_loss.item() +
                class_loss.item()
            )
            num_batches += 1
        
        return total_loss / num_batches
    
    def train(self):
        """训练模型"""
        print(f"Training on device: {self.device}")
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # 训练循环
            train_losses = []
            progress_bar = tqdm(self.train_loader, desc="Training")
            
            for batch in progress_bar:
                losses = self.train_step(batch)
                train_losses.append(losses)
                
                # 更新进度条
                avg_loss = sum(l['loss'] for l in train_losses) / len(train_losses)
                progress_bar.set_postfix(loss=f"{avg_loss:.4f}")
            
            # 验证
            val_loss = self.validate()
            print(f"Validation loss: {val_loss:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model("best_model.pt")
        
        print("\nTraining completed!")
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path) 