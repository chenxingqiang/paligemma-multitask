import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from tqdm import tqdm


class PaliGemmaTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        learning_rate=2e-4,
        num_epochs=3,
        max_grad_norm=1.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        caption_loss_weight=0.1
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm
        self.caption_loss_weight = caption_loss_weight

        # 优化器和学习率调度器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # 梯度缩放器用于混合精度训练 - 仅在GPU上启用
        self.use_amp = device.type == "cuda"
        if self.use_amp:
            self.scaler = GradScaler()
            print("Mixed precision training enabled")
        else:
            self.scaler = None
            print("Using full precision (FP32) training on CPU")
        
        # 追踪最佳验证损失
        self.best_val_loss = float('inf')
    
    def train_step(self, batch):
        """单个训练步骤"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # 将数据移到设备
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        # 根据是否启用AMP选择不同的训练方式
        if not self.use_amp:
            # 标准精度训练 (CPU or full precision)
            outputs = self.model(
                input_ids=batch.get("input_ids"),
                attention_mask=batch.get("attention_mask"),
                pixel_values=batch.get("pixel_values"),
                boxes=batch.get("boxes"),
                classes=batch.get("classes"),
                labels=batch.get("labels") if "labels" in batch else None,
                caption_loss_weight=self.caption_loss_weight
            )
            loss = outputs["loss"]
            
            # 直接反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # 更新权重
            self.optimizer.step()
        else:
            # 使用混合精度 (GPU)
            with autocast(device_type='cuda'):
                outputs = self.model(
                    input_ids=batch.get("input_ids"),
                    attention_mask=batch.get("attention_mask"),
                    pixel_values=batch.get("pixel_values"),
                    boxes=batch.get("boxes"),
                    classes=batch.get("classes"),
                    labels=batch.get("labels") if "labels" in batch else None,
                    caption_loss_weight=self.caption_loss_weight
                )
                loss = outputs["loss"]
            
            # 缩放损失，反向传播，更新权重
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        # 返回各损失组件
        return {
            "total_loss": loss.item(),
            "language_loss": outputs["language_loss"].item() if isinstance(outputs["language_loss"], torch.Tensor) else 0.0,
            "detection_loss": outputs["detection_loss"].item() if isinstance(outputs["detection_loss"], torch.Tensor) else 0.0,
            "caption_loss": outputs.get("caption_loss", torch.tensor(0.0)).item() if "caption_loss" in outputs else 0.0
        }
    
    @torch.no_grad()
    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        language_loss_sum = 0
        detection_loss_sum = 0
        caption_loss_sum = 0
        num_batches = 0
        
        progress_bar = tqdm(self.val_loader, desc="Validation")
        
        for batch in progress_bar:
            # 将数据移到设备
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # 前向传播 - 根据是否使用AMP决定
            if not self.use_amp:
                outputs = self.model(
                    input_ids=batch.get("input_ids"),
                    attention_mask=batch.get("attention_mask"),
                    pixel_values=batch.get("pixel_values"),
                    boxes=batch.get("boxes"),
                    classes=batch.get("classes"),
                    labels=batch.get("labels") if "labels" in batch else None,
                    caption_loss_weight=self.caption_loss_weight
                )
            else:
                with autocast(device_type='cuda'):
                    outputs = self.model(
                        input_ids=batch.get("input_ids"),
                        attention_mask=batch.get("attention_mask"),
                        pixel_values=batch.get("pixel_values"),
                        boxes=batch.get("boxes"),
                        classes=batch.get("classes"),
                        labels=batch.get("labels") if "labels" in batch else None,
                        caption_loss_weight=self.caption_loss_weight
                    )
            
            # 记录损失
            total_loss += outputs["loss"].item()
            language_loss_sum += outputs["language_loss"].item() if isinstance(outputs["language_loss"], torch.Tensor) else 0.0
            detection_loss_sum += outputs["detection_loss"].item() if isinstance(outputs["detection_loss"], torch.Tensor) else 0.0
            caption_loss_sum += outputs.get("caption_loss", torch.tensor(0.0)).item() if "caption_loss" in outputs else 0.0
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                "val_loss": total_loss / num_batches
            })
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        avg_language_loss = language_loss_sum / num_batches
        avg_detection_loss = detection_loss_sum / num_batches
        avg_caption_loss = caption_loss_sum / num_batches
        
        return avg_loss, avg_language_loss, avg_detection_loss, avg_caption_loss
    
    def train(self):
        """训练模型"""
        print(f"Training on device: {self.device}")
        print(f"Caption loss weight: {self.caption_loss_weight}")
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # 训练
            print("Training:")
            total_loss = 0
            language_loss_sum = 0
            detection_loss_sum = 0
            caption_loss_sum = 0
            num_batches = 0
            
            progress_bar = tqdm(self.train_loader, desc=f"Training")
            
            for batch in progress_bar:
                losses = self.train_step(batch)
                
                # 记录损失
                total_loss += losses["total_loss"]
                language_loss_sum += losses["language_loss"]
                detection_loss_sum += losses["detection_loss"]
                caption_loss_sum += losses["caption_loss"]
                num_batches += 1
                
                # 更新进度条
                progress_bar.set_postfix({
                    "loss": total_loss / num_batches,
                    "lang_loss": language_loss_sum / num_batches,
                    "det_loss": detection_loss_sum / num_batches,
                    "cap_loss": caption_loss_sum / num_batches
                })
            
            # 计算平均训练损失
            avg_train_loss = total_loss / num_batches
            avg_train_language_loss = language_loss_sum / num_batches
            avg_train_detection_loss = detection_loss_sum / num_batches
            avg_train_caption_loss = caption_loss_sum / num_batches
            
            # 验证
            print("Validating:")
            avg_val_loss, avg_val_language_loss, avg_val_detection_loss, avg_val_caption_loss = self.validate()
            
            # 打印结果
            print(f"Train Loss: {avg_train_loss:.4f} | "
                  f"Lang Loss: {avg_train_language_loss:.4f} | "
                  f"Det Loss: {avg_train_detection_loss:.4f} | "
                  f"Cap Loss: {avg_train_caption_loss:.4f}")
            print(f"Val Loss: {avg_val_loss:.4f} | "
                  f"Lang Loss: {avg_val_language_loss:.4f} | "
                  f"Det Loss: {avg_val_detection_loss:.4f} | "
                  f"Cap Loss: {avg_val_caption_loss:.4f}")
            
            # 如果验证损失更好，保存模型
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                checkpoint_path = f"checkpoints/model_epoch_{epoch + 1}.pt"
                os.makedirs("checkpoints", exist_ok=True)
                self.save_model(checkpoint_path)
                print(f"New best model saved to {checkpoint_path}")
    
    def save_model(self, path):
        """保存模型权重"""
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        
        self.model.save_pretrained(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """加载模型权重"""
        self.model.load_pretrained(path)
        print(f"Model loaded from {path}")
