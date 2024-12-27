import argparse
import torch
from paligemma_multitask.model import create_model
from paligemma_multitask.data import create_data_loaders
from paligemma_multitask.training import PaliGemmaTrainer
from transformers import AutoProcessor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="google/paligemma-3b-mix-224")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()
    
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
    
    # 保存最终模型
    trainer.save_model("final_model.pt")
    
    # 合并 LoRA 权重并保存
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained("merged_model")

if __name__ == "__main__":
    main() 