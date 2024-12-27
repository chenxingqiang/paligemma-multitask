import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
import re

class PaliGemmaDataset(Dataset):
    def __init__(self, dataset_path, processor, split="train"):
        self.dataset_path = dataset_path
        self.processor = processor
        self.split = split
        
        # 加载标注文件
        if split == "train":
            annotation_file = os.path.join(dataset_path, "_annotations.train1.jsonl")
        else:
            annotation_file = os.path.join(dataset_path, "_annotations.valid1.jsonl")
            
        self.annotations = []
        with open(annotation_file, "r") as f:
            for line in f:
                self.annotations.append(json.loads(line))
    
    def _extract_boxes_and_classes(self, text):
        """从文本描述中提取边界框和类别信息"""
        # 初始化空列表
        boxes = []
        classes = []
        
        # 定义类别映射
        class_map = {
            "void": 0,
            "crack": 1
        }
        
        # 定义位��映射
        position_map = {
            "center": [0.5, 0.5],
            "top": [0.5, 0.25],
            "bottom": [0.5, 0.75],
            "left": [0.25, 0.5],
            "right": [0.75, 0.5],
            "top left": [0.25, 0.25],
            "top right": [0.75, 0.25],
            "bottom left": [0.25, 0.75],
            "bottom right": [0.75, 0.75],
            "lower left": [0.25, 0.75],
            "lower right": [0.75, 0.75],
            "center towards the far left": [0.2, 0.5],
            "center towards the far right": [0.8, 0.5]
        }
        
        # 查找所有损伤描述
        void_matches = re.finditer(r"void.*?(?=\.|$)", text.lower())
        crack_matches = re.finditer(r"crack.*?(?=\.|$)", text.lower())
        
        # 处理 void
        for match in void_matches:
            desc = match.group()
            for pos, coords in position_map.items():
                if pos in desc:
                    # 创建一个简单的边界框 [x1, y1, x2, y2]
                    x, y = coords
                    box = [
                        max(0.0, x - 0.1),  # x1
                        max(0.0, y - 0.1),  # y1
                        min(1.0, x + 0.1),  # x2
                        min(1.0, y + 0.1)   # y2
                    ]
                    boxes.append(box)
                    classes.append(class_map["void"])
                    break
        
        # 处理 crack
        for match in crack_matches:
            desc = match.group()
            for pos, coords in position_map.items():
                if pos in desc:
                    # 创建一个简单的边界框 [x1, y1, x2, y2]
                    x, y = coords
                    box = [
                        max(0.0, x - 0.1),  # x1
                        max(0.0, y - 0.1),  # y1
                        min(1.0, x + 0.1),  # x2
                        min(1.0, y + 0.1)   # y2
                    ]
                    boxes.append(box)
                    classes.append(class_map["crack"])
                    break
        
        # 如果没有找到任何边界框，添加一个默认的
        if not boxes:
            boxes = [[0.4, 0.4, 0.6, 0.6]]
            classes = [0]  # 默认为 void
        
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(classes, dtype=torch.long)
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        # 加载图像
        image_path = os.path.join(self.dataset_path, ann["image"])
        image = Image.open(image_path).convert("RGB")
        
        # 处理文本 - 添加图像标记
        text = "<image> " + ann["prefix"] + " " + ann["suffix"]
        
        # 处理图像和文本
        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # 移除批次维度
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        # 从文本描述中提取边界框和类别
        boxes, classes = self._extract_boxes_and_classes(ann["suffix"])
        inputs["boxes"] = boxes
        inputs["classes"] = classes
        
        return inputs

def collate_fn(batch):
    """处理变长序列的批处理函数"""
    # 收集所有键
    keys = batch[0].keys()
    collated = {}
    
    for key in keys:
        if key in ["input_ids", "attention_mask"]:
            # 对序列进行填充
            tensors = [item[key] for item in batch]
            collated[key] = pad_sequence(tensors, batch_first=True)
        elif key == "pixel_values":
            # 图像已经是固定大小
            collated[key] = torch.stack([item[key] for item in batch])
        elif key in ["boxes", "classes"]:
            # 检测目标 - 使用填充
            max_boxes = max(item[key].size(0) for item in batch)
            padded_boxes = []
            padded_classes = []
            
            for item in batch:
                if key == "boxes":
                    num_boxes = item[key].size(0)
                    padded = torch.zeros((max_boxes, 4), dtype=torch.float32)
                    padded[:num_boxes] = item[key]
                    padded_boxes.append(padded)
                else:  # classes
                    num_classes = item[key].size(0)
                    padded = torch.full((max_boxes,), -1, dtype=torch.long)  # 使用 -1 作为填充值
                    padded[:num_classes] = item[key]
                    padded_classes.append(padded)
            
            if key == "boxes":
                collated[key] = torch.stack(padded_boxes)
            else:
                collated[key] = torch.stack(padded_classes)
        else:
            # 其他键直接堆叠
            try:
                collated[key] = torch.stack([item[key] for item in batch])
            except:
                collated[key] = [item[key] for item in batch]
    
    return collated

def create_data_loaders(dataset_path, processor, batch_size=4, num_workers=2):
    """创建训练和验证数据加载器"""
    # 创建训练集
    train_dataset = PaliGemmaDataset(
        dataset_path=dataset_path,
        processor=processor,
        split="train"
    )
    
    # 创建验证集
    val_dataset = PaliGemmaDataset(
        dataset_path=dataset_path,
        processor=processor,
        split="valid"
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader 