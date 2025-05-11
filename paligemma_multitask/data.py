import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
import re

class PaliGemmaDataset(Dataset):
    def __init__(self, dataset_path, processor, split="train", annotation_type="multimodal"):
        self.dataset_path = dataset_path
        self.processor = processor
        self.split = split
        self.annotation_type = annotation_type
        
        # 加载标注文件
        if split == "train":
            if annotation_type == "multimodal":
                # Try loading the multimodal annotations
                annotations_dir = os.path.join(dataset_path, "annotations", "multimodal")
                annotation_file = os.path.join(annotations_dir, "annotations.train.jsonl")
                print(f"Using multimodal annotations from {annotation_file}")
            else:
                # Use the original annotations
                annotations_dir = os.path.join(dataset_path, "annotations", "p-1.v1i.paligemma")
                annotation_file = os.path.join(annotations_dir, "annotations.train.jsonl")
                print(f"Using original annotations from {annotation_file}")
                
        else:  # valid or test
            if annotation_type == "multimodal":
                annotations_dir = os.path.join(dataset_path, "annotations", "multimodal")
                annotation_file = os.path.join(annotations_dir, "annotations.valid.jsonl")
                print(f"Using multimodal annotations from {annotation_file}")
            else:
                annotations_dir = os.path.join(dataset_path, "annotations", "p-1.v1i.paligemma")
                annotation_file = os.path.join(annotations_dir, "annotations.valid.jsonl")
                print(f"Using original annotations from {annotation_file}")
        
        # Check if the file exists
        if not os.path.exists(annotation_file):
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
            
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
        
        # 定义位置映射
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
            position_found = False
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
                    position_found = True
                    break
            
            # If no specific position found, use center
            if not position_found:
                x, y = 0.5, 0.5  # Default to center
                box = [
                    max(0.0, x - 0.1),  # x1
                    max(0.0, y - 0.1),  # y1
                    min(1.0, x + 0.1),  # x2
                    min(1.0, y + 0.1)   # y2
                ]
                boxes.append(box)
                classes.append(class_map["void"])
        
        # 处理 crack
        for match in crack_matches:
            desc = match.group()
            position_found = False
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
                    position_found = True
                    break
            
            # If no specific position found, use center
            if not position_found:
                x, y = 0.5, 0.5  # Default to center
                box = [
                    max(0.0, x - 0.1),  # x1
                    max(0.0, y - 0.1),  # y1
                    min(1.0, x + 0.1),  # x2
                    min(1.0, y + 0.1)   # y2
                ]
                boxes.append(box)
                classes.append(class_map["crack"])
        
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
        # For the multimodal dataset, images are in dataset/images/datasets directory
        image_path = os.path.join(self.dataset_path, "images", "datasets", ann["image"])
        
        # Check if image exists, if not try other possible locations
        if not os.path.exists(image_path):
            # Try to find in common image directories
            alternative_paths = [
                os.path.join(self.dataset_path, "images", ann["image"]),
                os.path.join(self.dataset_path, ann["image"])
            ]
            
            for path in alternative_paths:
                if os.path.exists(path):
                    image_path = path
                    break
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path} (original: {ann['image']})")
            
        # Load and resize the image to ensure consistent processing
        image = Image.open(image_path).convert("RGB")
        
        # Handle different annotation formats
        prefix = ann.get("prefix", "")
        suffix = ann.get("suffix", "")
        
        # For new annotation format, check for caption field
        caption = ann.get("caption", suffix)
        
        # Remove any existing <image> tags from text to prevent duplication
        prefix = prefix.replace("<image>", "").strip()
        suffix = suffix.replace("<image>", "").strip()
        
        # Create input text without image placeholder token - PaliGemma will add it
        prompt = f"{prefix} {suffix}".strip()
        
        # Process image and text separately to avoid mismatch
        # Process the image
        pixel_values = self.processor.image_processor(images=image, return_tensors="pt").pixel_values[0]
        
        # Process text (without <image> token)
        text_inputs = self.processor.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Create the inputs dictionary
        inputs = {
            "pixel_values": pixel_values,
            "input_ids": text_inputs.input_ids[0],
            "attention_mask": text_inputs.attention_mask[0]
        }
        
        # Extract bounding boxes and classes from the text description
        boxes, classes = self._extract_boxes_and_classes(suffix)
        inputs["boxes"] = boxes
        inputs["classes"] = classes
        
        # Prepare labels for caption generation if we have a caption
        if caption:
            # For PaliGemma, use the same input_ids as labels for language modeling
            # This is standard approach for causal language modeling
            labels = inputs["input_ids"].clone()
            
            # Set padding tokens to -100 to ignore them in loss calculation
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            
            # Store the labels in the inputs dictionary
            inputs["labels"] = labels
        
        return inputs

def collate_fn(batch):
    """处理变长序列的批处理函数"""
    # 收集所有键
    keys = batch[0].keys()
    collated = {}
    
    for key in keys:
        if key in ["input_ids", "attention_mask", "labels"]:
            # 对序列进行填充
            tensors = [item[key] for item in batch if key in item]
            if tensors:  # Only process if we have any tensors for this key
                # 确保统一长度 - 使用最大长度填充
                max_len = max(t.size(0) for t in tensors)
                padded_tensors = []
                
                for t in tensors:
                    if t.size(0) < max_len:
                        if key == "labels":
                            # 对于labels，使用-100作为填充值（通常模型会忽略这个值）
                            padding = torch.full((max_len - t.size(0),), -100, dtype=t.dtype)
                        else:
                            # 对于其他token，使用tokenizer的pad_token_id作为填充值
                            padding = torch.zeros((max_len - t.size(0),), dtype=t.dtype)
                        padded = torch.cat([t, padding], dim=0)
                    else:
                        padded = t
                    padded_tensors.append(padded)
                
                collated[key] = torch.stack(padded_tensors)
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

def create_data_loaders(dataset_path, processor, batch_size=4, num_workers=2, debug_mode=False, annotation_type="multimodal"):
    """创建训练和验证数据加载器"""
    # 创建训练集
    train_dataset = PaliGemmaDataset(
        dataset_path=dataset_path,
        processor=processor,
        split="train",
        annotation_type=annotation_type
    )
    
    # 创建验证集
    val_dataset = PaliGemmaDataset(
        dataset_path=dataset_path,
        processor=processor,
        split="valid",
        annotation_type=annotation_type
    )
    
    # 如果是调试模式，仅使用少量数据
    if debug_mode:
        # 仅使用前10个训练样本和前5个验证样本
        debug_size_train = min(10, len(train_dataset))
        debug_size_val = min(5, len(val_dataset))
        
        # 使用子集
        from torch.utils.data import Subset
        train_indices = list(range(debug_size_train))
        val_indices = list(range(debug_size_val))
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
        
        print(f"DEBUG MODE: Using {debug_size_train} training samples and {debug_size_val} validation samples")
    
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