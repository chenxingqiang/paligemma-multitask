import os
import json

print("开始执行脚本...")

# 定义分割名称映射
split_mapping = {
    "train": ["train"],
    "val": ["val", "valid"],
    "test": ["test"]
}

print("分割名称映射:", split_mapping)

for split, variants in split_mapping.items():
    print(f"\n处理 {split} 分割...")
    unified_annotations = []
    
    # 处理 JSONL 注释
    for variant in variants:
        for jsonl_variant in [f"_annotations.{variant}.jsonl", f"_annotations.{variant}1.jsonl"]:
            jsonl_path = f"annotations/{jsonl_variant}"
            print(f"检查 JSONL 文件: {jsonl_path}")
            if os.path.exists(jsonl_path):
                print(f"找到 JSONL 文件: {jsonl_path}")
                annotation_count = 0
                with open(jsonl_path, encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            line = line.strip()
                            if not line:
                                print(f"  跳过第 {line_num} 行: 空行")
                                continue
                            
                            ann = json.loads(line)
                            image_filename = ann.get("image", "")
                            
                            if not image_filename:
                                print(f"  跳过第 {line_num} 行: 没有图像文件名")
                                continue
                            
                            print(f"  处理第 {line_num} 行, 图像: {image_filename}")
                            
                            # 转换为统一格式
                            boxes = []
                            labels = []
                            caption = ann.get("prefix", "")
                            
                            if "suffix" in ann:
                                parts = ann["suffix"].split(";")
                                for part in parts:
                                    part = part.strip()
                                    if "<loc" in part:
                                        # 解析位置和标签
                                        loc_parts = part.split()
                                        if len(loc_parts) >= 2:
                                            # 提取坐标
                                            coords = []
                                            for loc in loc_parts[0].split("><"):
                                                if loc.startswith("<loc"):
                                                    try:
                                                        coords.append(int(loc[4:-1]) / 1024)
                                                    except ValueError:
                                                        continue
                                            
                                            if len(coords) == 4:
                                                boxes.append(coords)
                                                label = 0 if "void" in loc_parts[1] else 1
                                                labels.append(label)
                            
                            unified_annotations.append({
                                "image_filename": image_filename,
                                "boxes": boxes,
                                "labels": labels,
                                "caption": caption,
                                "source": "p1v1"
                            })
                            annotation_count += 1
                        except json.JSONDecodeError as e:
                            print(f"  警告: {jsonl_path} 第 {line_num} 行不是有效的 JSON: {e}")
                            continue
                print(f"从 {jsonl_path} 加载了 {annotation_count} 条注释")
            else:
                print(f"未找到 JSONL 文件: {jsonl_path}")
    
    # 保存统一格式的注释
    if unified_annotations:
        print(f"为 {split} 创建统一格式注释，共 {len(unified_annotations)} 条记录")
        unified_path = f"annotations/{split}_unified.json"
        with open(unified_path, "w", encoding="utf-8") as f:
            json.dump(unified_annotations, f, ensure_ascii=False, indent=2)
        print(f"已保存统一格式注释到: {unified_path}")
    else:
        print(f"警告: {split} 没有有效的注释，跳过创建统一格式文件")

print("\n脚本执行完成。") 