import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_model(model_id="xingqiang/paligemma-multitask-detector"):
    """加载模型和处理器"""
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(model_id)
    return model, processor

def visualize_detection(image, boxes, classes):
    """可视化检测结果"""
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()
    
    # 定义颜色映射
    color_map = {
        0: 'red',    # void
        1: 'blue'    # crack
    }
    
    # 绘制每个边界框
    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        
        # 创建矩形
        rect = patches.Rectangle(
            (x1 * image.width, y1 * image.height),
            w * image.width,
            h * image.height,
            linewidth=2,
            edgecolor=color_map[cls],
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # 添加类别标签
        label = "void" if cls == 0 else "crack"
        plt.text(
            x1 * image.width,
            y1 * image.height,
            label,
            color='white',
            bbox=dict(facecolor=color_map[cls], alpha=0.8)
        )
    
    plt.axis('off')
    plt.show()

def object_detection(model, processor, image_path):
    """执行目标检测任务"""
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    
    # 准备输入
    inputs = processor(
        images=image,
        text="Detect defects in this image.",
        return_tensors="pt"
    )
    
    # 获取预测
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 处理检测结果
    boxes = outputs['boxes'][0].cpu().numpy()
    class_logits = outputs['class_logits'][0].cpu()
    classes = torch.argmax(class_logits, dim=-1).numpy()
    
    # 可视化结果
    visualize_detection(image, boxes, classes)
    
    return boxes, classes

def text_generation(model, processor, image_path):
    """执行文本生成任务"""
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    
    # 准备输入
    inputs = processor(
        images=image,
        text="Describe the defects in this image:",
        return_tensors="pt"
    )
    
    # 生成文本
    outputs = model.generate(
        **inputs,
        max_length=100,
        num_beams=4,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    # 解码生成的文本
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return generated_text

def visual_qa(model, processor, image_path, question):
    """执行视觉问答任务"""
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    
    # 准备输入
    inputs = processor(
        images=image,
        text=question,
        return_tensors="pt"
    )
    
    # 生成答案
    outputs = model.generate(
        **inputs,
        max_length=50,
        num_beams=4,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    # 解码生成的答案
    answer = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return answer

def main():
    # 加载模型
    model, processor = load_model()
    
    # 示例图片路径
    image_path = "path/to/your/image.jpg"
    
    print("1. 执行目标检测...")
    boxes, classes = object_detection(model, processor, image_path)
    print(f"检测到 {len(boxes)} 个缺陷")
    
    print("\n2. 生成缺陷描述...")
    description = text_generation(model, processor, image_path)
    print(f"描述: {description}")
    
    print("\n3. 视觉问答...")
    questions = [
        "What types of defects are present in the image?",
        "Where is the largest defect located?",
        "How many cracks are there in the image?"
    ]
    
    for question in questions:
        answer = visual_qa(model, processor, image_path, question)
        print(f"问题: {question}")
        print(f"答案: {answer}\n")

if __name__ == "__main__":
    main() 