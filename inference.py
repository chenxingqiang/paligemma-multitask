import os
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import AutoProcessor
from paligemma_multitask.model import PaliGemmaMultitaskModel, create_model
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model():
    """Load the model with proper architecture and weights"""
    model_id = "google/paligemma-3b-mix-224"
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Create model using the helper function that includes LoRA configuration
    model = create_model(model_id=model_id, num_classes=2)
    
    # Load saved weights
    logger.info("Loading saved weights from best_model.pt")
    checkpoint = torch.load('best_model.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set model to evaluation mode
    model.eval()
    logger.info("Model loaded and set to evaluation mode")
    
    return model, processor

def detect_defects(model, processor, image_path):
    """Detect defects in an image"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Prepare input
    inputs = processor(
        images=image,
        text="<image> Detect defects in this image:",
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    # Get device from model's base model
    device = next(model.base_model.parameters()).device
    
    # Move inputs to model device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # Get predictions
    outputs = model(**inputs)
    boxes = outputs['boxes']
    class_logits = outputs['class_logits']
    
    # Visualize results
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    # Draw bounding boxes
    for box in boxes:
        x1, y1, x2, y2 = box.detach().cpu().numpy()
        x1, y1, x2, y2 = x1 * image.width, y1 * image.height, x2 * image.width, y2 * image.height
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
    
    plt.axis('off')
    plt.savefig('detection_result.png')
    plt.close()
    
    return boxes, class_logits

def generate_description(model, processor, image_path):
    """Generate text description of defects"""
    # First get detection results
    boxes, class_logits = detect_defects(model, processor, image_path)
    
    # Convert class logits to probabilities
    probs = torch.softmax(class_logits, dim=-1)
    class_id = torch.argmax(probs).item()
    confidence = probs[0][class_id].item()
    
    # Map class id to name
    class_names = ["void", "crack"]
    defect_type = class_names[class_id]
    
    # Map box coordinates to position
    box = boxes[0]  # Get first box
    x_center = (box[0] + box[2]) / 2
    y_center = (box[1] + box[3]) / 2
    
    # Map coordinates to position description
    position = "bottom right" if x_center > 0.5 else "bottom left" if x_center < 0.5 else "center"
    
    # Generate description
    if confidence > 0.5:
        description = f"There is a {defect_type} defect in the {position} of the image with {confidence:.2f} confidence."
    else:
        description = f"No significant defects detected in the image (confidence: {confidence:.2f})."
    
    return description

def answer_question(model, processor, image_path, question):
    """Answer questions about the image"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Prepare input
    inputs = processor(
        images=image,
        text=f"<image> {question}",
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    # Move inputs to model device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate answer
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        num_beams=4,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    # Decode generated text
    answer = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return answer

def main():
    # Load model and processor
    model, processor = load_model()
    
    # Use test image
    image_path = "test.jpg"
    
    # Create test image if it doesn't exist
    if not os.path.exists(image_path):
        # Create a simple test image
        img = Image.new('RGB', (224, 224), color='white')
        # Add some simulated defects
        draw = ImageDraw.Draw(img)
        draw.ellipse([50, 50, 100, 100], fill='gray')  # void
        draw.line([150, 150, 200, 200], fill='black', width=3)  # crack
        img.save(image_path)
    
    # Detect defects
    boxes, class_logits = detect_defects(model, processor, image_path)
    
    # Convert class logits to probabilities
    probs = torch.softmax(class_logits, dim=-1)
    class_id = torch.argmax(probs).item()
    confidence = probs[0][class_id].item()
    
    # Map class id to name
    class_names = ["void", "crack"]
    defect_type = class_names[class_id]
    
    # Map box coordinates to position
    box = boxes[0]  # Get first box
    x_center = (box[0] + box[2]) / 2
    y_center = (box[1] + box[3]) / 2
    
    # Map coordinates to position description
    position = "bottom right" if x_center > 0.5 else "bottom left" if x_center < 0.5 else "center"
    
    print("\nDefect Detection Results:")
    print(f"Location: {position}")
    print(f"Type: {defect_type}")
    print(f"Confidence: {confidence:.2f}")
    
    print("\nGenerating description...")
    description = generate_description(model, processor, image_path)
    print(f"\nDescription:\n{description}")
    
    print("\nAnswering questions...")
    questions = [
        "What types of defects are present in the image?",
        "Where is the largest defect located?",
        "How many defects are there in the image?"
    ]
    
    for question in questions:
        answer = answer_question(model, processor, image_path, question)
        print(f"\nQ: {question}")
        print(f"A: {answer}")

if __name__ == "__main__":
    main() 