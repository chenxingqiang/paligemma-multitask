# PaliGemma Multitask

A toolkit for defect detection and analysis using PaliGemma. This project combines vision-language capabilities with specialized defect detection features.

## Features

- Defect detection with bounding box visualization
- Position-aware defect localization
- Automated defect description generation
- Basic visual question answering
- LoRA-based fine-tuning support

## Requirements

See `inference_requirements.txt` for the complete list:
- Python >= 3.7
- PyTorch >= 1.9.0
- transformers >= 4.30.0
- peft >= 0.4.0
- accelerate >= 0.20.0
- Other dependencies as specified in requirements.txt

## Installation

1. Clone the repository:

```bash
git clone https://github.com/chenxingqiang/paligemma-multitask-finetune.git
cd paligemma-multitask-finetune
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

3. Install the requirements:

```bash
pip install -r inference_requirements.txt
```

## Model Architecture

The model uses a multi-task architecture with:
1. PaLiGemma base model for vision-language understanding
2. Custom detection head for bounding box prediction
3. Custom classification head for defect type classification
4. LoRA adapters for efficient fine-tuning

## Inference Usage

The model provides three main functionalities:

### 1. Defect Detection

```python
from transformers import AutoProcessor
from paligemma_multitask.model import create_model

# Load model and processor
model_id = "google/paligemma-3b-mix-224"
processor = AutoProcessor.from_pretrained(model_id)
model = create_model(model_id=model_id, num_classes=2)

# Load weights
checkpoint = torch.load('best_model.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Detect defects
boxes, class_logits = detect_defects(model, processor, "your_image.jpg")
```

The detection results include:
- Bounding box coordinates (normalized [0,1])
- Class logits for defect types (void/crack)
- Visualization saved as 'detection_result.png'

### 2. Description Generation

```python
description = generate_description(model, processor, "your_image.jpg")
print(description)
```

Generates a natural language description including:
- Defect type (void/crack)
- Location in the image (e.g., "bottom left", "center")
- Confidence score

### 3. Visual Question Answering

```python
answer = answer_question(model, processor, "your_image.jpg", 
                        "Where is the largest defect located?")
print(answer)
```

Supports basic questions about:
- Defect location
- Number of defects
- Defect types

## Training Process

The model is trained using:
1. LoRA adapters for efficient fine-tuning
2. Multi-task learning with:
   - MSE loss for bounding box coordinates
   - Cross-entropy loss for defect classification

## Model Weights

The trained model weights are saved in `best_model.pt` which includes:
- LoRA adapter weights
- Custom detection head weights
- Custom classification head weights

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

- chenxingqiang (<chen.xingqiang@iechor.com>)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
