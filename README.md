# PaliGemma Multitask

A toolkit for object detection and multimodal tasks using PaliGemma. This project integrates with Google Research's big_vision repository to provide advanced computer vision capabilities.

## Features

- Object detection and multimodal task support
- Integration with Google's big_vision repository
- Text-to-detection conversion
- Text generation capabilities
- Visual question answering
- Visualization utilities

## Requirements

- Python >= 3.7
- PyTorch >= 1.9.0
- transformers
- pillow
- matplotlib
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

3. Install the package:

```bash
pip install -e .
```

## Training Process

The training process converts text-image pairs into object detection tasks through the following steps:

1. **Text Parsing Stage**:
   - Parses location information from text descriptions
   - When text contains phrases like "void in the top left" or "crack at the center":
     - Identifies defect type (void/crack)
     - Extracts position descriptions (top left, center, etc.)
     - Converts position descriptions to normalized coordinates

2. **Coordinate Mapping**:
   - Uses predefined position mapping:
     - "center" → [0.5, 0.5]
     - "top left" → [0.25, 0.25]
     - "bottom right" → [0.75, 0.75]

3. **Bounding Box Generation**:
   - Generates boxes from center coordinates
   - Format: [x1, y1, x2, y2]
   - Uses fixed size (center point ±0.1)

4. **Class Mapping**:
   - Maps text labels to numeric classes:
     - "void" → 0
     - "crack" → 1

5. **Model Training**:
   - Processes both image and text inputs
   - Uses two prediction heads:
     - Detection head: predicts box coordinates
     - Classification head: predicts defect types

6. **Loss Calculation**:
   - Detection loss: MSE loss for box coordinates
   - Classification loss: Cross-entropy loss for defect types

## Training Usage

1. Set up environment variable for Hugging Face:

```bash
export HF=your_huggingface_token
```

2. Run training script:

```bash
python examples/train.py \
    --dataset_path ./dataset/p-1.v1i.paligemma-multimodal/dataset \
    --model_id google/paligemma-3b-mix-224 \
    --batch_size 4 \
    --num_epochs 3 \
    --learning_rate 2e-4
```

After training, the model will be:

1. Saved locally as `final_model.pt`
2. Uploaded to Hugging Face Hub
3. Complete with configurations and documentation

## Model Usage

Install required packages:

```bash
pip install torch transformers pillow matplotlib
```

Basic usage:

```python
from transformers import AutoProcessor, AutoModelForVision2Seq

# Load model and processor
processor = AutoProcessor.from_pretrained("xingqiang/paligemma-multitask-detector")
model = AutoModelForVision2Seq.from_pretrained("xingqiang/paligemma-multitask-detector")

# Prepare image
image_path = "your_image.jpg"
```

### 1. Object Detection

```python
# Detect defects
boxes, classes = object_detection(model, processor, image_path)
# Will automatically display image with bounding boxes
```

### 2. Text Generation

```python
# Generate defect description
description = text_generation(model, processor, image_path)
print(description)
```

### 3. Visual Question Answering

```python
# Ask questions about the image
question = "Where is the largest defect located?"
answer = visual_qa(model, processor, image_path, question)
print(answer)
```

For a complete example, see `examples/demo.py`:

```bash
python examples/demo.py
```

## Features of the Demo

1. Object detection with visualization
   - Different colors for different defect types
   - Bounding box display
   - Class labels

2. Text generation
   - Comprehensive defect descriptions
   - Natural language output

3. Visual QA
   - Interactive questioning
   - Context-aware answers
   - Multiple question types supported

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

- chenxingqiang (<chen.xingqiang@iechor.com>)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
