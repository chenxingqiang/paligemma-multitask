# PaliGemma Multitask

A toolkit for object detection and multimodal tasks using PaliGemma. This project integrates with Google Research's big_vision repository to provide advanced computer vision capabilities.

## Features

- Object detection and multimodal task support
- Integration with Google's big_vision repository
- GPR (Gaussian Process Regression) implementation
- Visualization utilities

## Requirements

- Python >= 3.7
- PyTorch >= 1.9.0
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

## Usage

The package provides several modules for different tasks:

```python
from paligemma_multitask import object_detection
from paligemma_multitask.utils.environment import setup_environment

# Setup the environment (required for big_vision integration)
setup_environment()

# Use object detection
detections = object_detection.detect_objects(image_path)
```

## Training

To train the model on your dataset:

```bash
python examples/train.py \
    --dataset_path ./dataset/p-1.v1i.paligemma-multimodal/dataset \
    --model_id google/paligemma-3b-mix-224 \
    --batch_size 4 \
    --num_epochs 3 \
    --learning_rate 2e-4
```

Parameters:
- `dataset_path`: Path to your dataset directory
- `model_id`: The HuggingFace model ID to use
- `batch_size`: Training batch size
- `num_epochs`: Number of training epochs
- `learning_rate`: Learning rate for optimization

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

- chenxingqiang (chen.xingqiang@iechor.com)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 