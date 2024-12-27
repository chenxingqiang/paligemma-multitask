# PaliGemma Multitask

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)

A toolkit for object detection and multimodal tasks using PaliGemma model.

## Modules

### Object Detection

The object detection module provides functionality for detecting objects in images using PaliGemma.

### Multimodal

The multimodal module handles tasks that combine different types of data (text, images, etc.).

### GPR

The GPR module contains specialized functionality for Ground Penetrating Radar applications.

## Usage Examples

See the `examples/` directory for detailed usage examples.

## API Reference

### Object Detection

- `detect_objects(image_path, annotation_path, num_images=8)`
- `visualize_detections(image_path, annotation_path, num_images=8)`

### Multimodal

- `load_model_and_tokenizer(model_path, tokenizer_path)`
- `train_multimodal(config)`

### GPR

- `train_gpr(config)`
- `evaluate_gpr(model, data)`
