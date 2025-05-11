# Ground-Penetrating Radar Damage Detection with PaLI-GEMMA

This repository contains scripts for fine-tuning Google's PaLI-GEMMA multimodal model on a ground-penetrating radar (GPR) damage detection dataset. The dataset contains images of GPR scans with annotations for void and crack damages.

## Dataset

The dataset consists of ground-penetrating radar images with annotations describing the presence, location, and characteristics of voids and cracks. The annotations are provided in two formats:

1. **Basic descriptions**: General descriptions of damage appearance and location
2. **Technical descriptions**: Detailed technical descriptions including amplitude, attenuation, and distribution range

## Scripts

The repository includes the following scripts:

1. `process_dataset.py`: Processes the raw dataset files and organizes them into training and testing sets
2. `prepare_paligemma_dataset.py`: Prepares the dataset for fine-tuning with PaLI-GEMMA, creating different versions with various prompt formats
3. `finetune_paligemma.py`: Fine-tunes PaLI-GEMMA on the prepared dataset using standard supervised learning
4. `evaluate_model.py`: Evaluates the fine-tuned model on the test dataset
5. `finetune_paligemma_grpo.py`: Fine-tunes PaLI-GEMMA using Group Relative Policy Optimization (GRPO)
6. `evaluate_grpo_model.py`: Evaluates the GRPO-trained model and generates responses in reporter format
7. `generate_report.py`: Generates technical reports in reporter format for new images
8. `run_all.sh`: Shell script to run the standard fine-tuning pipeline
9. `run_grpo.sh`: Shell script to run the GRPO fine-tuning pipeline

## Requirements

Install the required packages:

```bash
pip install torch transformers pillow numpy tqdm scikit-learn matplotlib seaborn datasets trl
```

For GRPO with vLLM acceleration (optional):

```bash
pip install vllm
```

## Usage

### Standard Fine-tuning

Run the standard fine-tuning pipeline:

```bash
./run_all.sh
```

### GRPO Fine-tuning

Run the GRPO fine-tuning pipeline, which enhances the model's reasoning capabilities and produces reporter-style outputs:

```bash
./run_grpo.sh
```

### Generate Technical Reports

Generate technical reports for new images using the GRPO-trained model:

```bash
# For a single image
python generate_report.py --image_path path/to/your/image.jpg

# For a directory of images
python generate_report.py --image_path path/to/your/images/directory
```

Available options:
- `--model_path`: Path to the fine-tuned model (default: "paligemma_grpo_finetuned/final_model")
- `--image_path`: Path to the image or directory of images (required)
- `--output_dir`: Output directory for the generated reports (default: "reports")
- `--device`: Device to run inference on (default: "cuda" if available, otherwise "cpu")

### 1. Process the Dataset

First, process the raw dataset files:

```bash
python process_dataset.py
```

This will create a `processed_dataset` directory with the following structure:

```
processed_dataset/
├── metadata.json
├── train/
│   ├── annotations_basic.jsonl
│   ├── annotations_technical.jsonl
│   ├── annotations_combined.jsonl
│   └── images/
└── test/
    ├── annotations_basic.jsonl
    ├── annotations_technical.jsonl
    ├── annotations_combined.jsonl
    └── images/
```

### 2. Prepare the Dataset for PaLI-GEMMA

Next, prepare the dataset for fine-tuning with PaLI-GEMMA:

```bash
python prepare_paligemma_dataset.py
```

This will create a `paligemma_dataset` directory with different versions of the dataset:

- `basic_detection_train.jsonl` / `basic_detection_test.jsonl`: Simple detection of void or crack presence
- `descriptive_train.jsonl` / `descriptive_test.jsonl`: General descriptions of damage appearance and location
- `technical_train.jsonl` / `technical_test.jsonl`: Detailed technical descriptions of damage characteristics
- `combined_train.jsonl` / `combined_test.jsonl`: Mix of all prompt types and description formats

### 3. Fine-tune PaLI-GEMMA

#### Standard Fine-tuning

Fine-tune PaLI-GEMMA on the prepared dataset using standard supervised learning:

```bash
python finetune_paligemma.py --model_name google/paligemma-3b-mix-224 --dataset_version combined --num_epochs 3 --batch_size 8 --fp16
```

#### GRPO Fine-tuning

Fine-tune PaLI-GEMMA using Group Relative Policy Optimization (GRPO), which enhances the model's reasoning capabilities:

```bash
python finetune_paligemma_grpo.py --model_name google/paligemma-3b-mix-224 --dataset_version combined --num_epochs 3 --batch_size 4 --num_generations 8 --fp16
```

GRPO-specific parameters:

- `--num_generations`: Number of generations per prompt to sample (default: 8)
- `--beta`: KL coefficient for GRPO (default: 0.04)
- `--epsilon`: Epsilon value for clipping in GRPO (default: 0.2)
- `--use_vllm`: Use vLLM for generation acceleration (flag)

### 4. Evaluate the Fine-tuned Model

#### Standard Evaluation

Evaluate the standard fine-tuned model:

```bash
python evaluate_model.py --model_path paligemma_finetuned/final_model --dataset_version combined
```

#### GRPO Evaluation with Reporter Format

Evaluate the GRPO-trained model and generate responses in reporter format:

```bash
python evaluate_grpo_model.py --model_path paligemma_grpo_finetuned/final_model --dataset_version combined
```

The reporter format provides concise, factual, and structured descriptions of the damage in the radar images, similar to a professional technical report.

## Dataset Versions

The dataset is prepared in different versions for different training scenarios:

1. **Basic Detection**: Simple detection of void or crack presence
   - Prompt: "detect void" or "detect crack"
   - Response: "void" or "crack"

2. **Descriptive**: General descriptions of damage appearance and location
   - Prompt: Various templates like "describe the void in this image", "analyze the ground-penetrating radar image for crack", etc.
   - Response: Detailed description of the damage

3. **Technical**: Detailed technical descriptions of damage characteristics
   - Prompt: "provide technical details about the void/crack in this radar image"
   - Response: Technical description including amplitude, attenuation, and distribution range

4. **Combined**: Mix of all prompt types and description formats

## GRPO Training

Group Relative Policy Optimization (GRPO) is an advanced reinforcement learning technique that enhances the model's reasoning capabilities. It was introduced in the paper "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models".

The GRPO implementation in this repository:

1. Generates multiple completions for each prompt
2. Computes rewards based on technical detail, location accuracy, and reporter format
3. Optimizes the model to maximize these rewards while staying close to the reference policy
4. Produces outputs in a concise, factual reporter style format

## Reporter Format

The reporter format provides a structured, professional way to describe damage in radar images:

1. **Concise**: Avoids unnecessary words and focuses on facts
2. **Factual**: Presents information in a clear, objective manner
3. **Structured**: Organizes information logically
4. **Professional**: Avoids first-person references and maintains a formal tone

Example:
```
Void damage detected in the center of the image. Characterized by an irregular hyperbolic shape with strong amplitude. Distribution range is large with noticeable multiple reflections.
```

## Model Outputs

The fine-tuned models can:

1. Detect the presence of voids and cracks in GPR images
2. Describe the location and appearance of damages
3. Provide technical details about the damage characteristics
4. (GRPO model) Generate professional reporter-style technical reports

## Evaluation Metrics

The evaluation scripts calculate:

1. Accuracy, precision, recall, and F1 score for void and crack detection
2. Confusion matrix for damage type classification
3. Sample predictions for descriptive tasks
4. (GRPO model) Technical reports in reporter format

## License

This project is licensed under the MIT License - see the LICENSE file for details.

# PaliGemma Multitask Fine-tuning for Civil Engineering Damage Detection

This repository contains code for fine-tuning Google's PaliGemma multimodal model on ground-penetrating radar (GPR) images for civil engineering damage detection. The project implements a multitask approach that combines caption generation with damage detection (void/crack classification and localization).

## Features

- **Multitask Training**: Combined caption generation and damage detection in a single model
- **Caption Generation**: Detailed descriptions of GPR images with technical terminology
- **Damage Detection**: Classification and localization of voids and cracks in GPR images
- **Inference Scripts**: Separate test scripts for caption generation and damage detection

## Project Structure

```
.
├── annotations/
│   ├── p-1.v1i.paligemma/
│   │   ├── README.dataset.md
│   │   └── README.roboflow.md
│   └── p-1.v1i.paligemma-multimodal/
│       └── README.md
├── debug_model_structure.py
├── debug_pali.py
├── debug_tokenization.py
├── paligemma_multitask/
│   ├── __init__.py
│   ├── config.py
│   ├── convert_annotations.py
│   ├── convert.py
│   ├── data.py
│   ├── model.py
│   ├── training.py
│   ├── object_detection.py
│   ├── paligemma_dataset.py
│   └── utils/
│       ├── environment.py
│       └── metrics.py
├── run_custom_training.py
├── run.md
├── test_caption_inference.py
├── test_detection_inference.py
├── test_inference_simple.py
├── train_caption_simple.py
└── train_caption.py
```

## Quick Start

### Installation

```bash
pip install torch transformers peft datasets matplotlib pillow
```

### Training

For multitask training:
```bash
python run_custom_training.py \
  --model_name "google/paligemma-3b-mix-224" \
  --dataset_path "dataset" \
  --annotation_type "multimodal" \
  --batch_size 2 \
  --learning_rate 1e-4 \
  --num_epochs 3
```

For caption-only training:
```bash
python train_caption_simple.py \
  --dataset_path "dataset" \
  --output_dir "caption_model" \
  --batch_size 1 \
  --learning_rate 5e-5 \
  --num_epochs 3
```

### Inference

Test caption generation:
```bash
python test_inference_simple.py --dataset_path "dataset" --num_samples 3
```

Test damage detection:
```bash
python test_detection_inference.py --dataset_path "dataset" --num_samples 3
```

## Technical Details

### Problem Addressed

This project tackles several challenges in civil engineering damage detection:
1. Generating detailed captions for specialized GPR images
2. Detecting and localizing damage (voids and cracks)
3. Overcoming training challenges like gradient explosion

### Implementation Highlights

- **Gradient Clipping**: Prevents NaN losses during training
- **Label Processing**: Proper handling of caption labels for language model loss
- **Combined Loss**: Weighted combination of caption loss and detection loss
- **Specialized Prompts**: Domain-specific prompts for civil engineering applications

### Common Issues and Solutions

See the [run.md](run.md) file for detailed troubleshooting guidance, including:
- Image token mismatch issues
- Gradient explosion during training
- GPU memory optimization

## Dataset

The dataset includes ground-penetrating radar (GPR) images with annotations describing damage types (void/crack) and their locations, organized as follows:

```
dataset/
├── annotations/
│   ├── multimodal/
│   │   ├── annotations.train.jsonl
│   │   └── annotations.valid.jsonl
│   └── p-1.v1i.paligemma/
│       ├── annotations.train.jsonl
│       └── annotations.valid.jsonl
└── images/
    └── datasets/
        └── [image files]
```

## Full Documentation

For complete documentation on the training and inference process, see [run.md](run.md).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

# PaliGemma 多任务微调 (中文说明)

本仓库包含用于微调Google的PaLI-GEMMA多模态模型的代码，用于土木工程中地质雷达(GPR)图像的损伤检测。项目实现了一个多任务方法，结合了图像描述生成和损伤检测（空洞/裂缝分类和定位）。

## 功能特点

- **多任务训练**：在单个模型中结合图像描述生成和损伤检测
- **描述生成**：使用专业术语对GPR图像进行详细描述
- **损伤检测**：GPR图像中空洞和裂缝的分类和定位
- **推理脚本**：用于图像描述生成和损伤检测的单独测试脚本

## 项目结构

```
.
├── annotations/                     # 数据集注释文件
│   ├── p-1.v1i.paligemma/          # 主要数据集元数据
│   └── p-1.v1i.paligemma-multimodal/ # 多模态数据集元数据
├── debug_*.py                       # 调试脚本
├── paligemma_multitask/            # 核心实现
│   ├── data.py                     # 数据集加载和预处理
│   ├── model.py                    # PaliGemma多任务模型
│   ├── training.py                 # 训练循环和损失计算
│   └── utils/                      # 辅助功能
├── run_custom_training.py          # 主训练脚本
├── test_*_inference.py             # 推理测试脚本
└── train_caption*.py               # 图像描述训练脚本
```

## 快速开始

### 安装

```bash
pip install torch transformers peft datasets matplotlib pillow
```

### 训练

多任务训练:
```bash
python run_custom_training.py \
  --model_name "google/paligemma-3b-mix-224" \
  --dataset_path "dataset" \
  --annotation_type "multimodal" \
  --batch_size 2 \
  --learning_rate 1e-4 \
  --num_epochs 3
```

仅图像描述训练:
```bash
python train_caption_simple.py \
  --dataset_path "dataset" \
  --output_dir "caption_model" \
  --batch_size 1 \
  --learning_rate 5e-5 \
  --num_epochs 3
```

### 推理

测试图像描述生成:
```bash
python test_inference_simple.py --dataset_path "dataset" --num_samples 3
```

测试损伤检测:
```bash
python test_detection_inference.py --dataset_path "dataset" --num_samples 3
```

## 详细文档

有关训练和推理过程的完整文档，请参阅[run.md](run.md)。
