# PaliGemma Multitask Fine-tuning: Train to Inference Guide

This guide documents the complete process for fine-tuning PaliGemma on a multimodal dataset of civil engineering damage detection, from training to inference.

## Table of Contents
- [Environment Setup](#environment-setup)
- [Dataset Structure](#dataset-structure)
- [Training](#training)
  - [Multitask Training](#multitask-training)
  - [Caption-only Training](#caption-only-training)
- [Inference](#inference)
  - [Caption Generation](#caption-generation)
  - [Damage Detection](#damage-detection)
- [Troubleshooting](#troubleshooting)

## Environment Setup

1. Install dependencies:

```bash
pip install torch transformers peft datasets matplotlib pillow
```

2. Clone this repository:

```bash
git clone https://github.com/chenxingqiang/paligemma-multitask.git
cd paligemma-multitask-finetune
```

## Dataset Structure

The dataset includes ground-penetrating radar (GPR) images with annotations describing damage types (void/crack) and their locations. The dataset is organized as follows:

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

Each annotation includes image paths, captions, and damage descriptions.

## Training

### Multitask Training

The multitask model is trained to perform both caption generation and damage detection simultaneously. This approach leverages the vision-language capabilities of PaliGemma for both tasks.

Run the custom training script:

```bash
python run_custom_training.py \
  --model_name "google/paligemma-3b-mix-224" \
  --dataset_path "dataset" \
  --annotation_type "multimodal" \
  --output_dir "checkpoints" \
  --batch_size 2 \
  --learning_rate 1e-4 \
  --num_epochs 3 \
  --max_grad_norm 1.0 \
  --caption_loss_weight 0.1
```

For debugging or on systems with limited resources, use the `--debug` flag:

```bash
python run_custom_training.py --debug --batch_size 2 --learning_rate 1e-4 --num_epochs 1
```

The multitask training combines:
- Language modeling loss for text generation
- Caption generation loss (weighted by `caption_loss_weight`)
- Detection loss (bounding box regression and classification loss)

### Caption-only Training

For caption-focused fine-tuning, use the simplified caption training script:

```bash
python train_caption_simple.py \
  --dataset_path "dataset" \
  --output_dir "caption_model" \
  --batch_size 1 \
  --learning_rate 5e-5 \
  --num_epochs 3 \
  --freeze_vision
```

This script focuses solely on optimizing the language generation capabilities of the model for detailed caption generation.

## Inference

### Caption Generation

Test the caption generation capabilities using either the pre-trained PaliGemma model or the fine-tuned model.

#### Using Pre-trained Model

```bash
python test_inference_simple.py --dataset_path "dataset" --num_samples 3
```

This script runs inference with three different prompts:
1. General description: "Describe the image in detail."
2. Specialized prompt: "This is a ground-penetrating radar image used in civil engineering. Describe in detail what damage is visible in this image, including the type (void or crack), location, and characteristics."
3. VQA format: "What type of damage is shown in this ground-penetrating radar image: void or crack?"

#### Using Fine-tuned Caption Model

```bash
python test_caption_inference.py \
  --model_path "caption_model/checkpoint_epoch_1" \
  --dataset_path "dataset" \
  --num_samples 3
```

This loads the fine-tuned caption model and generates descriptions for the test images.

### Damage Detection

Test damage detection and localization:

```bash
python test_detection_inference.py \
  --dataset_path "dataset" \
  --num_samples 3 \
  --output_dir "detection_results"
```

This script:
1. Processes test images using the PaliGemma model
2. Extracts damage type and location information using a combination of:
   - Direct model responses to localization prompts
   - Heuristic extraction from captions
3. Visualizes the detected damages with bounding boxes
4. Saves the detection results as images and text files

## Troubleshooting

### Image Token Mismatch

If you encounter errors like:
```
ValueError: Number of images does not match number of special image tokens in the input text.
```

This is due to PaliGemma's specific way of handling image tokens. Solution approaches:

1. Use the processor correctly:
   ```python
   inputs = processor(images=image, text=prompt, return_tensors="pt")
   ```

2. Don't manually include `<image>` tokens in your text; the processor will handle this.

3. For custom implementations, ensure the image token sequence length matches what the model expects.

### Gradient Explosion

If you see `NaN` values in losses:

1. Reduce learning rate (try 1e-5 or lower)
2. Increase gradient clipping threshold (`--max_grad_norm 1.0`)
3. Use a smaller batch size
4. Implement warm-up steps

### GPU Memory Issues

For limited GPU memory:

1. Use `--debug` mode for testing
2. Set `--batch_size` to a smaller value (1 or 2)
3. Use gradient accumulation steps
4. Enable gradient checkpointing
5. Fall back to CPU with `--fp16 False`

## Advanced Usage

For fine-grained control over the trade-offs between caption quality and detection accuracy, adjust the `--caption_loss_weight` parameter:

- Higher values (e.g., 0.3): Better caption quality, potentially reduced detection performance
- Lower values (e.g., 0.05): Better detection performance, potentially reduced caption quality
- Default (0.1): Balanced performance

## Conclusion

This project demonstrates how to fine-tune PaliGemma for specialized multimodal tasks in civil engineering. By combining caption generation and damage detection, we leverage the model's vision-language capabilities for practical applications.

The fine-tuned model shows promising results in:
1. Generating detailed technical descriptions of civil engineering images
2. Identifying damage types (voids vs. cracks)
3. Localizing damages within the images

Future improvements could include:
- Larger and more diverse training datasets
- More robust detection head architectures
- Integration with specialized civil engineering knowledge bases 