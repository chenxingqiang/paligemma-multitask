import torch
import numpy as np
from PIL import Image
import requests
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
import cv2
import json
import supervision as sv
from typing import List
import os
import sys
import time
from tqdm import tqdm
import jax
import matplotlib.pyplot as plt
import big_vision.utils

def setup_environment():
    # Setup code same as in paligemma-for-multitask.py
    current_dir = os.getcwd()
    big_vision_path = os.path.join(current_dir, "big_vision_repo")
    os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + os.pathsep + big_vision_path
    sys.path.append(big_vision_path)
    
    if not os.path.exists("big_vision_repo"):
        os.system("git clone --quiet --branch=main --depth=1 https://github.com/google-research/big_vision big_vision_repo")

def calculate_map(data_iterator, batch_size, classes):
    targets = []
    predictions = []
    for image, label, prediction in make_predictions(data_iterator, batch_size=batch_size):
        h, w, _ = image.shape
        target = sv.Detections.from_lmm(
            lmm='paligemma',
            result=label,
            resolution_wh=(w, h),
            classes=classes)
        targets.append(target)
        prediction = sv.Detections.from_lmm(
            lmm='paligemma',
            result=prediction,
            resolution_wh=(w, h),
            classes=classes)
        prediction.confidence = np.ones(len(prediction))
        predictions.append(prediction)

    mean_average_precision = sv.MeanAveragePrecision.from_detections(
        predictions=predictions,
        targets=targets,
    )

    return mean_average_precision.map50, mean_average_precision.map75, mean_average_precision.map50_95, predictions, targets

def train():
    # Training configuration
    BATCH_SIZE = 16
    TRAIN_EXAMPLES = 170
    LEARNING_RATE = 0.01
    EPOCHS = 30
    TRAIN_STEPS = TRAIN_EXAMPLES // BATCH_SIZE
    
    # Initialize metrics
    losses = []
    train_metrics = {"map50_t": [], "map75_t": [], "map50_95_t": []}
    val_metrics = {"map50_v": [], "map75_v": [], "map50_95_v": []}
    best_map50_v = 0
    best_confusion_matrix = None
    
    # Training loop
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        epoch_loss = 0
        pbar = tqdm(total=TRAIN_STEPS, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for _ in range(TRAIN_STEPS):
            step += 1
            examples = [next(train_data_it) for _ in range(BATCH_SIZE)]
            batch = jax.tree_map(lambda *x: np.stack(x), *examples)
            batch = big_vision.utils.reshard(batch, data_sharding)
            learning_rate = sched_fn(step)
            params, loss = update_fn(params, batch, learning_rate)
            loss = jax.device_get(loss)
            epoch_loss += loss
            pbar.update(1)

        pbar.close()
        epoch_loss /= TRAIN_STEPS
        losses.append(epoch_loss)
        
        # Evaluation and model saving logic
        if (epoch + 1) % 3 == 0 or epoch == EPOCHS - 1:
            # Calculate metrics and save model
            map50_t, map75_t, map50_95_t, train_predictions, train_targets = calculate_map(
                validation_data_iterator_trained(), batch_size=8, classes=CLASSES)
            
            # Save metrics
            train_metrics["map50_t"].append(map50_t)
            train_metrics["map75_t"].append(map75_t)
            train_metrics["map50_95_t"].append(map50_95_t)
            
            # Print progress
            print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f}")
            print(f"mAP@50_t: {map50_t:.2f}, mAP@75_t: {map75_t:.2f}, mAP@50-95_t: {map50_95_t:.2f}")

    # Plot results
    plot_training_results(losses, train_metrics, val_metrics)

def plot_training_results(losses, train_metrics, val_metrics):
    # Plot loss
    plt.figure()
    plt.plot(range(1, len(losses)+1), losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.show()
    
    # Plot mAP metrics
    plt.figure()
    plt.plot(range(1, len(train_metrics["map50_t"])*3+1, 3), 
             train_metrics["map50_t"], marker='o', label="Training mAP@50")
    plt.plot(range(1, len(val_metrics["map50_v"])*3+1, 3), 
             val_metrics["map50_v"], marker='x', label="Validation mAP@50")
    plt.xlabel('Epoch')
    plt.ylabel('mAP@50')
    plt.title('Training vs Validation mAP@50')
    plt.legend()
    plt.show()

def main():
    setup_environment()
    train()

if __name__ == "__main__":
    main() 