import torch
import numpy as np
from PIL import Image
import requests
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
import os
import sys
import io
import base64
import html
import time
from tqdm import tqdm
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import functools
import big_vision.utils
import sentencepiece
import ml_collections
from IPython.display import display, HTML

def setup_environment():
    current_dir = os.getcwd()
    big_vision_path = os.path.join(current_dir, "big_vision_repo")
    os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + os.pathsep + big_vision_path
    sys.path.append(big_vision_path)
    
    if not os.path.exists("big_vision_repo"):
        os.system("git clone --quiet --branch=main --depth=1 https://github.com/google-research/big_vision big_vision_repo")

def load_model_and_tokenizer():
    MODEL_PATH = "/root/model/paligemma-3b-pt-224.f16.npz"
    TOKENIZER_PATH = "/root/model/paligemma_tokenizer.model"
    
    print(f"Loading model from {MODEL_PATH}...")
    model = np.load(MODEL_PATH)
    
    print(f"Loading tokenizer from {TOKENIZER_PATH}...")
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = f.read()
    
    return model, tokenizer

def train_model():
    BATCH_SIZE = 16
    TRAIN_EXAMPLES = 170
    LEARNING_RATE = 0.01
    EPOCHS = 30
    TRAIN_STEPS = TRAIN_EXAMPLES // BATCH_SIZE
    
    train_data_it = train_data_iterator()
    sched_fn = big_vision.utils.create_learning_rate_schedule(
        total_steps=TRAIN_STEPS+1, 
        base=LEARNING_RATE,
        decay_type="cosine", 
        warmup_percent=0.10)
    
    for epoch in range(1, EPOCHS+1):
        start_time = time.time()
        
        for step in range(1, TRAIN_STEPS+1):
            examples = [next(train_data_it) for _ in range(BATCH_SIZE)]
            batch = jax.tree.map(lambda *x: np.stack(x), *examples)
            batch = big_vision.utils.reshard(batch, data_sharding)
            
            learning_rate = sched_fn(step)
            params, loss = update_fn(params, batch, learning_rate)
            loss = jax.device_get(loss)
            
            print(f"Epoch: {epoch}/{EPOCHS} - step: {step:2d}/{TRAIN_STEPS:2d} "
                  f"lr: {learning_rate:.5f} loss: {loss:.4f}")
            
            if step == 1 or (step % EVAL_STEPS) == 0:
                evaluate_and_display(step)
        
        end_time = time.time()
        print(f"Epoch {epoch}/{EPOCHS} completed in {end_time - start_time:.2f} seconds")

def evaluate_model():
    print("Model predictions")
    html_out = ""
    for image, caption in make_predictions(validation_data_iterator(), batch_size=4):
        html_out += render_example(image, caption)
    display(HTML(html_out))

def main():
    setup_environment()
    model, tokenizer = load_model_and_tokenizer()
    train_model()
    evaluate_model()

if __name__ == "__main__":
    main() 