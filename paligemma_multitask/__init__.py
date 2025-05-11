"""
PaliGemma Multitask - A toolkit for object detection and multimodal tasks using PaliGemma
"""

__version__ = "0.1.0"

from paligemma_multitask.model import create_model, PaliGemmaMultitaskModel
from paligemma_multitask.data import create_data_loaders, PaliGemmaDataset
from paligemma_multitask.training import PaliGemmaTrainer
