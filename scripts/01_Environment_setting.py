# 1. Environment setting
from google.colab import drive
import os

# Google Drive mount & setup the directory for dataset
GDRIVE_ROOT = '/content/drive/MyDrive'
DATASET_PATH = os.path.join(GDRIVE_ROOT, 'VisDrone')  # for VisDrone dataset
EXPERIMENT_PATH = os.path.join(GDRIVE_ROOT, 'CCTV_MultiRes_Experiments')  # for Experiment results save

os.makedirs(EXPERIMENT_PATH, exist_ok=True)
os.makedirs(os.path.join(EXPERIMENT_PATH, 'checkpoints'), exist_ok=True)
os.makedirs(os.path.join(EXPERIMENT_PATH, 'results'), exist_ok=True)
os.makedirs(os.path.join(EXPERIMENT_PATH, 'logs'), exist_ok=True)

print("Google Drive mount complete")
print(f"Dataset directory: {DATASET_PATH}")
print(f"Experiment results directory: {EXPERIMENT_PATH}")

# GPU usability check
import torch

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\nGPU available: {gpu_name}")
    print(f"GPU memory: {gpu_memory:.1f} GB")
else:
    print("\nCan't use GPU.")

# Dataset structure check
print("\nDataset structure check:")

if os.path.exists(DATASET_PATH):
    print("\nDataset Directory:")
    for item in os.listdir(DATASET_PATH):
        item_path = os.path.join(DATASET_PATH, item)
        if os.path.isdir(item_path):
            print(f"{item}/")
            try:
                sub_items = os.listdir(item_path)[:5]
                for sub_item in sub_items:
                    print(f"      - {sub_item}")
                if len(os.listdir(item_path)) > 5:
                    print(f"      ... and {len(os.listdir(item_path)) - 5} more")
            except:
                pass
else:
    print(f"\nCan't find the dataset directory: {DATASET_PATH}")

# Save the preferences
config = {
    'GDRIVE_ROOT': GDRIVE_ROOT,
    'DATASET_PATH': DATASET_PATH,
    'EXPERIMENT_PATH': EXPERIMENT_PATH,
    'TRAIN_DIR': os.path.join(DATASET_PATH, 'VisDrone2019-DET-train'),
    'VAL_DIR': os.path.join(DATASET_PATH, 'VisDrone2019-DET-val'),
    'TEST_DIR': os.path.join(DATASET_PATH, 'VisDrone2019-DET-test-dev'),
}

print("\nData path & Directory:")
for key, value in config.items():
    print(f"  {key}: {value}")

print("\nEnvironment setting Complete")


# YOLOv8 & Library install
print("Installing the library...")

# Ultralytics YOLOv8 install
!pip install -q ultralytics

# Other library install
!pip install -q opencv-python-headless
!pip install -q albumentations
!pip install -q wandb
!pip install -q scikit-learn
!pip install -q seaborn

print("Library install Complete")

# Library import & Version check
import sys
import numpy as np
import cv2
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import yaml
from datetime import datetime
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# Ultralytics YOLO
from ultralytics import YOLO

print("\nLibrary Version:")
print(f"  Python: {sys.version.split()[0]}")
print(f"  PyTorch: {torch.__version__}")
print(f"  Torchvision: {torchvision.__version__}")
print(f"  NumPy: {np.__version__}")
print(f"  OpenCV: {cv2.__version__}")

# Visualization setup
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
sns.set_style("whitegrid")

print("\nLibrary import Complete")