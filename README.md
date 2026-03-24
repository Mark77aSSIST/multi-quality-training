# Multi-Quality Training for Robust YOLO Object Detection in CCTV Systems

Files and guides to reproduce the experiments of the paper "No-Trade-Off Quality Robustness: Simultaneous Multi-Quality Training Improves Low-Quality Object Detection Without Sacrificing High-Quality Performance".

## 📝 Authors
Mukeun Choi_1 and Taeyeon Oh_2*

* 1 : Seoul AI School, aSSIST University, Seoul, Republic of Korea
* 2 : Seoul AI School, aSSIST University, Seoul, Republic of Korea
* \* : Corresponding Author

## 📦 File Path

| Path | Description |
|------|------|
| `analysis` | Analysis Results |
| `evaluation` | Evaluation Results |
| `experiments` | Experiments Results |
| `models` | Best pretrainined Models (Standard model & Multi-Quality model) |
| `scripts` | Experiment Code |

### 🔍 Experiment Code in scripts path

| File | Description |
|------|------|
| `01_Environment_setting.py` | Experiments Environment Setup |
| `02_Quality_degradation_implement.py` | Build of the Quality Degradation Function |
| `03_Data_preprocessing.py` | Data Preprocessing |
| `04_Multi_quality_training.py` | Model Training |
| `05_Evaluation.py` | Evaluation |
| `06_Visualization.py` | Figure & Table Generation |

### 📚 Reference File

| File | Description |
|------|------|
| `README.md` | This File |
| `requirements.txt` | Library installation information required to run in a local environment |

## 🔥 Key Results

| Quality | Standard Training | Multi-Quality Training | Improvement |
|---------|-------------------|------------------------|-------------|
| **Q20** | 0.236 mAP@0.5 | **0.337 mAP@0.5** | **+43.1%** ⭐ |
| Q40 | 0.388 | 0.398 | +2.6% |
| Q60 | 0.407 | 0.410 | +0.7% |
| Q80 | 0.408 | 0.410 | +0.4% |
| Q100 | 0.404 | 0.406 | +0.4% |
| **Avg** | **0.366** | **0.390** | **+6.5%** |
| **Std** | **0.066** | **0.028** | **-58.2%** |

## 🚀 Quick Start
1) Login to Google Colab environment (A100 GPU 40GB RAM)
2) Copy and paste Python files in the "scripts" folder into each cell in order of number
3) Upload raw data file to your Google Drive folder (ex. VisDrone_datasets)
4) Connect Google Drive to your colab file
5) Set up and verify the path in the code
6) Run each cell's code in order
7) You can check and download the results

## 💾 Pre-trained Models

We provide pre-trained models for reproducibility:

| Model | Training Data | Q20 mAP | Q100 mAP | Download |
|-------|---------------|---------|----------|----------|
| YOLOv8m-Standard | Q100 only | 0.236 | 0.404 | [Link](https://github.com/Mark77aSSIST/multi-quality-training/blob/main/models/standard/best.pt) |
| YOLOv8m-MultiQuality | Q20-Q100 | **0.337** | **0.406** | [Link](#) |

**Usage:**
```python
from ultralytics import YOLO

# Load pre-trained model
model = YOLO('path/to/multiquality_best.pt')

# Inference
results = model.predict('path/to/image.jpg', conf=0.25)
```

---

## 🔬 Technical Details

### Hardware & Software

- **GPU**: NVIDIA A100 40GB
- **Framework**: PyTorch 2.0.1, YOLOv8 (Ultralytics)
- **Training time**: ~6.5 hours (multi-quality), ~1.5 hours (standard)
- **Cost**: ~$12 (multi-quality) on Google Colab Pro+

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Input size | 640×640 |
| Batch size | 48 |
| Initial LR | 0.06 |
| Optimizer | SGD (momentum=0.937) |
| Weight decay | 0.0005 |
| Epochs | 100 (early stopping) |
| Warmup epochs | 3 |
| Mixed precision | FP16 (AMP) |

## ✅ Raw Data Source

Thank you to team VisDrone (Zhu et al., 2018; Zhu et al., 2021); Machine Learning and Data Mining Lab at Tianjin University in China for providing the data.
The VisDrone dataset (raw-data) used in this study is available at the following link.
* Dataset Link: https://github.com/VisDrone/VisDrone-Dataset

## 📝 License

This project is licensed under the MIT License.

**Note**: YOLOv8 is licensed under AGPL-3.0. Please refer to [Ultralytics License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) for commercial use.
