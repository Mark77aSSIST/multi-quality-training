import os
import glob
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Google Drive mount
from google.colab import drive
drive.mount('/content/drive')

GDRIVE_ROOT = '/content/drive/MyDrive'
EXPERIMENT_PATH = os.path.join(GDRIVE_ROOT, 'CCTV_MultiRes_Experiments')

YOLO_DATASET_DIR = os.path.join(EXPERIMENT_PATH, 'yolo_dataset')
EXPERIMENTS_DIR = os.path.join(EXPERIMENT_PATH, 'experiments')
EVAL_DIR = os.path.join(EXPERIMENT_PATH, 'evaluation_results')
os.makedirs(EVAL_DIR, exist_ok=True)

assert os.path.exists(YOLO_DATASET_DIR), f" Cannot find directory: {YOLO_DATASET_DIR}"

CLASS_NAMES = ['pedestrian', 'people', 'bicycle', 'car', 'van',
               'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
MULTIQUALITY_QUALITY_FOLDERS = ['q20', 'q40', 'q60', 'q80', 'q100']

print(" Directory check completed")


# YAML generation
def make_standard_yaml():
    d = {
        'path': YOLO_DATASET_DIR,
        'train': 'images/train/q100',   
        'val': 'images/val/q100',
        'test': 'images/test',
        'nc': len(CLASS_NAMES),
        'names': CLASS_NAMES,
    }
    path = os.path.join(YOLO_DATASET_DIR, 'visdrone_standard.yaml')
    with open(path, 'w') as f:
        yaml.dump(d, f, default_flow_style=False)
    return path


def make_multiquality_yaml():
    d = {
        'path': YOLO_DATASET_DIR,
        'train': [f'images/train/{q}' for q in MULTIQUALITY_QUALITY_FOLDERS],  
        'val': 'images/val/q100',
        'test': 'images/test',
        'nc': len(CLASS_NAMES),
        'names': CLASS_NAMES,
    }
    path = os.path.join(YOLO_DATASET_DIR, 'visdrone_multiquality.yaml')
    with open(path, 'w') as f:
        yaml.dump(d, f, default_flow_style=False)
    return path


STANDARD_YAML = make_standard_yaml()
MULTIQUALITY_YAML = make_multiquality_yaml()
print(f" YAML generation completed!\n  Standard: {STANDARD_YAML}\n  Multi-Quality: {MULTIQUALITY_YAML}")


# Pre-validation of dataset integrity prior to start of learning
def count_images(folder_path):
    return len([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))])


def assert_dataset_integrity():
    print(f"\n{'='*80}\n Pre-validation of dataset integrity prior to start of learning\n{'='*80}")

    # 1) Standard: q100 
    q100_dir = os.path.join(YOLO_DATASET_DIR, 'images', 'train', 'q100')
    n_standard = count_images(q100_dir)
    print(f"  Standard (q100): {n_standard:,} ea")

    # 2) Multi-Quality: 5 quality levels
    n_multiquality_total = 0
    for q in MULTIQUALITY_QUALITY_FOLDERS:
        folder = os.path.join(YOLO_DATASET_DIR, 'images', 'train', q)
        n = count_images(folder)
        print(f"  Multi-Quality ({q}): {n:,} ea")
        n_multiquality_total += n

    print(f"  Multi-Quality sum: {n_multiquality_total:,} ea "
          f"(an expected value: {n_standard} × 5 = {n_standard * 5:,})")

    # 3) Verification: Make sure it's exactly 5x
    expected = n_standard * len(MULTIQUALITY_QUALITY_FOLDERS)
    tolerance = 0.02  # Allow for minor differences (±2%) due to damaged images, etc
    if not (expected * (1 - tolerance) <= n_multiquality_total <= expected * (1 + tolerance)):
        raise RuntimeError(
            f" Suspected dataset contamination! The total number of Multi-Quality images ({n_multiquality_total:,}) differs significantly from the expected value ({expected:,}, Standard×5).\n"
        )

    # 4) reconfirm the YAML file itself (whether the training field is a list)
    with open(MULTIQUALITY_YAML, 'r') as f:
        mq_yaml = yaml.safe_load(f)
    if not isinstance(mq_yaml['train'], list):
        raise RuntimeError(
            f" visdrone_multiquality.yaml의 'train' field is not a list: "
        )

    print(f"\n Integrity verification passes.\n")


assert_dataset_integrity()


# Library Installation + Evaluation Function Definition
print(" Library installing...")
os.system("pip install -q ultralytics opencv-python-headless")

import torch
from ultralytics import YOLO

if torch.cuda.is_available():
    print(f" GPU: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
else:
    print(" GPU is not detected.")

SEED = 42


def evaluate_on_quality_levels(model_path, dataset_base_dir,
                                quality_levels=[20, 40, 60, 80, 100],
                                split='test', save_dir=None):
    print(f"\n{'='*80}\n Evaluation: {os.path.basename(model_path)} (split={split})\n{'='*80}")
    model = YOLO(model_path)
    results_dict = {}
    for quality in quality_levels:
        q_tag = f'q{quality}'
        temp_yaml_path = os.path.join(save_dir or '/tmp', f'temp_{split}_{q_tag}.yaml')
        temp_yaml = {
            'path': dataset_base_dir,
            'train': f'images/train/{q_tag}',   
            'val': f'images/{split}/{q_tag}',
            'test': f'images/{split}/{q_tag}',
            'nc': 10, 'names': CLASS_NAMES
        }
        with open(temp_yaml_path, 'w') as f:
            yaml.dump(temp_yaml, f)
        metrics = model.val(data=temp_yaml_path, split=split, batch=16, imgsz=640, verbose=False)
        results_dict[quality] = {
            'mAP50': float(metrics.box.map50), 'mAP50-95': float(metrics.box.map),
            'mAP75': float(metrics.box.map75), 'precision': float(metrics.box.p.mean()),
            'recall': float(metrics.box.r.mean()),
        }
        print(f"  Q{quality:>3}: mAP50={results_dict[quality]['mAP50']:.4f}  "
              f"mAP50-95={results_dict[quality]['mAP50-95']:.4f}")
    if os.path.exists(temp_yaml_path):
        os.remove(temp_yaml_path)
    return results_dict


def compute_robustness_score(results_dict):
    map50_values = np.array([v['mAP50'] for v in results_dict.values()])
    mean_map, sd_map = map50_values.mean(), map50_values.std()
    score = mean_map / sd_map if sd_map > 0 else float('inf')
    return {'mean_mAP50': float(mean_map), 'sd_mAP50': float(sd_map), 'robustness_score': float(score)}


def get_best_epoch_info(exp_folder_path):
    results_csv = os.path.join(exp_folder_path, 'results.csv')
    if not os.path.exists(results_csv):
        return None
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    map50_col = next((c for c in df.columns if 'mAP50(B)' in c and '95' not in c), None)
    map5095_col = next((c for c in df.columns if 'mAP50-95(B)' in c), None)
    df['_fitness'] = 0.1 * df[map50_col] + 0.9 * df[map5095_col]
    best_row = df.loc[df['_fitness'].idxmax()]
    return {
        'total_epoch': int(df['epoch'].iloc[-1]),
        'total_time_hr': float(df['time'].iloc[-1]) / 3600,
        'best_epoch': int(best_row['epoch']),
        'time_to_best_hr': float(best_row['time']) / 3600,
    }


print(" Evaluation Function Definition completed")


# Training setting
class TrainingConfig:
    EPOCHS = 100
    BATCH_SIZE = 48
    IMG_SIZE = 640
    OPTIMIZER = 'SGD'
    LR0 = 0.06
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    WARMUP_EPOCHS = 3.0
    WARMUP_MOMENTUM = 0.8
    WARMUP_BIAS_LR = 0.1
    PATIENCE = 50
    SAVE_PERIOD = 10
    WORKERS = 8

    @classmethod
    def to_dict(cls):
        return {
            'epochs': cls.EPOCHS, 'batch': cls.BATCH_SIZE, 'imgsz': cls.IMG_SIZE,
            'optimizer': cls.OPTIMIZER, 'lr0': cls.LR0, 'momentum': cls.MOMENTUM,
            'weight_decay': cls.WEIGHT_DECAY, 'warmup_epochs': cls.WARMUP_EPOCHS,
            'warmup_momentum': cls.WARMUP_MOMENTUM, 'warmup_bias_lr': cls.WARMUP_BIAS_LR,
            'patience': cls.PATIENCE, 'save_period': cls.SAVE_PERIOD, 'workers': cls.WORKERS,
        }


def train_one(weights_name, data_yaml, exp_name, experiments_dir, seed=SEED):
    print(f"\n{'='*80}\n🚀 {exp_name} 학습 시작\n   weights: {weights_name}\n   data: {data_yaml}\n{'='*80}")
    train_args = TrainingConfig.to_dict()
    train_args.update({'data': data_yaml, 'name': exp_name, 'project': experiments_dir,
                        'seed': seed, 'deterministic': True, 'exist_ok': True, 'verbose': True})
    try:
        model = YOLO(weights_name)
        model.train(**train_args)
    except torch.cuda.OutOfMemoryError:
        print("⚠️ OOM — batch/lr을 절반으로 낮춰 재시도")
        torch.cuda.empty_cache()
        train_args['batch'] = max(4, train_args['batch'] // 2)
        train_args['lr0'] = train_args['lr0'] / 2
        model = YOLO(weights_name)
        model.train(**train_args)
    return os.path.join(experiments_dir, exp_name, 'weights', 'best.pt')


# Running 4 experiments sequentially (YOLOv8n×2, YOLO11m×2)
ARCHITECTURES = {'YOLOv8n': 'yolov8n.pt', 'YOLO11m': 'yolo11m.pt'}
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

trained_models = {}
best_epoch_info = {}

print("\n" + "=" * 80)
print(" Running 4 experiments sequentially (YOLOv8n×2, YOLO11m×2)")
print("=" * 80)

for arch_name, weights_name in ARCHITECTURES.items():
    for tag, data_yaml in [('Standard', STANDARD_YAML), ('MultiQuality', MULTIQUALITY_YAML)]:
        exp_name = f'exp_{arch_name}_{tag}_a100_{timestamp}'
        model_path = train_one(weights_name, data_yaml, exp_name, EXPERIMENTS_DIR)
        key = f'{arch_name}-{tag}'
        trained_models[key] = model_path
        best_epoch_info[key] = get_best_epoch_info(os.path.join(EXPERIMENTS_DIR, exp_name))
        print(f" {key} Completed: best epoch={best_epoch_info[key]['best_epoch']}, "
              f"time={best_epoch_info[key]['total_time_hr']:.2f}hr")

print("\n Training completed! Trained Model:")
for k, v in trained_models.items():
    print(f"  {k}: {v}")


# Evaluation of 4 models
print("\n" + "=" * 80 + "\n Evaluation of 4 models (YOLOv8n, YOLO11m)\n" + "=" * 80)

all_results = {}
existing_path = os.path.join(EVAL_DIR, 'all_results.json')
if os.path.exists(existing_path):
    with open(existing_path, 'r') as f:
        loaded = json.load(f)
    for k, q_dict in loaded.items():
        all_results[k] = {int(q): v for q, v in q_dict.items()}
    print(f" Existing YOLOv8m results loaded: {list(all_results.keys())}")

for key, model_path in trained_models.items():
    all_results[key] = evaluate_on_quality_levels(
        model_path=model_path, dataset_base_dir=YOLO_DATASET_DIR,
        quality_levels=[20, 40, 60, 80, 100], split='test', save_dir=EVAL_DIR
    )

with open(os.path.join(EVAL_DIR, 'all_results_with_architectures.json'), 'w') as f:
    json.dump(all_results, f, indent=2)


# Comparison Table by Architecture
print("\n" + "=" * 80 + "\n Comparison Table by Architecture\n" + "=" * 80)

GPU_RATE = 1.86
rows = []
for arch_name in list(ARCHITECTURES.keys()):
    std_key, mq_key = f'{arch_name}-Standard', f'{arch_name}-MultiQuality'
    if std_key not in all_results or mq_key not in all_results:
        continue
    std_r = compute_robustness_score(all_results[std_key])
    mq_r = compute_robustness_score(all_results[mq_key])
    std_info, mq_info = best_epoch_info[std_key], best_epoch_info[mq_key]

    rows.append({
        'Architecture': arch_name,
        'Robustness (Standard)': round(std_r['robustness_score'], 2),
        'Robustness (Multi-Quality)': round(mq_r['robustness_score'], 2),
        'Improvement (x)': round(mq_r['robustness_score'] / std_r['robustness_score'], 2),
        'mAP50@Q20 (Standard)': round(all_results[std_key][20]['mAP50'], 4),
        'mAP50@Q20 (Multi-Quality)': round(all_results[mq_key][20]['mAP50'], 4),
        'Best Epoch (Std to MQ)': f"{std_info['best_epoch']} -> {mq_info['best_epoch']}",
        'Convergence Speed-up': round(std_info['best_epoch'] / mq_info['best_epoch'], 2),
        'Total Time hr (Std to MQ)': f"{std_info['total_time_hr']:.2f} -> {mq_info['total_time_hr']:.2f}",
        'Cost USD (Std to MQ)': f"{std_info['total_time_hr']*GPU_RATE:.2f} -> {mq_info['total_time_hr']*GPU_RATE:.2f}",
    })

df_arch = pd.DataFrame(rows)
print(df_arch.to_string(index=False))
df_arch.to_csv(os.path.join(EVAL_DIR, 'architecture_comparison_final.csv'), index=False)

print("\n" + "=" * 80)
print(" Architecture comparison xperiment completed!")
print("=" * 80)
