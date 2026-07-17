import os
import re
import glob
import random
import json
import yaml
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm


# Google Drive mount & path setup for new session
from google.colab import drive
drive.mount('/content/drive')

GDRIVE_ROOT = '/content/drive/MyDrive'
DATASET_PATH = os.path.join(GDRIVE_ROOT, 'VisDrone')
EXPERIMENT_PATH = os.path.join(GDRIVE_ROOT, 'CCTV_MultiRes_Experiments')

assert os.path.exists(EXPERIMENT_PATH), \
    f" Can't find the experiment path: {EXPERIMENT_PATH}\n "

config = {
    'GDRIVE_ROOT': GDRIVE_ROOT,
    'DATASET_PATH': DATASET_PATH,
    'EXPERIMENT_PATH': EXPERIMENT_PATH,
    'TRAIN_DIR': os.path.join(DATASET_PATH, 'VisDrone2019-DET-train'),
    'VAL_DIR': os.path.join(DATASET_PATH, 'VisDrone2019-DET-val'),
    'TEST_DIR': os.path.join(DATASET_PATH, 'VisDrone2019-DET-test-dev'),
    'YOLO_DATASET_DIR': os.path.join(EXPERIMENT_PATH, 'yolo_dataset'),
}

assert os.path.exists(config['YOLO_DATASET_DIR']), \
    f" Existing YOLO dataset not found: {config['YOLO_DATASET_DIR']}\n" \

print(" Drive mount and existing path verification completed")
print(f"  EXPERIMENT_PATH : {config['EXPERIMENT_PATH']}")
print(f"  YOLO_DATASET_DIR: {config['YOLO_DATASET_DIR']}")


# Library install for new session
print("\n Installing required libraries...")
os.system("pip install -q ultralytics opencv-python-headless albumentations")
print(" Library install complete!")

import torch
from ultralytics import YOLO
import albumentations as A

if torch.cuda.is_available():
    print(f" GPU: {torch.cuda.get_device_name(0)}  "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
else:
    print(" GPU not detected")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

YOLO_DATASET_DIR = config['YOLO_DATASET_DIR']
TRAIN_DIR = config['TRAIN_DIR']
CLASS_NAMES = ['pedestrian', 'people', 'bicycle', 'car', 'van',
               'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']


# VisDroneParser / QualityDegrader redefine for new session
class VisDroneParser:
    CLASS_NAMES = {
        0: 'ignored', 1: 'pedestrian', 2: 'people', 3: 'bicycle', 4: 'car',
        5: 'van', 6: 'truck', 7: 'tricycle', 8: 'awning-tricycle',
        9: 'bus', 10: 'motor', 11: 'others'
    }
    VALID_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    @staticmethod
    def parse_annotation(annotation_file):
        annotations = []
        if not os.path.exists(annotation_file):
            return annotations
        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                bbox_left, bbox_top, bbox_width, bbox_height = map(int, parts[:4])
                object_category = int(parts[5])
                if object_category == 0 or bbox_width <= 0 or bbox_height <= 0:
                    continue
                if object_category in VisDroneParser.VALID_CLASSES:
                    annotations.append({
                        'bbox': [bbox_left, bbox_top, bbox_width, bbox_height],
                        'class': object_category,
                    })
        return annotations

    @staticmethod
    def visdrone_to_yolo(annotations, img_width, img_height):
        yolo_annotations = []
        for ann in annotations:
            x, y, w, h = ann['bbox']
            class_id = ann['class']
            yolo_class = VisDroneParser.VALID_CLASSES.index(class_id)
            x_center = max(0, min(1, (x + w / 2) / img_width))
            y_center = max(0, min(1, (y + h / 2) / img_height))
            norm_w = max(0, min(1, w / img_width))
            norm_h = max(0, min(1, h / img_height))
            yolo_annotations.append({'class': yolo_class,
                                      'bbox': [x_center, y_center, norm_w, norm_h]})
        return yolo_annotations


class QualityDegrader:
    QUALITY_LEVELS = [0.2, 0.4, 0.6, 0.8, 1.0]
    JPEG_QUALITY_MAP = {0.2: 20, 0.4: 40, 0.6: 60, 0.8: 80, 1.0: 100}

    @staticmethod
    def get_jpeg_quality(quality_level):
        if quality_level in QualityDegrader.JPEG_QUALITY_MAP:
            return QualityDegrader.JPEG_QUALITY_MAP[quality_level]
        levels = sorted(QualityDegrader.JPEG_QUALITY_MAP.keys())
        if quality_level <= levels[0]:
            return QualityDegrader.JPEG_QUALITY_MAP[levels[0]]
        if quality_level >= levels[-1]:
            return QualityDegrader.JPEG_QUALITY_MAP[levels[-1]]
        for lo, hi in zip(levels[:-1], levels[1:]):
            if lo <= quality_level <= hi:
                q_lo, q_hi = QualityDegrader.JPEG_QUALITY_MAP[lo], QualityDegrader.JPEG_QUALITY_MAP[hi]
                ratio = (quality_level - lo) / (hi - lo)
                return int(round(q_lo + (q_hi - q_lo) * ratio))
        return int(np.clip(quality_level * 100, 5, 95))

    @staticmethod
    def apply_jpeg_compression(image, jpeg_quality):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
        success, encoded_buf = cv2.imencode('.jpg', image, encode_param)
        if not success:
            return image.copy()
        return cv2.imdecode(encoded_buf, cv2.IMREAD_COLOR)

    @staticmethod
    def degrade_image(image, quality_level, apply_jpeg=True, jpeg_quality=None):
        if quality_level == 1.0:
            degraded = image.copy()
        else:
            h, w = image.shape[:2]
            new_h, new_w = max(1, int(h * quality_level)), max(1, int(w * quality_level))
            downsampled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            degraded = cv2.resize(downsampled, (w, h), interpolation=cv2.INTER_CUBIC)
        if apply_jpeg:
            q = jpeg_quality if jpeg_quality is not None else QualityDegrader.get_jpeg_quality(quality_level)
            degraded = QualityDegrader.apply_jpeg_compression(degraded, q)
        return degraded


print(" VisDroneParser / QualityDegrader redefine completed")


# Evaluation / Visualization function redefine for new session
def evaluate_on_quality_levels(model_path, dataset_base_dir,
                                quality_levels=[20, 40, 60, 80, 100],
                                split='test', save_dir=None):
    print(f"\n{'='*80}\n Multi-Quality Evaluation\n  model: {os.path.basename(model_path)}\n"
          f"  Split: {split}\n  image quality level: {quality_levels}\n{'='*80}")

    model = YOLO(model_path)
    results_dict = {}

    for quality in quality_levels:
        print(f"\n Quality {quality}% evaluation...")
        q_tag = f'q{quality}'
        temp_yaml_path = os.path.join(save_dir or '/tmp', f'temp_{split}_{q_tag}.yaml')
        temp_yaml = {
            'path': dataset_base_dir,
            'train': f'images/train/{q_tag}',
            'val': f'images/{split}/{q_tag}',
            'test': f'images/{split}/{q_tag}',
            'nc': 10,
            'names': CLASS_NAMES
        }
        with open(temp_yaml_path, 'w') as f:
            yaml.dump(temp_yaml, f)

        metrics = model.val(data=temp_yaml_path, split=split, batch=16, imgsz=640, verbose=False)

        results_dict[quality] = {
            'mAP50': float(metrics.box.map50),
            'mAP50-95': float(metrics.box.map),
            'mAP75': float(metrics.box.map75),
            'precision': float(metrics.box.p.mean()),
            'recall': float(metrics.box.r.mean()),
        }
        print(f"  mAP50: {results_dict[quality]['mAP50']:.4f}   mAP50-95: {results_dict[quality]['mAP50-95']:.4f}")

    if os.path.exists(temp_yaml_path):
        os.remove(temp_yaml_path)

    return results_dict


def plot_quality_performance(results_dict, model_names, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics_to_plot = ['mAP50-95', 'mAP50', 'mAP75']
    quality_levels = sorted(list(results_dict[model_names[0]].keys()))

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        for model_name in model_names:
            values = [results_dict[model_name][q][metric] for q in quality_levels]
            ax.plot(quality_levels, values, marker='o', linewidth=2, label=model_name, markersize=8)
        ax.set_xlabel('Quality Level (%)', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'{metric} vs Quality Level', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(quality_levels)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" graph save: {save_path}")
    plt.show()


print(" evaluate_on_quality_levels / plot_quality_performance redefine completed")


# Automatic discovery of existing learning completion models (Standard, Multi-Quality)
EXPERIMENTS_DIR = os.path.join(config['EXPERIMENT_PATH'], 'experiments')
EVAL_DIR = os.path.join(config['EXPERIMENT_PATH'], 'evaluation_results')
os.makedirs(EVAL_DIR, exist_ok=True)


def find_latest_model(pattern):
    candidates = glob.glob(os.path.join(EXPERIMENTS_DIR, pattern, 'weights', 'best.pt'))
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


config['MODEL_STANDARD'] = find_latest_model('exp1_standard*')
config['MODEL_MULTIQUALITY'] = find_latest_model('exp2_multiquality*')

print("\n detection result of existing learning completion models:")
print(f"  Standard      : {config['MODEL_STANDARD']}")
print(f"  Multi-Quality : {config['MODEL_MULTIQUALITY']}")

assert config['MODEL_STANDARD'] and os.path.exists(config['MODEL_STANDARD']), \
    " Cannot find the Standard model(best.pt) at the experiments folder. "
assert config['MODEL_MULTIQUALITY'] and os.path.exists(config['MODEL_MULTIQUALITY']), \
    " Cannot find the Multi-Quality model(best.pt) at the experiments folder. "

# Load existing evaluation results (if any, skip re-evaluation)
all_results_path_existing = os.path.join(EVAL_DIR, 'all_results.json')
all_results = {}

if os.path.exists(all_results_path_existing):
    with open(all_results_path_existing, 'r') as f:
        loaded = json.load(f)
    for model_name, q_dict in loaded.items():
        all_results[model_name] = {int(q): v for q, v in q_dict.items()}
    print(f"\n Load existing evaluation results Completed: {all_results_path_existing}")
    print(f"   Loaded Model: {list(all_results.keys())}")
else:
    print(f"\n Cannot find the existing evaluation results file: {all_results_path_existing}")
    print("   New evaluation start of Standard / Multi-Quality (No re-training, only evaluation).")
    for model_name, model_path in [('Standard (Q100)', config['MODEL_STANDARD']),
                                    ('Multi-Quality', config['MODEL_MULTIQUALITY'])]:
        all_results[model_name] = evaluate_on_quality_levels(
            model_path=model_path,
            dataset_base_dir=YOLO_DATASET_DIR,
            quality_levels=[20, 40, 60, 80, 100],
            split='test',
            save_dir=EVAL_DIR
        )

print("\n Complete and generate the dataset of New Baselines.")


# Baseline B1: Q100-Augmented-5x Dataset Generation
def build_augmentation_pipelines():
    HSV_H_BASE, HSV_S_BASE, HSV_V_BASE = 0.015, 0.7, 0.4
    TRANSLATE_BASE, SCALE_BASE = 0.1, 0.5
    FLIPLR_P, FLIPUD_P = 0.5, 0.0
    STRENGTH_MULTIPLIERS = [0.2, 0.4, 0.6, 0.8, 1.0]

    bbox_params = A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.2)

    pipelines = []
    for m in STRENGTH_MULTIPLIERS:
        translate_m, scale_m = TRANSLATE_BASE * m, SCALE_BASE * m
        hsv_h_m, hsv_s_m, hsv_v_m = HSV_H_BASE * m, HSV_S_BASE * m, HSV_V_BASE * m

        transforms = [A.HorizontalFlip(p=FLIPLR_P)]
        if FLIPUD_P > 0:
            transforms.append(A.VerticalFlip(p=FLIPUD_P))

        transforms.append(A.Affine(
            scale=(1 - scale_m, 1 + scale_m),
            translate_percent=(-translate_m, translate_m),
            rotate=(0, 0), shear=(0, 0), p=1.0
        ))
        transforms.append(A.ColorJitter(
            brightness=hsv_v_m, contrast=0.0, saturation=hsv_s_m,
            hue=min(hsv_h_m, 0.5), p=1.0
        ))
        pipelines.append(A.Compose(transforms, bbox_params=bbox_params))

    return pipelines


def generate_q100_augmented_dataset(visdrone_dir, output_dir, split_name,
                                     subfolder_name='q100_aug5x', limit=None):
    images_dir = os.path.join(visdrone_dir, 'images')
    annotations_dir = os.path.join(visdrone_dir, 'annotations')
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])
    if limit:
        image_files = image_files[:limit]

    pipelines = build_augmentation_pipelines()
    num_copies = len(pipelines)

    print(f"\n{'='*80}\n {subfolder_name} generating: {len(image_files)} image × {num_copies} augmentation level")
    print(f"   (No quality degradation / Q100 originals, 20-100% multiple of the paper's actual hyp)\n{'='*80}")

    for copy_idx in range(num_copies):
        os.makedirs(os.path.join(output_dir, 'images', split_name, subfolder_name, f'copy{copy_idx}'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split_name, subfolder_name, f'copy{copy_idx}'), exist_ok=True)

    stats = {'processed': 0, 'skipped': 0, 'total_objects': 0}

    for img_file in tqdm(image_files, desc=f"{subfolder_name} generation"):
        img_path = os.path.join(images_dir, img_file)
        ann_file = os.path.join(annotations_dir, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))
        img_bgr = cv2.imread(img_path)
        if img_bgr is None or not os.path.exists(ann_file):
            stats['skipped'] += 1
            continue

        h, w = img_bgr.shape[:2]
        annotations = VisDroneParser.parse_annotation(ann_file)
        if len(annotations) == 0:
            stats['skipped'] += 1
            continue

        yolo_annotations = VisDroneParser.visdrone_to_yolo(annotations, w, h)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        bboxes = [ann['bbox'] for ann in yolo_annotations]
        class_labels = [ann['class'] for ann in yolo_annotations]

        for copy_idx, pipeline in enumerate(pipelines):
            try:
                augmented = pipeline(image=img_rgb, bboxes=bboxes, class_labels=class_labels)
            except Exception:
                augmented = {'image': img_rgb, 'bboxes': bboxes, 'class_labels': class_labels}

            aug_img_bgr = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
            out_img_path = os.path.join(output_dir, 'images', split_name, subfolder_name, f'copy{copy_idx}', img_file)
            cv2.imwrite(out_img_path, aug_img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            out_lbl_path = os.path.join(output_dir, 'labels', split_name, subfolder_name, f'copy{copy_idx}',
                                         img_file.replace('.jpg', '.txt').replace('.png', '.txt'))
            with open(out_lbl_path, 'w') as f:
                for cid, (x, y, wn, hn) in zip(augmented['class_labels'], augmented['bboxes']):
                    f.write(f"{cid} {x:.6f} {y:.6f} {wn:.6f} {hn:.6f}\n")

        stats['processed'] += 1
        stats['total_objects'] += len(yolo_annotations) * num_copies

    print(f"\n {subfolder_name} complete: processed {stats['processed']} → Total {stats['processed']*num_copies}images, "
          f"skipped {stats['skipped']}")
    return stats


# B2/B3 common: Random quality (resolution + JPEG) dataset generation
def generate_random_quality_dataset(visdrone_dir, output_dir, split_name, subfolder_name,
                                     num_copies, quality_range=(0.15, 1.0), limit=None):
    images_dir = os.path.join(visdrone_dir, 'images')
    annotations_dir = os.path.join(visdrone_dir, 'annotations')
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])
    if limit:
        image_files = image_files[:limit]

    print(f"\n{'='*80}\n {subfolder_name} generating: {len(image_files)} image × {num_copies} copies")
    print(f"   quality_level ∈ [{quality_range[0]}, {quality_range[1]}] Equal distribution random\n{'='*80}")

    stats = {'processed': 0, 'skipped': 0, 'total_objects': 0, 'sampled_qualities': []}
    for copy_idx in range(num_copies):
        os.makedirs(os.path.join(output_dir, 'images', split_name, subfolder_name, f'copy{copy_idx}'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split_name, subfolder_name, f'copy{copy_idx}'), exist_ok=True)

    for img_file in tqdm(image_files, desc=f"{subfolder_name} generation"):
        img_path = os.path.join(images_dir, img_file)
        ann_file = os.path.join(annotations_dir, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))
        img = cv2.imread(img_path)
        if img is None or not os.path.exists(ann_file):
            stats['skipped'] += 1
            continue

        h, w = img.shape[:2]
        annotations = VisDroneParser.parse_annotation(ann_file)
        if len(annotations) == 0:
            stats['skipped'] += 1
            continue

        yolo_annotations = VisDroneParser.visdrone_to_yolo(annotations, w, h)

        for copy_idx in range(num_copies):
            q_level = float(np.random.uniform(quality_range[0], quality_range[1]))
            stats['sampled_qualities'].append(q_level)
            jpeg_q = QualityDegrader.get_jpeg_quality(q_level)
            degraded_img = QualityDegrader.degrade_image(img, q_level, apply_jpeg=True, jpeg_quality=jpeg_q)

            out_img_path = os.path.join(output_dir, 'images', split_name, subfolder_name, f'copy{copy_idx}', img_file)
            cv2.imwrite(out_img_path, degraded_img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])

            out_lbl_path = os.path.join(output_dir, 'labels', split_name, subfolder_name, f'copy{copy_idx}',
                                         img_file.replace('.jpg', '.txt').replace('.png', '.txt'))
            with open(out_lbl_path, 'w') as f:
                for ann in yolo_annotations:
                    cid = ann['class']
                    x, y, wn, hn = ann['bbox']
                    f.write(f"{cid} {x:.6f} {y:.6f} {wn:.6f} {hn:.6f}\n")

        stats['processed'] += 1
        stats['total_objects'] += len(yolo_annotations) * num_copies

    q_arr = np.array(stats['sampled_qualities'])
    print(f"\n {subfolder_name} complete: processed {stats['processed']} → Total {stats['processed']*num_copies}images, "
          f"skipped {stats['skipped']}, quality mean={q_arr.mean():.3f} std={q_arr.std():.3f}")
    return stats


print("=" * 80)
print(" Start the generation of Fair Baseline dataset (B1, B2, B3)")
print("=" * 80)

b1_stats = generate_q100_augmented_dataset(
    visdrone_dir=TRAIN_DIR, output_dir=YOLO_DATASET_DIR, split_name='train',
    subfolder_name='q100_aug5x', limit=None
)
b2_stats = generate_random_quality_dataset(
    visdrone_dir=TRAIN_DIR, output_dir=YOLO_DATASET_DIR, split_name='train',
    subfolder_name='random1x', num_copies=1, quality_range=(0.15, 1.0), limit=None
)
b3_stats = generate_random_quality_dataset(
    visdrone_dir=TRAIN_DIR, output_dir=YOLO_DATASET_DIR, split_name='train',
    subfolder_name='random5x', num_copies=5, quality_range=(0.15, 1.0), limit=None
)


# Baseline dataset YAML generation
def make_baseline_yaml(subfolder_name, yaml_filename):
    baseline_yaml = {
        'path': YOLO_DATASET_DIR,
        'train': f'images/train/{subfolder_name}',
        'val': 'images/val/q100',
        'test': 'images/test',
        'nc': len(CLASS_NAMES),
        'names': CLASS_NAMES
    }
    yaml_path = os.path.join(YOLO_DATASET_DIR, yaml_filename)
    with open(yaml_path, 'w') as f:
        yaml.dump(baseline_yaml, f, default_flow_style=False)
    print(f" YAML generation: {yaml_path}")
    return yaml_path

config['B1_YAML'] = make_baseline_yaml('q100_aug5x', 'visdrone_b1_q100aug5x.yaml')
config['B2_YAML'] = make_baseline_yaml('random1x', 'visdrone_b2_random1x.yaml')
config['B3_YAML'] = make_baseline_yaml('random5x', 'visdrone_b3_random5x.yaml')

print("\n Completed! Go to training step")


# Baseline model training
class BaselineTrainingConfig:
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


def train_baseline(exp_name, data_yaml, experiments_dir, seed=SEED):
    print(f"\n{'='*80}\n🚀 Baseline 학습 시작: {exp_name}\n   데이터셋: {data_yaml}\n{'='*80}")
    model = YOLO('yolov8m.pt')
    train_args = BaselineTrainingConfig.to_dict()
    train_args.update({'data': data_yaml, 'name': exp_name, 'project': experiments_dir,
                        'seed': seed, 'deterministic': True, 'exist_ok': True, 'verbose': True})
    model.train(**train_args)
    return os.path.join(experiments_dir, exp_name, 'weights', 'best.pt')


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
exp_b1_name = f'expB1_q100aug5x_a100_{timestamp}'
exp_b2_name = f'expB2_random1x_a100_{timestamp}'
exp_b3_name = f'expB3_random5x_a100_{timestamp}'

print("\n" + "=" * 80 + "\n🔬 Experiment B1: Q100-Augmented-5x\n" + "=" * 80)
config['MODEL_B1'] = train_baseline(exp_b1_name, config['B1_YAML'], EXPERIMENTS_DIR)

print("\n" + "=" * 80 + "\n🔬 Experiment B2: Random-Quality-1x\n" + "=" * 80)
config['MODEL_B2'] = train_baseline(exp_b2_name, config['B2_YAML'], EXPERIMENTS_DIR)

print("\n" + "=" * 80 + "\n🔬 Experiment B3: Random-Quality-5x\n" + "=" * 80)
config['MODEL_B3'] = train_baseline(exp_b3_name, config['B3_YAML'], EXPERIMENTS_DIR)

print("\n Baseline model training completed!")
print(f"  B1: {config['MODEL_B1']}\n  B2: {config['MODEL_B2']}\n  B3: {config['MODEL_B3']}")


# Evaluation of 5 models
print("\n" + "=" * 80 + "\n Evaluation of 5 models\n" + "=" * 80)

models_to_evaluate = {
    'B1 (Q100-Aug-5x)': config['MODEL_B1'],
    'B2 (Random-1x)': config['MODEL_B2'],
    'B3 (Random-5x)': config['MODEL_B3'],
}

for model_name, model_path in models_to_evaluate.items():
    print(f"\n{'='*80}\n {model_name} evaluation\n{'='*80}")
    results = evaluate_on_quality_levels(
        model_path=model_path, dataset_base_dir=YOLO_DATASET_DIR,
        quality_levels=[20, 40, 60, 80, 100], split='test', save_dir=EVAL_DIR
    )
    all_results[model_name] = results
    result_file = os.path.join(EVAL_DIR, f'{model_name.replace(" ", "_").replace("(", "").replace(")", "")}_results.json')
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f" Result saved: {result_file}")

all_results_path = os.path.join(EVAL_DIR, 'all_results_with_baselines.json')
with open(all_results_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\n Result saved for All of 5 models : {all_results_path}")


# 5-Model Comparison Tables and Graphs + Diagnosis
print("\n" + "=" * 80 + "\n Standard → B1 → B2 → B3 → Multi-Quality Step-by-step comparison\n" + "=" * 80)

ordered_model_names = ['Standard (Q100)', 'B1 (Q100-Aug-5x)', 'B2 (Random-1x)',
                        'B3 (Random-5x)', 'Multi-Quality']
ordered_model_names = [m for m in ordered_model_names if m in all_results]

fair_plot_path = os.path.join(EVAL_DIR, 'fair_baseline_comparison_v2.png')
plot_quality_performance(all_results, ordered_model_names, save_path=fair_plot_path)

rows = []
for model_name in ordered_model_names:
    for q, m in all_results[model_name].items():
        rows.append({'Model': model_name, 'Quality (%)': q, **m})
df_all = pd.DataFrame(rows)
pivot_map5095 = df_all.pivot_table(values='mAP50-95', index='Model', columns='Quality (%)').reindex(ordered_model_names)

print("\n" + "=" * 80 + "\n mAP50-95 Comprehensive comparison\n" + "=" * 80)
print(pivot_map5095.to_string())

csv_path = os.path.join(EVAL_DIR, 'fair_baseline_comparison_table_v2.csv')
df_all.to_csv(csv_path, index=False)
print(f"\n CSV saved: {csv_path}")

print("\n" + "=" * 80 + "\n Diagnosis 1: Pure augmentation (B1) effect (not exposed to quality degradation)\n" + "=" * 80)
if 'Standard (Q100)' in all_results and 'B1 (Q100-Aug-5x)' in all_results:
    for q in [20, 40, 60, 80, 100]:
        std = all_results['Standard (Q100)'][q]['mAP50-95']
        b1 = all_results['B1 (Q100-Aug-5x)'][q]['mAP50-95']
        print(f"  Q{q:>3}: B1({b1:.4f}) - Standard({std:.4f}) = {b1-std:+.4f}")

print("\n" + "=" * 80 + "\n Diagnostics 2: Effectiveness of Structural Five-Step Design Only in Same Data Volume (5x) Conditions\n" + "=" * 80)
if 'B3 (Random-5x)' in all_results and 'Multi-Quality' in all_results:
    for q in [20, 40, 60, 80, 100]:
        b3 = all_results['B3 (Random-5x)'][q]['mAP50-95']
        multiq = all_results['Multi-Quality'][q]['mAP50-95']
        print(f"  Q{q:>3}: Multi-Quality({multiq:.4f}) - B3({b3:.4f}) = {multiq-b3:+.4f}")

print("\n" + "=" * 80 + "\n Diagnosis 3: Low cost (1x) random quality augmentation effect\n" + "=" * 80)
if 'Standard (Q100)' in all_results and 'B2 (Random-1x)' in all_results:
    for q in [20, 40, 60, 80, 100]:
        std = all_results['Standard (Q100)'][q]['mAP50-95']
        b2 = all_results['B2 (Random-1x)'][q]['mAP50-95']
        print(f"  Q{q:>3}: B2({b2:.4f}) - Standard({std:.4f}) = {b2-std:+.4f}")

print("\n" + "=" * 80)
print(" Fair Baseline Comparison Experiments Completed!")
print("=" * 80)
print(f"\n Results saved path: {EVAL_DIR}")