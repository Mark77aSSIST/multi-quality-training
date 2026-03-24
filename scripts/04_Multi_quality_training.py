# 4. YOLOv8 Multi Quality Training

import os
import yaml
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
from tqdm.auto import tqdm

# Setting class for YOLOv8 (A100 40GB optimization)
class TrainingConfig:

    MODEL_SIZE = 'm'
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
            'epochs': cls.EPOCHS,
            'batch': cls.BATCH_SIZE,
            'imgsz': cls.IMG_SIZE,
            'optimizer': cls.OPTIMIZER,
            'lr0': cls.LR0,
            'momentum': cls.MOMENTUM,
            'weight_decay': cls.WEIGHT_DECAY,
            'warmup_epochs': cls.WARMUP_EPOCHS,
            'warmup_momentum': cls.WARMUP_MOMENTUM,
            'warmup_bias_lr': cls.WARMUP_BIAS_LR,
            'patience': cls.PATIENCE,
            'save_period': cls.SAVE_PERIOD,
            'workers': cls.WORKERS,
        }

    @classmethod
    def get_test_config(cls, epochs=100):
        config = cls.to_dict()
        config['epochs'] = epochs
        
        return config

# Multi-Quality data preparation
# ============================================================================
def create_multiquality_txt_files(base_dir, output_dir):
    print(f"\n{'='*80}")
    print("Multi-Quality image directory lists Creation")
    print(f"{'='*80}")

    os.makedirs(output_dir, exist_ok=True)

    quality_levels = [20, 40, 60, 80, 100]

    # Train set
    print(f"\nTRAIN set directory collecting...")
    train_images = []

    for q in quality_levels:
        q_tag = f'q{q}'
        img_dir = os.path.join(base_dir, 'images', 'train', q_tag)

        if os.path.exists(img_dir):
            img_files = sorted([f for f in os.listdir(img_dir)
                               if f.endswith(('.jpg', '.png'))])

            for img_file in img_files:
                abs_path = os.path.abspath(os.path.join(img_dir, img_file))
                train_images.append(abs_path)

            print(f"  Q{q}%: {len(img_files)} image")

    # Train list file Save
    train_txt = os.path.join(output_dir, 'train.txt')
    with open(train_txt, 'w') as f:
        f.write('\n'.join(train_images))

    print(f"\nTrain list file Creation: {train_txt}")
    print(f"   Total {len(train_images)} image")

    # Val set - only use a raw quality
    val_dir = os.path.join(base_dir, 'images', 'val', 'q100')

    # Dataset YAML Creation
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    dataset_config = {
        'path': base_dir,
        'train': train_txt,
        'val': 'images/val/q100',
        'nc': 10,
        'names': ['pedestrian', 'people', 'bicycle', 'car', 'van',
                 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)

    print(f"\nDataset YAML Creation: {yaml_path}")

    return yaml_path

# YOLOv8 model Training function (A100 optimization)
def train_yolov8(dataset_yaml, experiment_name, project_dir, pretrained=True, config=None):

    print(f"\n{'='*80}")
    print(f"YOLOv8 Training Start (A100 40GB)")
    print(f"  Experiment Name: {experiment_name}")
    print(f"  Dataset: {dataset_yaml}")
    print(f"{'='*80}")

    model_name = f'yolov8{TrainingConfig.MODEL_SIZE}.pt'

    if pretrained:
        print(f"\nPretrained model Load: {model_name}")
        model = YOLO(model_name)
    else:
        model = YOLO(f'yolov8{TrainingConfig.MODEL_SIZE}.yaml')

    if config is None:
        config = TrainingConfig.to_dict()

    print(f"\nHyperparameters for Training:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    print(f"\nTraining Start...")

    results = model.train(
        data=dataset_yaml,
        name=experiment_name,
        project=project_dir,
        exist_ok=True,
        pretrained=pretrained,
        **config
    )

    print(f"\nTraining Complete")
    print(f"Result: {os.path.join(project_dir, experiment_name)}")

    return model, results

# Run an experiment
if __name__ == '__main__':

    TrainingConfig.print_config()

    print("=" * 80)
    print("Experiment of Multi-Quality Training (A100 40GB optimization)")
    print("=" * 80)

    EXPERIMENTS_DIR = os.path.join(config['EXPERIMENT_PATH'], 'experiments')
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    # Experiment 1: Standard Training (Baseline)
    print("\n" + "=" * 80)
    print("Experiment 1: Standard Training (100% quality only)")
    print("=" * 80)

    exp1_name = f"exp1_standard_a100_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    test_config = TrainingConfig.get_test_config(epochs=100)

    print("\nTraining Mode (100 epochs)")
    print(f"   Batch size: {test_config['batch']} (A100 optimization)")

    model_standard, results_standard = train_yolov8(
        dataset_yaml=config['STANDARD_YAML'],
        experiment_name=exp1_name,
        project_dir=EXPERIMENTS_DIR,
        pretrained=True,
        config=test_config
    )

    # Experiment 2: Multi-Quality Training
    print("\n" + "=" * 80)
    print("🔬 Experiment 2: Multi-Quality Training (5 levels)")
    print("=" * 80)

    multiquality_dir = os.path.join(config['EXPERIMENT_PATH'], 'multiquality_a100')
    multiquality_yaml = create_multiquality_txt_files(
        base_dir=config['YOLO_DATASET_DIR'],
        output_dir=multiquality_dir
    )

    exp2_name = f"exp2_multiquality_a100_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("\nTraining Mode (100 epochs)")
    print(f"   Data: 32,355 images (5-levels quality)")
    print(f"   Batch size: {test_config['batch']}")

    model_multiquality, results_multiquality = train_yolov8(
        dataset_yaml=multiquality_yaml,
        experiment_name=exp2_name,
        project_dir=EXPERIMENTS_DIR,
        pretrained=True,
        config=test_config
    )

    # Experiment results Save
    config['MODEL_STANDARD'] = os.path.join(EXPERIMENTS_DIR, exp1_name, 'weights', 'best.pt')
    config['MODEL_MULTIQUALITY'] = os.path.join(EXPERIMENTS_DIR, exp2_name, 'weights', 'best.pt')

    print(f"\nTrained Model:")
    print(f"  Standard: {config['MODEL_STANDARD']}")
    print(f"  Multi-Quality: {config['MODEL_MULTIQUALITY']}")