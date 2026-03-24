# 3. Data preprocessing

import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import yaml

# Create a directory structure for the yolo format dataset
def create_yolo_structure(base_path, include_multiquality=True):
    splits = ['train', 'val', 'test']

    for split in splits:
        if include_multiquality:
            for quality in [20, 40, 60, 80, 100]:
                os.makedirs(os.path.join(base_path, 'images', split, f'q{quality}'), exist_ok=True)
                os.makedirs(os.path.join(base_path, 'labels', split, f'q{quality}'), exist_ok=True)
        else:
            os.makedirs(os.path.join(base_path, 'images', split), exist_ok=True)
            os.makedirs(os.path.join(base_path, 'labels', split), exist_ok=True)

    print(f"Create a directory structure Complete: {base_path}")

# Reformating from VisDrone to YOLO & Multi-Quality Generation
def convert_visdrone_to_yolo_multiquality(
    visdrone_dir,
    output_dir,
    split_name,
    quality_levels=[0.2, 0.4, 0.6, 0.8, 1.0],
    limit=None
):
    images_dir = os.path.join(visdrone_dir, 'images')
    annotations_dir = os.path.join(visdrone_dir, 'annotations')

    if not os.path.exists(images_dir):
        print(f"Can't find the image directory: {images_dir}")
        return

    # Image files list
    image_files = sorted([f for f in os.listdir(images_dir)
                         if f.endswith(('.jpg', '.png'))])

    if limit:
        image_files = image_files[:limit]

    print(f"\n{'='*80}")
    print(f"🔄 {split_name.upper()} set reformating: {len(image_files)} image")
    print(f"{'='*80}")

    stats = {
        'processed': 0,
        'skipped': 0,
        'total_objects': 0
    }

    for img_file in tqdm(image_files, desc=f"{split_name} transformation"):
        img_path = os.path.join(images_dir, img_file)
        ann_file = os.path.join(annotations_dir,
                               img_file.replace('.jpg', '.txt').replace('.png', '.txt'))

        # image loading
        img = cv2.imread(img_path)
        if img is None:
            stats['skipped'] += 1
            continue

        h, w = img.shape[:2]

        # Annotation parsing
        if not os.path.exists(ann_file):
            stats['skipped'] += 1
            continue

        annotations = VisDroneParser.parse_annotation(ann_file)
        if len(annotations) == 0:
            stats['skipped'] += 1
            continue

        # Reformating to YOLO
        yolo_annotations = VisDroneParser.visdrone_to_yolo(annotations, w, h)

        # Generation to 5-levels quality
        for quality in quality_levels:
            quality_tag = f'q{int(quality * 100)}'

            degraded_img = QualityDegrader.degrade_image(img, quality)

            output_img_dir = os.path.join(output_dir, 'images', split_name, quality_tag)
            output_img_path = os.path.join(output_img_dir, img_file)
            cv2.imwrite(output_img_path, degraded_img)

            output_label_dir = os.path.join(output_dir, 'labels', split_name, quality_tag)
            output_label_path = os.path.join(output_label_dir,
                                            img_file.replace('.jpg', '.txt').replace('.png', '.txt'))

            with open(output_label_path, 'w') as f:
                for ann in yolo_annotations:
                    class_id = ann['class']
                    x, y, w_norm, h_norm = ann['bbox']
                    f.write(f"{class_id} {x:.6f} {y:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        stats['processed'] += 1
        stats['total_objects'] += len(yolo_annotations)

    print(f"\n{split_name} transformation Complete:")
    print(f"  Processed: {stats['processed']} image")
    print(f"  Skipped: {stats['skipped']} image")
    print(f"  Total objects: {stats['total_objects']}")

    return stats

# Creating a dataset setup file (YAML) for YOLOv8 training
def create_dataset_yaml(output_dir, dataset_name='visdrone_multiquality'):
    class_names = [
        'pedestrian',  # 1
        'people',      # 2
        'bicycle',     # 3
        'car',         # 4
        'van',         # 5
        'truck',       # 6
        'tricycle',    # 7
        'awning-tricycle',  # 8
        'bus',         # 9
        'motor'        # 10
    ]

    # Standard dataset (Only using q100)
    standard_yaml = {
        'path': output_dir,
        'train': 'images/train/q100',
        'val': 'images/val/q100',
        'test': 'images/test/q100',
        'nc': len(class_names),
        'names': class_names
    }

    yaml_path = os.path.join(output_dir, 'visdrone_standard.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(standard_yaml, f, default_flow_style=False)
    print(f"Standard dataset YAML Creation: {yaml_path}")

    # Multi-quality dataset
    multiquality_yaml = {
        'path': output_dir,
        'train': 'images/train',
        'val': 'images/val/q100',
        'test': 'images/test',
        'nc': len(class_names),
        'names': class_names
    }

    yaml_path = os.path.join(output_dir, 'visdrone_multiquality.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(multiquality_yaml, f, default_flow_style=False)
    print(f"📝 Multi-quality dataset YAML 생성: {yaml_path}")

    return standard_yaml, multiquality_yaml

# Dataset conversion
print("=" * 80)
print("VisDrone → YOLO Multi-Quality Dataset conversion Start")
print("=" * 80)

YOLO_DATASET_DIR = os.path.join(config['EXPERIMENT_PATH'], 'yolo_dataset')   # Output directory

create_yolo_structure(YOLO_DATASET_DIR, include_multiquality=True)   # Dataset structure

CONVERT_OPTIONS = {
    'quality_levels': [0.2, 0.4, 0.6, 0.8, 1.0],
    'limit_train': None,    # None = All, Number = limit (for Test: 100)
    'limit_val': None,      # None = All
    'limit_test': None,     # None = All
}

print(f"\nConversion option:")
print(f"  Image quality level: {[int(q*100) for q in CONVERT_OPTIONS['quality_levels']]}")
print(f"  Train limit: {CONVERT_OPTIONS['limit_train'] or 'All'}")
print(f"  Val limit: {CONVERT_OPTIONS['limit_val'] or 'All'}")
print(f"  Test limit: {CONVERT_OPTIONS['limit_test'] or 'All'}")

# Train set Conversion
train_stats = convert_visdrone_to_yolo_multiquality(
    visdrone_dir=config['TRAIN_DIR'],
    output_dir=YOLO_DATASET_DIR,
    split_name='train',
    quality_levels=CONVERT_OPTIONS['quality_levels'],
    limit=CONVERT_OPTIONS['limit_train']
)

# Validation set Conversion
val_stats = convert_visdrone_to_yolo_multiquality(
    visdrone_dir=config['VAL_DIR'],
    output_dir=YOLO_DATASET_DIR,
    split_name='val',
    quality_levels=CONVERT_OPTIONS['quality_levels'],
    limit=CONVERT_OPTIONS['limit_val']
)

# Test set Conversion
test_stats = convert_visdrone_to_yolo_multiquality(
    visdrone_dir=config['TEST_DIR'],
    output_dir=YOLO_DATASET_DIR,
    split_name='test',
    quality_levels=CONVERT_OPTIONS['quality_levels'],
    limit=CONVERT_OPTIONS['limit_test']
)

# Dataset YAML file Creation
standard_yaml, multiquality_yaml = create_dataset_yaml(YOLO_DATASET_DIR)

# Conversion results summary
print("\n" + "=" * 80)
print("Dataset conversion results Summary")
print("=" * 80)

total_images = train_stats['processed'] + val_stats['processed'] + test_stats['processed']
total_quality_variants = total_images * len(CONVERT_OPTIONS['quality_levels'])

print(f"\nRaw image:")
print(f"  Train: {train_stats['processed']}")
print(f"  Val: {val_stats['processed']}")
print(f"  Test: {test_stats['processed']}")
print(f"  Total: {total_images}")

print(f"\nImage quality variations generated:")
print(f"  Total image: {total_quality_variants} (Raw × {len(CONVERT_OPTIONS['quality_levels'])}개 화질)")

print(f"\nSave directory:")
print(f"  {YOLO_DATASET_DIR}")

# Disk usage check
def get_dir_size(path):
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / (1024**3)  # GB

dataset_size_gb = get_dir_size(YOLO_DATASET_DIR)
print(f"\nDisk usage: {dataset_size_gb:.2f} GB")

print(f"\nDataset conversion Complete")

# Preferences update
config['YOLO_DATASET_DIR'] = YOLO_DATASET_DIR
config['STANDARD_YAML'] = os.path.join(YOLO_DATASET_DIR, 'visdrone_standard.yaml')
config['MULTIQUALITY_YAML'] = os.path.join(YOLO_DATASET_DIR, 'visdrone_multiquality.yaml')