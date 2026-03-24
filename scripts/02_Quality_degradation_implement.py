# 2. Dataset analysis & Multi-Quality Degradation implement

import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


# VisDrone annotation parser
class VisDroneParser:
    CLASS_NAMES = {
        0: 'ignored',
        1: 'pedestrian',
        2: 'people',
        3: 'bicycle',
        4: 'car',
        5: 'van',
        6: 'truck',
        7: 'tricycle',
        8: 'awning-tricycle',
        9: 'bus',
        10: 'motor',
        11: 'others'
    }

    # Exclude the ignored and others to train YOLO (0-based indexing)
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

                bbox_left = int(parts[0])
                bbox_top = int(parts[1])
                bbox_width = int(parts[2])
                bbox_height = int(parts[3])
                score = int(parts[4])
                object_category = int(parts[5])

                if object_category == 0 or bbox_width <= 0 or bbox_height <= 0:
                    continue

                if object_category in VisDroneParser.VALID_CLASSES:
                    annotations.append({
                        'bbox': [bbox_left, bbox_top, bbox_width, bbox_height],
                        'class': object_category,
                        'score': score
                    })

        return annotations

    @staticmethod
    def visdrone_to_yolo(annotations, img_width, img_height):
        yolo_annotations = []

        for ann in annotations:
            x, y, w, h = ann['bbox']
            class_id = ann['class']

            # Transe to YOLO formet (0-based class index)
            # VisDrone class 1-10 → YOLO class 0-9
            yolo_class = VisDroneParser.VALID_CLASSES.index(class_id)

            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            norm_w = w / img_width
            norm_h = h / img_height

            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            norm_w = max(0, min(1, norm_w))
            norm_h = max(0, min(1, norm_h))

            yolo_annotations.append({
                'class': yolo_class,
                'bbox': [x_center, y_center, norm_w, norm_h]
            })

        return yolo_annotations

# Dataset statistics check
def analyze_dataset(dataset_dir):
    images_dir = os.path.join(dataset_dir, 'images')
    annotations_dir = os.path.join(dataset_dir, 'annotations')

    if not os.path.exists(images_dir):
        print(f"Can't find the image directory: {images_dir}")
        return None

    stats = {
        'num_images': 0,
        'num_annotations': 0,
        'class_distribution': {i: 0 for i in range(10)},
        'image_sizes': [],
        'bbox_sizes': []
    }

    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])

    print(f"Analyzing: {len(image_files)} image...")

    for img_file in tqdm(image_files[:100], desc="Sample analysis"):
        img_path = os.path.join(images_dir, img_file)
        ann_file = os.path.join(annotations_dir, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))

        # image size
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        stats['image_sizes'].append((w, h))
        stats['num_images'] += 1

        # Annotation
        if os.path.exists(ann_file):
            annotations = VisDroneParser.parse_annotation(ann_file)
            stats['num_annotations'] += len(annotations)

            for ann in annotations:
                yolo_class = VisDroneParser.VALID_CLASSES.index(ann['class'])
                stats['class_distribution'][yolo_class] += 1

                bbox_w, bbox_h = ann['bbox'][2], ann['bbox'][3]
                stats['bbox_sizes'].append((bbox_w, bbox_h))

    return stats

# Dataset analysis
print("=" * 80)
print("VisDrone dataset analysis")
print("=" * 80)

for split_name, split_dir in [('Train', config['TRAIN_DIR']),
                                ('Val', config['VAL_DIR']),
                                ('Test', config['TEST_DIR'])]:
    print(f"\n{split_name} Set:")

    images_dir = os.path.join(split_dir, 'images')
    if os.path.exists(images_dir):
        num_images = len([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])
        print(f"  Number of images: {num_images}")

        # Sample statistics
        stats = analyze_dataset(split_dir)
        if stats:
            print(f"  Total number of annotations: {stats['num_annotations']}")
            print(f"  Mean image size: {np.mean(stats['image_sizes'], axis=0).astype(int)}")

            # Class distribution
            print(f"\n  Class distribution (Top 5):")
            class_counts = sorted(stats['class_distribution'].items(),
                                key=lambda x: x[1], reverse=True)[:5]
            for cls_idx, count in class_counts:
                cls_id = VisDroneParser.VALID_CLASSES[cls_idx]
                cls_name = VisDroneParser.CLASS_NAMES[cls_id]
                print(f"    {cls_name}: {count}")
    else:
        print(f"  Can't find the directory.")

# Multi-Quality Degradation
class QualityDegrader:
    QUALITY_LEVELS = [0.2, 0.4, 0.6, 0.8, 1.0]

    @staticmethod
    def degrade_image(image, quality_level):
        if quality_level == 1.0:
            return image.copy()

        h, w = image.shape[:2]

        # Downsampling
        new_h = int(h * quality_level)
        new_w = int(w * quality_level)

        # Minimum image size
        new_h = max(1, new_h)
        new_w = max(1, new_w)

        downsampled = cv2.resize(image, (new_w, new_h),
                                interpolation=cv2.INTER_CUBIC)

        # Upsampling (Restore to original size)
        degraded = cv2.resize(downsampled, (w, h),
                             interpolation=cv2.INTER_CUBIC)

        return degraded

    @staticmethod
    def calculate_quality_metrics(original, degraded):
        # PSNR
        mse = np.mean((original.astype(float) - degraded.astype(float)) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))

        # Simple SSIM
        return {
            'psnr': psnr,
            'mse': mse
        }

# Visualization of the image quality degradation
def visualize_quality_degradation(image_path, save_path=None):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Can't load an image: {image_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create 5-levels Image Quality
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, quality in enumerate(QualityDegrader.QUALITY_LEVELS):
        degraded = QualityDegrader.degrade_image(img_rgb, quality)
        metrics = QualityDegrader.calculate_quality_metrics(img_rgb, degraded)

        axes[idx].imshow(degraded)
        axes[idx].set_title(f'Quality: {int(quality*100)}%\n'
                           f'PSNR: {metrics["psnr"]:.2f} dB',
                           fontsize=12)
        axes[idx].axis('off')

    axes[5].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()

print("\n" + "=" * 80)
print("Visualization of the image quality degradation")
print("=" * 80)

# Select a sample image in the Train set
train_images_dir = os.path.join(config['TRAIN_DIR'], 'images')
if os.path.exists(train_images_dir):
    sample_images = sorted([f for f in os.listdir(train_images_dir)
                           if f.endswith(('.jpg', '.png'))])[:3]

    for img_file in sample_images:
        img_path = os.path.join(train_images_dir, img_file)
        save_path = os.path.join(config['EXPERIMENT_PATH'], 'results',
                                f'quality_demo_{img_file}')

        print(f"\n{img_file}")
        visualize_quality_degradation(img_path, save_path)
        break

print("\nData analysis & Quality degradation implement Complete.")

