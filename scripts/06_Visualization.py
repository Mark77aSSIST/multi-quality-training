# 6. Visualization

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import cv2
from pathlib import Path

# Performance analysis per Class 
def analyze_per_class_performance(model_path, dataset_yaml, quality_level=100):
    print(f"\n{'='*80}")
    print(f"Performance analysis per Class (Quality: {quality_level}%)")
    print(f"{'='*80}")

    model = YOLO(model_path)

    metrics = model.val(data=dataset_yaml, split='test', verbose=False)

    class_names = ['pedestrian', 'people', 'bicycle', 'car', 'van',
                  'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']

    class_aps = metrics.box.ap_class_index

    # Dataframe Creation
    per_class_data = []
    for idx, class_name in enumerate(class_names):
        ap75_val = 0
        if idx < len(metrics.box.all_ap) and len(metrics.box.all_ap[idx]) > 5:
          ap75_val = float(metrics.box.all_ap[idx][5])

        per_class_data.append({
            'Class': class_name,
            'AP50': float(metrics.box.ap50[idx]) if idx < len(metrics.box.ap50) else 0,
            #'AP75': float(metrics.box.ap75[idx]) if idx < len(metrics.box.ap75) else 0,
            'AP75': ap75_val,
            'AP': float(metrics.box.ap[idx]) if idx < len(metrics.box.ap) else 0,
        })

    df = pd.DataFrame(per_class_data)
    df = df.sort_values('AP', ascending=False)

    print("\nPerformance per Class (Sort by mAP criteria):")
    print(df.to_string(index=False))

    return df

def plot_per_class_comparison(
    df_standard,
    df_multiquality,
    save_path=None
):

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(df_standard))
    width = 0.35

    bars1 = ax.bar(x - width/2, df_standard['AP'], width,
                   label='Standard', alpha=0.8, color='skyblue')
    bars2 = ax.bar(x + width/2, df_multiquality['AP'], width,
                   label='Multi-Quality', alpha=0.8, color='coral')

    ax.set_xlabel('Object Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Precision (AP)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_standard['Class'], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Show values
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Save: {save_path}")

    plt.show()

# Analysis of image quality degradation curves
def analyze_degradation_curve(all_results, save_dir):
    print(f"\n{'='*80}")
    print("Analysis of image quality degradation curves")
    print(f"{'='*80}")

    quality_levels = [20, 40, 60, 80, 100]

    # Calculate the degradation rate for each model
    for model_name, results in all_results.items():
        print(f"\n{model_name}:")

        baseline_map = results['100']['mAP50-95']

        print(f"  Baseline (100%): {baseline_map:.4f}")

        for q in [20, 40, 60, 80]:
            current_map = results[str(q)]['mAP50-95']
            degradation = baseline_map - current_map
            degradation_pct = (degradation / baseline_map) * 100

            print(f"  Quality {q}%:")
            print(f"    mAP: {current_map:.4f}")
            print(f"    Degradation: {degradation:.4f} ({degradation_pct:.1f}%)")

    # Performance degradation rate Graph
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, results in all_results.items():
        baseline = results['100']['mAP50-95']
        retention_rates = []

        for q in quality_levels:
            current = results[str(q)]['mAP50-95']
            retention = (current / baseline) * 100 if baseline > 0 else 0
            retention_rates.append(retention)

        ax.plot(quality_levels, retention_rates, marker='o',
               linewidth=2, markersize=8, label=model_name)

    ax.set_xlabel('Quality Level (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance Retention (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Retention vs Quality Degradation',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(quality_levels)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim([0, 105])

    plt.tight_layout()

    save_path = os.path.join(save_dir, 'degradation_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSave: {save_path}")

    plt.show()

# Feature prediction of Trained model
def visualize_predictions(
    model_path,
    image_dir,
    quality_levels=[20, 60, 100],
    num_samples=3,
    save_dir=None
):
    print(f"\n{'='*80}")
    print("Visualization for feature prediction samples of Trained model")
    print(f"{'='*80}")

    model = YOLO(model_path)

    # Select sample images
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    samples = np.random.choice(image_files, min(num_samples, len(image_files)),
                              replace=False)

    for img_file in samples:
        img_path = os.path.join(image_dir, img_file)
        img_original = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, len(quality_levels), figsize=(6*len(quality_levels), 6))

        if len(quality_levels) == 1:
            axes = [axes]

        for idx, quality in enumerate(quality_levels):
            # Quality degradation implement
            degraded_img = QualityDegrader.degrade_image(img_rgb, quality/100)

            # Prediction
            results = model.predict(degraded_img, verbose=False, conf=0.25)

            # Visualization
            annotated = results[0].plot()

            axes[idx].imshow(annotated)
            axes[idx].set_title(f'Quality: {quality}%\n'
                              f'Detections: {len(results[0].boxes)}',
                              fontsize=12, fontweight='bold')
            axes[idx].axis('off')

        plt.tight_layout()

        if save_dir:
            save_path = os.path.join(save_dir, f'predictions_{img_file}')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Save: {save_path}")

        plt.show()

# Overall Figure Creation
def create_publication_figure(all_results, save_path):
    print(f"\n{'='*80}")
    print("Overall Figure Creation")
    print(f"{'='*80}")

    # 2x2 subplot
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    quality_levels = [20, 40, 60, 80, 100]
    model_names = list(all_results.keys())

    # (a) mAP50-95 Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    for model_name in model_names:
        values = [all_results[model_name][str(q)]['mAP50-95'] for q in quality_levels]
        ax1.plot(quality_levels, values, marker='o', linewidth=2.5,
                markersize=10, label=model_name, alpha=0.8)
    ax1.set_xlabel('Image Quality (%)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('mAP50-95', fontsize=13, fontweight='bold')
    ax1.set_title('(a) Overall Performance', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(quality_levels)

    # (b) mAP50 Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    for model_name in model_names:
        values = [all_results[model_name][str(q)]['mAP50'] for q in quality_levels]
        ax2.plot(quality_levels, values, marker='s', linewidth=2.5,
                markersize=10, label=model_name, alpha=0.8)
    ax2.set_xlabel('Image Quality (%)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('mAP50', fontsize=13, fontweight='bold')
    ax2.set_title('(b) Detection at IoU=0.5', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(quality_levels)

    # (c) Absolute improvement
    ax3 = fig.add_subplot(gs[1, 0])
    if len(model_names) >= 2:
        baseline = model_names[0]
        proposed = model_names[1]

        improvements = []
        for q in quality_levels:
            imp = all_results[proposed][str(q)]['mAP50-95'] - all_results[baseline][str(q)]['mAP50-95']
            improvements.append(imp)

        colors = ['red' if x < 0 else 'green' for x in improvements]
        bars = ax3.bar(quality_levels, improvements, color=colors, alpha=0.7, width=8)

        # Show values (labels)
        for bar, val in zip(bars, improvements):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:+.3f}', ha='center',
                    va='bottom' if val > 0 else 'top', fontsize=10)

    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax3.set_xlabel('Image Quality (%)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('mAP50-95 Improvement', fontsize=13, fontweight='bold')
    ax3.set_title('(c) Absolute Performance Gain', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_xticks(quality_levels)

    # (d) Performance retention
    ax4 = fig.add_subplot(gs[1, 1])
    for model_name in model_names:
        baseline_map = all_results[model_name]['100']['mAP50-95']
        retention = []
        for q in quality_levels:
            ret = (all_results[model_name][str(q)]['mAP50-95'] / baseline_map) * 100
            retention.append(ret)
        ax4.plot(quality_levels, retention, marker='^', linewidth=2.5,
                markersize=10, label=model_name, alpha=0.8)

    ax4.axhline(y=100, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax4.set_xlabel('Image Quality (%)', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Performance Retention (%)', fontsize=13, fontweight='bold')
    ax4.set_title('(d) Robustness to Quality Degradation', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11, loc='lower right')
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(quality_levels)
    ax4.set_ylim([0, 105])

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Overall Figure Save: {save_path}")

    plt.show()

# Analysis experiment
print("=" * 80)
print("Advanced Analysis and Visualization")
print("=" * 80)

ANALYSIS_DIR = os.path.join(config['EXPERIMENT_PATH'], 'analysis')
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Evaluation results Load
results_file = os.path.join(config['EXPERIMENT_PATH'], 'evaluation_results', 'all_results.json')
if os.path.exists(results_file):
    with open(results_file, 'r') as f:
        all_results = json.load(f)

    # 1. Analysis of image quality degradation curves
    analyze_degradation_curve(all_results, ANALYSIS_DIR)

    # 2. Overall Figure
    pub_fig_path = os.path.join(ANALYSIS_DIR, 'publication_figure.png')
    create_publication_figure(all_results, pub_fig_path)

    # 3. Per-class Analysis (100% quality)
    if os.path.exists(config.get('MODEL_STANDARD', '')):
        df_standard = analyze_per_class_performance(
            config['MODEL_STANDARD'],
            config['STANDARD_YAML'],
            quality_level=100
        )

        df_multiquality = analyze_per_class_performance(
            config['MODEL_MULTIQUALITY'],
            config['STANDARD_YAML'],
            quality_level=100
        )

        # Per-class Comparison
        perclass_fig_path = os.path.join(ANALYSIS_DIR, 'per_class_comparison.png')
        plot_per_class_comparison(df_standard, df_multiquality, perclass_fig_path)

    # 4. Prediction results Visualization
    test_images_dir = os.path.join(config['YOLO_DATASET_DIR'], 'images', 'test', 'q100')
    if os.path.exists(test_images_dir) and os.path.exists(config.get('MODEL_MULTIQUALITY', '')):
        visualize_predictions(
            model_path=config['MODEL_MULTIQUALITY'],
            image_dir=test_images_dir,
            quality_levels=[20, 60, 100],
            num_samples=2,
            save_dir=ANALYSIS_DIR
        )

else:
    print(f"Can't find the evaluation results: {results_file}")

print("\n" + "=" * 80)
print("Analysis & Visualization Complete")
print("=" * 80)
print(f"\nAnalysis results Save: {ANALYSIS_DIR}")