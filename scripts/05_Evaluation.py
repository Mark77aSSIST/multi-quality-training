# 5. Evaluation

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from pathlib import Path

# Multi-Quality Evaluation function
def evaluate_on_quality_levels(
    model_path,
    dataset_base_dir,
    quality_levels=[20, 40, 60, 80, 100],
    split='test',
    save_dir=None
):
    print(f"\n{'='*80}")
    print(f"Multi-Quality Evaluation")
    print(f"  Model: {os.path.basename(model_path)}")
    print(f"  Split: {split}")
    print(f"  Quality level: {quality_levels}")
    print(f"{'='*80}")

    # Model Load
    model = YOLO(model_path)

    results_dict = {}

    for quality in quality_levels:
        print(f"\nQuality {quality}% evaluating...")

        q_tag = f'q{quality}'

        # Create temporary dataset YAML
        temp_yaml_path = os.path.join(save_dir or '/tmp', f'temp_{split}_{q_tag}.yaml')
        temp_yaml = {
            'path': dataset_base_dir,
            'train': f'images/train/{q_tag}',
            'val': f'images/{split}/{q_tag}',
            'test': f'images/{split}/{q_tag}',
            'nc': 10,
            'names': ['pedestrian', 'people', 'bicycle', 'car', 'van',
                     'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
        }

        with open(temp_yaml_path, 'w') as f:
            import yaml
            yaml.dump(temp_yaml, f)

        # Run an evaluation
        metrics = model.val(
            data=temp_yaml_path,
            split=split,
            batch=16,
            imgsz=640,
            verbose=False
        )

        # Results save
        results_dict[quality] = {
            'mAP50': float(metrics.box.map50),
            'mAP50-95': float(metrics.box.map),
            'mAP75': float(metrics.box.map75),
            'precision': float(metrics.box.p.mean()),
            'recall': float(metrics.box.r.mean()),
        }

        print(f"  mAP50: {results_dict[quality]['mAP50']:.4f}")
        print(f"  mAP50-95: {results_dict[quality]['mAP50-95']:.4f}")

    # temporary file cleanup
    if os.path.exists(temp_yaml_path):
        os.remove(temp_yaml_path)

    return results_dict

# Simple visualization
def plot_quality_performance(
    results_dict,
    model_names,
    save_path=None
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics_to_plot = ['mAP50-95', 'mAP50', 'mAP75']
    metric_titles = ['mAP50-95', 'mAP50', 'mAP75']

    quality_levels = sorted(list(results_dict[model_names[0]].keys()))

    for idx, (metric, title) in enumerate(zip(metrics_to_plot, metric_titles)):
        ax = axes[idx]

        for model_name in model_names:
            values = [results_dict[model_name][q][metric] for q in quality_levels]
            ax.plot(quality_levels, values, marker='o', linewidth=2,
                   label=model_name, markersize=8)

        ax.set_xlabel('Quality Level (%)', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'{title} vs Quality Level', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(quality_levels)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure save: {save_path}")

    plt.show()

def create_comparison_table(results_dict, model_names):
    data = []

    for model_name in model_names:
        for quality, metrics in results_dict[model_name].items():
            data.append({
                'Model': model_name,
                'Quality (%)': quality,
                'mAP50-95': metrics['mAP50-95'],
                'mAP50': metrics['mAP50'],
                'mAP75': metrics['mAP75'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall']
            })

    df = pd.DataFrame(data)

    # Pivot table for better visualization
    pivot_map = df.pivot_table(
        values='mAP50-95',
        index='Model',
        columns='Quality (%)',
        aggfunc='first'
    )

    print("\n" + "="*80)
    print("Performance Comparison: mAP50-95")
    print("="*80)
    print(pivot_map.to_string())
    print()

    # Calculation of improvement
    if len(model_names) == 2:
        baseline = model_names[0]
        proposed = model_names[1]

        print("\n" + "="*80)
        print(f"Improvement: {proposed} vs {baseline}")
        print("="*80)

        improvements = []
        for quality in sorted(results_dict[baseline].keys()):
            baseline_map = results_dict[baseline][quality]['mAP50-95']
            proposed_map = results_dict[proposed][quality]['mAP50-95']
            improvement = proposed_map - baseline_map
            improvement_pct = (improvement / baseline_map) * 100 if baseline_map > 0 else 0

            improvements.append({
                'Quality (%)': quality,
                f'{baseline} mAP': f"{baseline_map:.4f}",
                f'{proposed} mAP': f"{proposed_map:.4f}",
                'Absolute Gain': f"{improvement:+.4f}",
                'Relative Gain (%)': f"{improvement_pct:+.2f}%"
            })

        df_improvement = pd.DataFrame(improvements)
        print(df_improvement.to_string(index=False))
        print()

    return df

# Run an evaluation
print("=" * 80)
print("Multi-Quality Evaluation Start")
print("=" * 80)

# Results save directory
EVAL_DIR = os.path.join(config['EXPERIMENT_PATH'], 'evaluation_results')
os.makedirs(EVAL_DIR, exist_ok=True)

# Evaluation model
models_to_evaluate = {
    'Standard (Q100)': config['MODEL_STANDARD'],
    'Multi-Quality': config['MODEL_MULTIQUALITY']
}

all_results = {}

for model_name, model_path in models_to_evaluate.items():
    print(f"\n{'='*80}")
    print(f"{model_name} evaluation")
    print(f"{'='*80}")

    if not os.path.exists(model_path):
        print(f"Can't find model file: {model_path}")
        continue

    results = evaluate_on_quality_levels(
        model_path=model_path,
        dataset_base_dir=config['YOLO_DATASET_DIR'],
        quality_levels=[20, 40, 60, 80, 100],
        split='test',
        save_dir=EVAL_DIR
    )

    all_results[model_name] = results

    # Each results Save
    result_file = os.path.join(EVAL_DIR, f'{model_name.replace(" ", "_")}_results.json')
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results Save: {result_file}")

# Simple visualization & analysis
if len(all_results) >= 2:
    print("\n" + "=" * 80)
    print("Simple visualization & analysis")
    print("=" * 80)

    # Performance comparison graph
    plot_save_path = os.path.join(EVAL_DIR, 'performance_comparison.png')
    plot_quality_performance(
        results_dict=all_results,
        model_names=list(all_results.keys()),
        save_path=plot_save_path
    )

    # Table
    df_comparison = create_comparison_table(
        results_dict=all_results,
        model_names=list(all_results.keys())
    )

    # Save to CSV
    csv_path = os.path.join(EVAL_DIR, 'comparison_table.csv')
    df_comparison.to_csv(csv_path, index=False)
    print(f"CSV save: {csv_path}")

    # All results Save to JSON
    all_results_path = os.path.join(EVAL_DIR, 'all_results.json')
    with open(all_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"All results save: {all_results_path}")

print("\n" + "=" * 80)
print("Evaluation Complete")
print("=" * 80)
print(f"\nEvaluation results save directory: {EVAL_DIR}")