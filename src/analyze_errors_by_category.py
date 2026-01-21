#!/usr/bin/env python3
"""Analyze model errors by category to identify areas for finetuning."""

import argparse
import json
from collections import defaultdict
from typing import Dict, List


def load_predictions(predictions_path: str) -> Dict[str, str]:
    """Load predictions from JSONL file."""
    predictions = {}
    with open(predictions_path, 'r') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                predictions[record['id']] = record.get('answer_prediction', '')
    return predictions


def load_ground_truth(gt_path: str) -> List[Dict]:
    """Load ground truth from JSON file."""
    with open(gt_path, 'r') as f:
        return json.load(f)


def analyze_by_category(predictions: Dict[str, str], ground_truth: List[Dict]):
    """Analyze accuracy by different categories."""

    # Initialize category statistics
    stats = {
        'modality': defaultdict(lambda: {'correct': 0, 'total': 0, 'incorrect_samples': []}),
        'category': defaultdict(lambda: {'correct': 0, 'total': 0, 'incorrect_samples': []}),
        'sub-category': defaultdict(lambda: {'correct': 0, 'total': 0, 'incorrect_samples': []}),
        'language': defaultdict(lambda: {'correct': 0, 'total': 0, 'incorrect_samples': []}),
        'source': defaultdict(lambda: {'correct': 0, 'total': 0, 'incorrect_samples': []}),
    }

    # Analyze each sample
    for sample in ground_truth:
        sample_id = sample['id']
        gt_answer = sample.get('answer', '')
        pred_answer = predictions.get(sample_id, '')

        is_correct = (pred_answer == gt_answer)

        # Track by each dimension
        for dimension in ['modality', 'category', 'sub-category', 'language', 'source']:
            if dimension in sample and sample[dimension] is not None:
                key = sample[dimension]
                stats[dimension][key]['total'] += 1
                if is_correct:
                    stats[dimension][key]['correct'] += 1
                else:
                    # Store incorrect sample info
                    stats[dimension][key]['incorrect_samples'].append({
                        'id': sample_id,
                        'question': sample.get('question', ''),
                        'choices': sample.get('choices', []),
                        'gt_answer': gt_answer,
                        'pred_answer': pred_answer,
                        'audio_path': sample.get('audio_path', ''),
                    })

    return stats


def print_statistics(stats: Dict, top_n: int = 10):
    """Print statistics for each dimension."""

    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE ANALYSIS BY CATEGORY")
    print("=" * 80)

    for dimension, categories in stats.items():
        print(f"\n{'='*80}")
        print(f"Performance by {dimension.upper()}")
        print(f"{'='*80}")

        # Calculate accuracy for each category
        category_accuracies = []
        for category, data in categories.items():
            accuracy = (data['correct'] / data['total'] * 100) if data['total'] > 0 else 0
            category_accuracies.append((category, accuracy, data['correct'], data['total']))

        # Sort by accuracy (ascending) to see worst performers first
        category_accuracies.sort(key=lambda x: x[1])

        print(f"\n{'Category':<40} {'Accuracy':<12} {'Correct/Total':<15}")
        print("-" * 80)

        for category, accuracy, correct, total in category_accuracies:
            category_str = str(category) if category is not None else "(Unknown)"
            print(f"{category_str:<40} {accuracy:>6.2f}%     {correct:>4}/{total:<4}")

        # Highlight worst performers
        print(f"\n⚠️  WORST {top_n} PERFORMERS (areas for finetuning):")
        print("-" * 80)
        for i, (category, accuracy, correct, total) in enumerate(category_accuracies[:top_n], 1):
            category_str = str(category) if category is not None else "(Unknown)"
            print(f"{i}. {category_str:<38} {accuracy:>6.2f}%  ({correct}/{total})")


def export_incorrect_samples(stats: Dict, dimension: str, output_path: str, min_samples: int = 10):
    """Export incorrect samples for categories with poor performance."""

    # Collect all categories with enough incorrect samples
    export_data = []

    for category, data in stats[dimension].items():
        incorrect_count = len(data['incorrect_samples'])
        if incorrect_count >= min_samples:
            accuracy = (data['correct'] / data['total'] * 100) if data['total'] > 0 else 0

            for sample in data['incorrect_samples']:
                export_data.append({
                    dimension: category,
                    'accuracy': accuracy,
                    **sample
                })

    # Sort by accuracy (worst first)
    export_data.sort(key=lambda x: x['accuracy'])

    # Write to JSON file
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Exported {len(export_data)} incorrect samples to: {output_path}")

    # Print summary
    categories_exported = set(item[dimension] for item in export_data)
    print(f"   Categories included: {len(categories_exported)}")
    print(f"   Categories: {', '.join(sorted(categories_exported))}")


def export_incorrect_for_reinference(predictions: Dict[str, str], ground_truth: List[Dict], output_path: str):
    """Export all incorrect samples in MMAR format for re-inference."""

    incorrect_samples = []

    for sample in ground_truth:
        sample_id = sample['id']
        gt_answer = sample.get('answer', '')
        pred_answer = predictions.get(sample_id, '')

        # Only include incorrect predictions
        if pred_answer != gt_answer:
            incorrect_samples.append(sample)

    # Write to JSON file in MMAR format
    with open(output_path, 'w') as f:
        json.dump(incorrect_samples, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Exported {len(incorrect_samples)} incorrect samples for re-inference: {output_path}")
    print(f"   You can now run inference on these samples using:")
    print(f"   python src/inference/infer_single_model_baseline.py \\")
    print(f"     --dataset_meta_path {output_path} \\")
    print(f"     --dataset_audio_prefix ./src/data \\")
    print(f"     ... [other parameters]")

    return incorrect_samples


def create_finetuning_dataset(stats: Dict, output_path: str, worst_n: int = 5):
    """Create a finetuning dataset focusing on worst-performing categories."""

    finetuning_data = []

    # Collect incorrect samples from worst-performing categories in each dimension
    for dimension, categories in stats.items():
        # Sort by accuracy
        category_accuracies = []
        for category, data in categories.items():
            accuracy = (data['correct'] / data['total'] * 100) if data['total'] > 0 else 0
            if len(data['incorrect_samples']) > 0:
                category_accuracies.append((category, accuracy, data['incorrect_samples']))

        category_accuracies.sort(key=lambda x: x[1])

        # Take worst N categories
        for category, accuracy, incorrect_samples in category_accuracies[:worst_n]:
            for sample in incorrect_samples:
                finetuning_data.append({
                    'dimension': dimension,
                    'dimension_value': category,
                    'accuracy': accuracy,
                    'id': sample['id'],
                    'audio_path': sample['audio_path'],
                    'question': sample['question'],
                    'choices': sample['choices'],
                    'answer': sample['gt_answer'],
                })

    # Remove duplicates based on ID
    seen_ids = set()
    unique_data = []
    for item in finetuning_data:
        if item['id'] not in seen_ids:
            unique_data.append(item)
            seen_ids.add(item['id'])

    # Write to JSON file
    with open(output_path, 'w') as f:
        json.dump(unique_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Created finetuning dataset with {len(unique_data)} samples: {output_path}")

    # Print breakdown
    dimension_counts = defaultdict(int)
    for item in unique_data:
        dimension_counts[item['dimension']] += 1

    print(f"\n   Breakdown by dimension:")
    for dim, count in sorted(dimension_counts.items()):
        print(f"   - {dim}: {count} samples")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze model errors by category to identify finetuning opportunities"
    )
    parser.add_argument(
        'predictions',
        help='Path to predictions JSONL file (e.g., outputs/result.jsonl)'
    )
    parser.add_argument(
        'ground_truth',
        help='Path to ground truth JSON file (e.g., src/data/MMAR-meta.json)'
    )
    parser.add_argument(
        '--export-incorrect',
        metavar='PATH',
        help='Export all incorrect samples to JSON file'
    )
    parser.add_argument(
        '--create-finetune-dataset',
        metavar='PATH',
        help='Create finetuning dataset from worst-performing categories'
    )
    parser.add_argument(
        '--export-incorrect-mmar',
        metavar='PATH',
        help='Export all incorrect samples in MMAR format for re-inference'
    )
    parser.add_argument(
        '--dimension',
        default='category',
        choices=['modality', 'category', 'sub-category', 'language', 'source'],
        help='Dimension to analyze for incorrect samples export (default: category)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=10,
        help='Number of top worst performers to show (default: 10)'
    )
    parser.add_argument(
        '--worst-n',
        type=int,
        default=5,
        help='Number of worst categories per dimension for finetuning dataset (default: 5)'
    )

    args = parser.parse_args()

    # Load data
    print("Loading predictions...")
    predictions = load_predictions(args.predictions)

    print("Loading ground truth...")
    ground_truth = load_ground_truth(args.ground_truth)

    print(f"\nAnalyzing {len(predictions)} predictions against {len(ground_truth)} ground truth samples...")

    # Analyze
    stats = analyze_by_category(predictions, ground_truth)

    # Print statistics
    print_statistics(stats, args.top_n)

    # Export incorrect samples if requested
    if args.export_incorrect:
        export_incorrect_samples(stats, args.dimension, args.export_incorrect, min_samples=5)

    # Export incorrect samples in MMAR format for re-inference
    if args.export_incorrect_mmar:
        export_incorrect_for_reinference(predictions, ground_truth, args.export_incorrect_mmar)

    # Create finetuning dataset if requested
    if args.create_finetune_dataset:
        create_finetuning_dataset(stats, args.create_finetune_dataset, args.worst_n)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
