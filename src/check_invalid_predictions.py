#!/usr/bin/env python3
"""
Check how many model predictions are not in the valid options/choices.

This script compares model predictions against the valid choices from the
ground truth data to identify predictions that fall outside the expected options.

Usage:
    python check_invalid_predictions.py --ground_truth path/to/MMAR-meta.json \
        --baseline path/to/baseline/result.jsonl \
        --finetuned path/to/finetuned/prediction.jsonl
"""

import argparse
import json
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def load_ground_truth(gt_path: str) -> Dict[str, Dict]:
    """Load ground truth and index by ID."""
    with open(gt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {item['id']: item for item in data}


def load_predictions(pred_path: str) -> Dict[str, str]:
    """Load predictions from JSONL file."""
    predictions = {}
    with open(pred_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                predictions[record['id']] = record.get('answer_prediction', '')
    return predictions


def clean_prediction(raw_pred: str) -> str:
    """
    Clean and normalize a prediction string.

    Handles various formats:
    - "ANSWER: X" -> "X"
    - Multiple repeated answers -> take first
    - Extra whitespace and newlines
    """
    if not raw_pred:
        return ""

    # Take only the first line if there are multiple
    first_line = raw_pred.strip().split('\n')[0].strip()

    # Remove common prefixes
    prefixes_to_remove = [
        r'^ANSWER[S]?:\s*',
        r'^The answer is[:\s]*',
        r'^Answer[:\s]*',
    ]

    cleaned = first_line
    for pattern in prefixes_to_remove:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

    return cleaned.strip()


def normalize_for_comparison(text: str) -> str:
    """Normalize text for comparison (lowercase, strip whitespace)."""
    return text.lower().strip()


def check_prediction_in_choices(
    prediction: str,
    choices: List[str],
    ground_truth_answer: str
) -> Tuple[bool, Optional[str]]:
    """
    Check if prediction matches any of the valid choices.

    Returns:
        Tuple of (is_valid, matched_choice or None)
    """
    if not prediction:
        return False, None

    pred_norm = normalize_for_comparison(prediction)

    # Check exact match with choices
    for choice in choices:
        choice_norm = normalize_for_comparison(choice)
        if pred_norm == choice_norm:
            return True, choice

    # Check if prediction contains choice or vice versa
    for choice in choices:
        choice_norm = normalize_for_comparison(choice)
        # Exact substring match for shorter strings
        if len(pred_norm) >= 2 and len(choice_norm) >= 2:
            if pred_norm in choice_norm or choice_norm in pred_norm:
                return True, choice

    # Check if prediction matches ground truth answer (might be formatted differently)
    gt_norm = normalize_for_comparison(ground_truth_answer)
    if pred_norm == gt_norm:
        return True, ground_truth_answer

    return False, None


def analyze_predictions(
    ground_truth: Dict[str, Dict],
    predictions: Dict[str, str],
    model_name: str
) -> Dict:
    """
    Analyze predictions and return statistics.
    """
    results = {
        'model_name': model_name,
        'total_samples': 0,
        'valid_predictions': 0,
        'invalid_predictions': 0,
        'missing_predictions': 0,
        'correct_predictions': 0,
        'invalid_samples': [],
        'by_category': defaultdict(lambda: {'total': 0, 'valid': 0, 'invalid': 0, 'correct': 0})
    }

    for sample_id, gt_data in ground_truth.items():
        results['total_samples'] += 1

        choices = gt_data.get('choices', [])
        gt_answer = gt_data.get('answer', '')
        category = gt_data.get('category', 'Unknown')

        # Skip samples without choices (open-ended questions)
        if not choices:
            continue

        results['by_category'][category]['total'] += 1

        if sample_id not in predictions:
            results['missing_predictions'] += 1
            continue

        raw_pred = predictions[sample_id]
        cleaned_pred = clean_prediction(raw_pred)

        is_valid, matched_choice = check_prediction_in_choices(
            cleaned_pred, choices, gt_answer
        )

        if is_valid:
            results['valid_predictions'] += 1
            results['by_category'][category]['valid'] += 1

            # Check if correct
            if matched_choice and normalize_for_comparison(matched_choice) == normalize_for_comparison(gt_answer):
                results['correct_predictions'] += 1
                results['by_category'][category]['correct'] += 1
        else:
            results['invalid_predictions'] += 1
            results['by_category'][category]['invalid'] += 1
            results['invalid_samples'].append({
                'id': sample_id,
                'question': gt_data.get('question', ''),
                'choices': choices,
                'ground_truth': gt_answer,
                'raw_prediction': raw_pred[:200] + '...' if len(raw_pred) > 200 else raw_pred,
                'cleaned_prediction': cleaned_pred,
                'category': category
            })

    return results


def print_results(results: Dict, verbose: bool = False):
    """Print analysis results."""
    print(f"\n{'='*70}")
    print(f"MODEL: {results['model_name']}")
    print(f"{'='*70}")

    total = results['total_samples']
    valid = results['valid_predictions']
    invalid = results['invalid_predictions']
    missing = results['missing_predictions']
    correct = results['correct_predictions']

    print(f"\nOVERALL STATISTICS:")
    print(f"  Total samples:        {total}")
    print(f"  Valid predictions:    {valid} ({100*valid/total:.1f}%)")
    print(f"  Invalid predictions:  {invalid} ({100*invalid/total:.1f}%)")
    print(f"  Missing predictions:  {missing} ({100*missing/total:.1f}%)")
    print(f"  Correct predictions:  {correct} ({100*correct/total:.1f}%)")

    print(f"\nBY CATEGORY:")
    print(f"  {'Category':<30} {'Total':>8} {'Correct':>8} {'Accuracy':>10} {'Invalid':>8} {'Invalid%':>10}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*10}")

    for category, stats in sorted(results['by_category'].items()):
        cat_total = stats['total']
        cat_valid = stats['valid']
        cat_invalid = stats['invalid']
        cat_correct = stats['correct']
        accuracy_pct = 100 * cat_correct / cat_total if cat_total > 0 else 0
        invalid_pct = 100 * cat_invalid / cat_total if cat_total > 0 else 0
        print(f"  {category:<30} {cat_total:>8} {cat_correct:>8} {accuracy_pct:>9.1f}% {cat_invalid:>8} {invalid_pct:>9.1f}%")

    if verbose and results['invalid_samples']:
        print(f"\nSAMPLE INVALID PREDICTIONS (first 10):")
        print("-" * 70)
        for sample in results['invalid_samples'][:10]:
            print(f"\n  ID: {sample['id']}")
            print(f"  Question: {sample['question'][:80]}...")
            print(f"  Choices: {sample['choices']}")
            print(f"  Ground truth: {sample['ground_truth']}")
            print(f"  Cleaned pred: {sample['cleaned_prediction'][:100]}")


def main():
    parser = argparse.ArgumentParser(
        description="Check how many predictions are not in valid options"
    )
    parser.add_argument(
        '--ground_truth', '-g',
        type=str,
        default='./src/data/MMAR-meta.json',
        help='Path to ground truth JSON file with choices'
    )
    parser.add_argument(
        '--baseline', '-b',
        type=str,
        default=None,
        help='Path to baseline predictions JSONL file'
    )
    parser.add_argument(
        '--finetuned', '-f',
        type=str,
        default=None,
        help='Path to finetuned predictions JSONL file'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show sample invalid predictions'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path to save detailed results as JSON'
    )

    args = parser.parse_args()

    # Load ground truth
    print(f"Loading ground truth from: {args.ground_truth}")
    ground_truth = load_ground_truth(args.ground_truth)
    print(f"Loaded {len(ground_truth)} samples")

    all_results = []

    # Analyze baseline if provided
    if args.baseline:
        print(f"\nLoading baseline predictions from: {args.baseline}")
        baseline_preds = load_predictions(args.baseline)
        print(f"Loaded {len(baseline_preds)} predictions")

        baseline_results = analyze_predictions(
            ground_truth, baseline_preds, f"Baseline ({args.baseline})"
        )
        print_results(baseline_results, args.verbose)
        all_results.append(baseline_results)

    # Analyze finetuned if provided
    if args.finetuned:
        print(f"\nLoading finetuned predictions from: {args.finetuned}")
        finetuned_preds = load_predictions(args.finetuned)
        print(f"Loaded {len(finetuned_preds)} predictions")

        finetuned_results = analyze_predictions(
            ground_truth, finetuned_preds, f"Finetuned ({args.finetuned})"
        )
        print_results(finetuned_results, args.verbose)
        all_results.append(finetuned_results)

    # Comparison summary
    if args.baseline and args.finetuned:
        print(f"\n{'='*70}")
        print("COMPARISON SUMMARY")
        print(f"{'='*70}")
        print(f"{'Model':<40} {'Invalid':>10} {'Invalid%':>10}")
        print(f"{'-'*40} {'-'*10} {'-'*10}")
        for res in all_results:
            name = res['model_name'].split('/')[-1][:38]
            invalid = res['invalid_predictions']
            total = res['total_samples']
            pct = 100 * invalid / total if total > 0 else 0
            print(f"{name:<40} {invalid:>10} {pct:>9.1f}%")

    # Save detailed results if requested
    if args.output:
        output_data = []
        for res in all_results:
            # Convert defaultdict to regular dict for JSON serialization
            res_copy = dict(res)
            res_copy['by_category'] = dict(res_copy['by_category'])
            output_data.append(res_copy)

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {args.output}")

    print("\nDone!")


if __name__ == '__main__':
    main()