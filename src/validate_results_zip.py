#!/usr/bin/env python3
"""Validate result.jsonl format, evaluate accuracy, and create submission ZIP."""

import argparse
import json
import os
import zipfile
from collections import defaultdict


def evaluate_results(predictions: list[dict], ground_truth_path: str):
    """
    Evaluate predictions against ground truth dataset.

    Args:
        predictions: List of prediction records with 'id' and 'answer_prediction'
        ground_truth_path: Path to MMAR metadata JSON with ground truth answers
    """
    # Load ground truth
    with open(ground_truth_path, "r") as f:
        ground_truth = json.load(f)

    # Build lookup by ID
    gt_by_id = {sample["id"]: sample for sample in ground_truth}
    pred_by_id = {p["id"]: p.get("answer_prediction", "") for p in predictions}

    # Evaluate
    correct = 0
    incorrect = 0
    missing_gt = 0
    missing_pred = 0
    empty_pred = 0

    incorrect_samples = []

    for pred in predictions:
        pred_id = pred["id"]
        pred_answer = pred.get("answer_prediction", "")

        if pred_id not in gt_by_id:
            missing_gt += 1
            continue

        gt_sample = gt_by_id[pred_id]
        gt_answer = gt_sample.get("answer", "")

        if not pred_answer:
            empty_pred += 1
            incorrect_samples.append((pred_id, gt_answer, pred_answer, "empty"))
        elif pred_answer == gt_answer:
            correct += 1
        else:
            incorrect += 1
            incorrect_samples.append((pred_id, gt_answer, pred_answer, "wrong"))

    # Check for samples in GT but not in predictions
    for gt_id in gt_by_id:
        if gt_id not in pred_by_id:
            missing_pred += 1

    total_evaluated = correct + incorrect + empty_pred
    accuracy = (correct / total_evaluated * 100) if total_evaluated > 0 else 0

    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total predictions:     {len(predictions)}")
    print(f"Total ground truth:    {len(ground_truth)}")
    print(f"Evaluated:             {total_evaluated}")
    print()
    print(f"Correct:               {correct}")
    print(f"Incorrect:             {incorrect}")
    print(f"Empty predictions:     {empty_pred}")
    print(f"Missing from GT:       {missing_gt}")
    print(f"Missing predictions:   {missing_pred}")
    print()
    print(f"ACCURACY:              {accuracy:.2f}%")
    print("=" * 50)

    # Show some incorrect examples
    if incorrect_samples and len(incorrect_samples) <= 20:
        print("\nIncorrect predictions:")
        for pid, gt, pred, reason in incorrect_samples[:20]:
            print(f"  ID: {pid} | GT: {gt} | Pred: {pred} ({reason})")

    return accuracy


def validate_results(filepath: str) -> tuple[bool, list[dict]]:
    """
    Validate result.jsonl submission format.

    Required format per line:
    {
        "id": str,
        "thinking_prediction": str (< 10,000 chars),
        "answer_prediction": str (A, B, C, or D)
    }
    """
    MAX_THINKING_CHARS = 10_000

    errors = []
    records = []
    total = 0

    with open(filepath, "r") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            total += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON - {e}")
                continue

            records.append(record)

            # Check required fields
            if "id" not in record:
                errors.append(f"Line {line_num}: Missing 'id' field")

            if "thinking_prediction" not in record:
                errors.append(f"Line {line_num}, ID {record.get('id', '?')}: Missing 'thinking_prediction' field")
            else:
                thinking = record["thinking_prediction"]
                if not isinstance(thinking, str):
                    errors.append(f"Line {line_num}, ID {record.get('id', '?')}: 'thinking_prediction' must be string")
                elif len(thinking) >= MAX_THINKING_CHARS:
                    errors.append(
                        f"Line {line_num}, ID {record.get('id', '?')}: "
                        f"'thinking_prediction' is {len(thinking)} chars (max: {MAX_THINKING_CHARS - 1})"
                    )

            if "answer_prediction" not in record:
                errors.append(f"Line {line_num}, ID {record.get('id', '?')}: Missing 'answer_prediction' field")
            else:
                answer = record["answer_prediction"]
                if not isinstance(answer, str):
                    errors.append(f"Line {line_num}, ID {record.get('id', '?')}: 'answer_prediction' must be string")
                elif not answer:
                    errors.append(f"Line {line_num}, ID {record.get('id', '?')}: 'answer_prediction' is empty")

    print(f"Validated {total} records")
    print()

    if errors:
        print(f"FAILED: {len(errors)} error(s) found:\n")
        for err in errors[:20]:  # Show first 20 errors
            print(f"  {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors")
        return False, records
    else:
        print("PASSED: All records are valid")
        return True, records


def create_submission_zip(jsonl_path: str, output_zip: str):
    """Create submission ZIP with result.jsonl at top level."""
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(jsonl_path, "result.jsonl")

    size_kb = os.path.getsize(output_zip) / 1024
    print(f"Created submission: {output_zip} ({size_kb:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Validate result.jsonl, evaluate accuracy, and create submission ZIP")
    parser.add_argument("filepath", nargs="?", default="result.jsonl", help="Path to JSONL file")
    parser.add_argument("--eval", "-e", metavar="GROUND_TRUTH", help="Evaluate against ground truth JSON file")
    parser.add_argument("--zip", "-z", metavar="OUTPUT", help="Create submission ZIP file")
    args = parser.parse_args()

    valid, records = validate_results(args.filepath)

    if args.eval:
        print()
        evaluate_results(records, args.eval)

    if valid and args.zip:
        print()
        create_submission_zip(args.filepath, args.zip)
    elif args.zip and not valid:
        print("\nSkipping ZIP creation due to validation errors")

    exit(0 if valid else 1)


if __name__ == "__main__":
    main()
