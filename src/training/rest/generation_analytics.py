"""
Generation Analytics Script

Analyzes the generations.jsonl file to show distribution of correct answers
without performing any filtering. Useful to understand model performance
before deciding on a filtering strategy.

Usage:
    python generation_analytics.py --generations_path ./outputs/rest/generations/generations.jsonl

    # Save analytics to file
    python generation_analytics.py --generations_path ./outputs/rest/generations/generations.jsonl \
        --output_path ./outputs/analytics.json
"""

import argparse
import json
import os
import re
import logging
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Learning Zone Categories
# =============================================================================

@dataclass
class CategoryBoundaries:
    """Boundaries for learning zone categories (for 16 samples)."""
    too_hard_max: int = 0           # 0 correct
    challenging_max: int = 3        # 1-3 correct
    learning_zone_max: int = 11     # 4-11 correct
    almost_mastered_max: int = 14   # 12-14 correct
    # 15-16 = mastered


def categorize_problem(correct_count: int, bounds: CategoryBoundaries) -> str:
    """Categorize a problem based on correct count."""
    if correct_count == 0:
        return "too_hard"
    elif correct_count <= bounds.challenging_max:
        return "challenging"
    elif correct_count <= bounds.learning_zone_max:
        return "learning_zone"
    elif correct_count <= bounds.almost_mastered_max:
        return "almost_mastered"
    else:
        return "mastered"


# =============================================================================
# Answer Extraction and Matching
# =============================================================================

def get_answer_from_candidate(candidate) -> Optional[str]:
    """Get answer from candidate using pre-parsed answer_prediction field.

    The generation process already extracts answers into answer_prediction,
    so we just use that directly.

    Format expected:
    {
        "raw_output": "<think>...</think>\nANSWER: Double bass",
        "thinking_prediction": "...",
        "answer_text": "ANSWER: Double bass",
        "answer_prediction": "Double bass"  <- We use this directly
    }
    """
    if isinstance(candidate, dict):
        # Use pre-parsed answer_prediction directly
        return candidate.get("answer_prediction", "") or None
    else:
        # Legacy string format - shouldn't happen with current generation
        return None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    normalized = answer.strip()
    normalized = re.sub(r"^\s*\([A-Da-d]\)\s*", "", normalized)
    normalized = re.sub(r"^\s*[A-Da-d]\.\s*", "", normalized)
    return normalized.lower().strip()


def check_answer_match(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth."""
    if not predicted:
        return False

    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)

    # Exact match
    if pred_norm == gt_norm:
        return True

    # Option letter match
    pred_letter = re.search(r"^([A-Da-d])$", pred_norm)
    gt_letter = re.search(r"^\(([A-Da-d])\)", ground_truth)

    if pred_letter and gt_letter:
        return pred_letter.group(1).upper() == gt_letter.group(1).upper()

    if gt_letter:
        gt_option = gt_letter.group(1).upper()
        if pred_norm.upper() == gt_option:
            return True

    # Content containment
    if len(gt_norm) > 2 and gt_norm in pred_norm:
        return True
    if len(pred_norm) > 2 and pred_norm in gt_norm:
        return True

    return False


# =============================================================================
# Analytics Functions
# =============================================================================

def analyze_generations(generations_path: str, bounds: CategoryBoundaries = None) -> Dict:
    """Analyze generations file and compute statistics.

    Returns:
        Dictionary with comprehensive analytics
    """
    if bounds is None:
        bounds = CategoryBoundaries()

    # Load generations
    logger.info(f"Loading generations from {generations_path}")
    generations = []
    with open(generations_path, "r", encoding="utf-8") as f:
        for line in f:
            generations.append(json.loads(line))

    logger.info(f"Loaded {len(generations)} problems")

    # Initialize analytics
    analytics = {
        "summary": {},
        "category_distribution": {
            "too_hard": {"count": 0, "percentage": 0, "problem_ids": []},
            "challenging": {"count": 0, "percentage": 0, "problem_ids": []},
            "learning_zone": {"count": 0, "percentage": 0, "problem_ids": []},
            "almost_mastered": {"count": 0, "percentage": 0, "problem_ids": []},
            "mastered": {"count": 0, "percentage": 0, "problem_ids": []},
        },
        "correct_count_histogram": {i: 0 for i in range(17)},
        "correct_rate_buckets": {
            "0%": {"count": 0, "problem_ids": []},
            "1-25%": {"count": 0, "problem_ids": []},
            "26-50%": {"count": 0, "problem_ids": []},
            "51-75%": {"count": 0, "problem_ids": []},
            "76-99%": {"count": 0, "problem_ids": []},
            "100%": {"count": 0, "problem_ids": []},
        },
        "per_problem_details": [],
    }

    total_problems = len(generations)
    total_candidates = 0
    total_correct = 0
    correct_rates = []

    # Process each problem
    for gen in generations:
        problem_id = gen.get("id", "unknown")
        candidates = gen.get("candidates", [])
        ground_truth = gen.get("ground_truth", "")
        total_count = len(candidates)
        total_candidates += total_count

        # Count correct answers using pre-parsed answer_prediction
        correct_count = 0
        for candidate in candidates:
            answer = get_answer_from_candidate(candidate)

            if answer and check_answer_match(answer, ground_truth):
                correct_count += 1

        total_correct += correct_count

        # Calculate correct rate
        correct_rate = (correct_count / total_count * 100) if total_count > 0 else 0
        correct_rates.append(correct_rate)

        # Categorize
        category = categorize_problem(correct_count, bounds)
        analytics["category_distribution"][category]["count"] += 1
        analytics["category_distribution"][category]["problem_ids"].append(problem_id)

        # Histogram
        if correct_count <= 16:
            analytics["correct_count_histogram"][correct_count] += 1

        # Rate buckets
        if correct_rate == 0:
            bucket = "0%"
        elif correct_rate <= 25:
            bucket = "1-25%"
        elif correct_rate <= 50:
            bucket = "26-50%"
        elif correct_rate <= 75:
            bucket = "51-75%"
        elif correct_rate < 100:
            bucket = "76-99%"
        else:
            bucket = "100%"
        analytics["correct_rate_buckets"][bucket]["count"] += 1
        analytics["correct_rate_buckets"][bucket]["problem_ids"].append(problem_id)

        # Per-problem details
        analytics["per_problem_details"].append({
            "id": problem_id,
            "total_candidates": total_count,
            "correct_count": correct_count,
            "correct_rate": round(correct_rate, 2),
            "category": category,
            "ground_truth": ground_truth,
        })

    # Compute summary
    analytics["summary"] = {
        "total_problems": total_problems,
        "total_candidates": total_candidates,
        "total_correct": total_correct,
        "overall_accuracy": round(total_correct / total_candidates * 100, 2) if total_candidates > 0 else 0,
        "mean_correct_rate": round(sum(correct_rates) / len(correct_rates), 2) if correct_rates else 0,
        "median_correct_rate": round(sorted(correct_rates)[len(correct_rates)//2], 2) if correct_rates else 0,
        "min_correct_rate": round(min(correct_rates), 2) if correct_rates else 0,
        "max_correct_rate": round(max(correct_rates), 2) if correct_rates else 0,
        "std_correct_rate": round(
            (sum((r - sum(correct_rates)/len(correct_rates))**2 for r in correct_rates) / len(correct_rates))**0.5, 2
        ) if correct_rates else 0,
    }

    # Compute percentages for categories
    for category in analytics["category_distribution"]:
        count = analytics["category_distribution"][category]["count"]
        analytics["category_distribution"][category]["percentage"] = round(
            count / total_problems * 100, 2
        ) if total_problems > 0 else 0

    return analytics


def print_analytics(analytics: Dict):
    """Print analytics in a nicely formatted way."""

    summary = analytics["summary"]

    print("")
    print("=" * 80)
    print("                         GENERATION ANALYTICS REPORT")
    print("=" * 80)

    # Summary
    print("")
    print("SUMMARY STATISTICS")
    print("-" * 80)
    print(f"  Total problems:           {summary['total_problems']:,}")
    print(f"  Total candidates:         {summary['total_candidates']:,}")
    print(f"  Candidates per problem:   {summary['total_candidates'] // summary['total_problems'] if summary['total_problems'] > 0 else 0}")
    print(f"  Total correct:            {summary['total_correct']:,}")
    print(f"  Overall accuracy:         {summary['overall_accuracy']}%")
    print("")
    print(f"  Per-Problem Correct Rate:")
    print(f"    Mean:                   {summary['mean_correct_rate']}%")
    print(f"    Median:                 {summary['median_correct_rate']}%")
    print(f"    Std Dev:                {summary['std_correct_rate']}%")
    print(f"    Min:                    {summary['min_correct_rate']}%")
    print(f"    Max:                    {summary['max_correct_rate']}%")

    # Category Distribution
    print("")
    print("LEARNING ZONE CATEGORIES")
    print("-" * 80)
    print(f"  {'Category':<20} {'Count':>8} {'Percentage':>12}    {'Action':<25}")
    print(f"  {'-'*20} {'-'*8} {'-'*12}    {'-'*25}")

    category_actions = {
        "too_hard": "SKIP (no signal)",
        "challenging": "Keep all, weight 1.0x",
        "learning_zone": "Keep top 3, weight 1.5x ***",
        "almost_mastered": "Keep best 1, weight 0.5x",
        "mastered": "Skip or 10% sample",
    }

    for category in ["too_hard", "challenging", "learning_zone", "almost_mastered", "mastered"]:
        dist = analytics["category_distribution"][category]
        action = category_actions[category]
        marker = "<<<" if category == "learning_zone" else ""
        print(f"  {category:<20} {dist['count']:>8} {dist['percentage']:>11}%    {action:<25} {marker}")

    # Visual distribution bar
    print("")
    print("CATEGORY DISTRIBUTION (VISUAL)")
    print("-" * 80)

    for category in ["too_hard", "challenging", "learning_zone", "almost_mastered", "mastered"]:
        dist = analytics["category_distribution"][category]
        bar_len = int(dist["percentage"] / 2)
        bar = "█" * bar_len
        highlight = " <<<< PRIORITIZE!" if category == "learning_zone" else ""
        print(f"  {category:<18} {bar:<50} {dist['percentage']:>5}%{highlight}")

    # Correct Rate Buckets
    print("")
    print("CORRECT RATE DISTRIBUTION")
    print("-" * 80)

    buckets = analytics["correct_rate_buckets"]
    total = summary["total_problems"]

    for bucket in ["0%", "1-25%", "26-50%", "51-75%", "76-99%", "100%"]:
        count = buckets[bucket]["count"]
        pct = round(count / total * 100, 1) if total > 0 else 0
        bar_len = int(pct / 2)
        bar = "█" * bar_len
        print(f"  {bucket:>8}: {count:>6} ({pct:>5}%) {bar}")

    # Correct Count Histogram
    print("")
    print("CORRECT COUNT HISTOGRAM (per problem, out of 16)")
    print("-" * 80)

    histogram = analytics["correct_count_histogram"]
    max_count = max(histogram.values()) if histogram else 1

    for i in range(17):
        count = histogram.get(i, 0)
        bar_len = int(count / max_count * 40) if max_count > 0 else 0
        bar = "█" * bar_len

        # Add category label
        if i == 0:
            label = " <- Too Hard"
        elif i <= 3:
            label = " <- Challenging" if i == 2 else ""
        elif i <= 11:
            label = " <- LEARNING ZONE" if i == 7 else ""
        elif i <= 14:
            label = " <- Almost Mastered" if i == 13 else ""
        else:
            label = " <- Mastered" if i == 16 else ""

        print(f"  {i:>2} correct: {count:>6} {bar}{label}")

    # Recommendations
    print("")
    print("RECOMMENDATIONS")
    print("-" * 80)

    too_hard_pct = analytics["category_distribution"]["too_hard"]["percentage"]
    mastered_pct = analytics["category_distribution"]["mastered"]["percentage"]
    learning_zone_pct = analytics["category_distribution"]["learning_zone"]["percentage"]

    if too_hard_pct > 50:
        print("  ⚠️  WARNING: >50% problems are 'too hard' (0 correct)")
        print("      - Model may be too weak for this dataset")
        print("      - Consider: more samples (32-64), higher temperature, easier problems")
        print("")

    if mastered_pct > 50:
        print("  ⚠️  WARNING: >50% problems are 'mastered' (15-16 correct)")
        print("      - Dataset may be too easy for current model")
        print("      - Consider: stopping ReST, adding harder problems, or moving to GRPO")
        print("")

    if learning_zone_pct < 20:
        print("  ⚠️  NOTE: Learning zone (<25%) contains few problems")
        print("      - Most learning value comes from 4-11 correct range")
        print("      - May want to adjust temperature or sample count")
        print("")

    if learning_zone_pct >= 30:
        print("  ✅ GOOD: Learning zone contains {:.0f}% of problems".format(learning_zone_pct))
        print("      - This is ideal for ReST training")
        print("")

    estimated_samples = (
        analytics["category_distribution"]["challenging"]["count"] * 3 +  # Keep all (avg ~2)
        analytics["category_distribution"]["learning_zone"]["count"] * 3 +  # Keep top 3
        analytics["category_distribution"]["almost_mastered"]["count"] * 1  # Keep best 1
    )
    print(f"  📊 Estimated training samples (learning_zone strategy): ~{estimated_samples:,}")

    print("")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze generations.jsonl to see correct answer distribution"
    )

    parser.add_argument(
        "--generations_path", type=str, required=True,
        help="Path to generations JSONL file"
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Optional: Save analytics to JSON file"
    )
    parser.add_argument(
        "--show_problem_ids", action="store_true",
        help="Include problem IDs in output (verbose)"
    )

    args = parser.parse_args()

    # Run analytics
    analytics = analyze_generations(args.generations_path)

    # Print to console
    print_analytics(analytics)

    # Save to file if requested
    if args.output_path:
        # Remove problem_ids from categories if not requested (to keep file small)
        if not args.show_problem_ids:
            for category in analytics["category_distribution"]:
                analytics["category_distribution"][category]["problem_ids"] = \
                    f"[{len(analytics['category_distribution'][category]['problem_ids'])} items]"
            for bucket in analytics["correct_rate_buckets"]:
                analytics["correct_rate_buckets"][bucket]["problem_ids"] = \
                    f"[{len(analytics['correct_rate_buckets'][bucket]['problem_ids'])} items]"
            # Keep per_problem_details but make it optional
            analytics["per_problem_details"] = f"[{len(analytics['per_problem_details'])} items - use --show_problem_ids to include]"

        output_dir = os.path.dirname(args.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(analytics, f, indent=2, ensure_ascii=False)

        print(f"\nAnalytics saved to: {args.output_path}")


if __name__ == "__main__":
    main()
