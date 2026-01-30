"""
Phase 2: Evaluate and Filter candidate solutions using Learning Zone approach.

Compares generated candidates to ground truth answers and filters
based on problem difficulty categories (Learning Zone framework).

Learning Zone Framework:
- Too Hard (0 correct):      SKIP - no signal to learn from
- Challenging (1-3 correct): KEEP all correct, weight 1.0x
- Learning Zone (4-11):      KEEP top 3, weight 1.5x (prioritize!)
- Almost Mastered (12-14):   KEEP best 1, weight 0.5x
- Mastered (15-16):          SKIP or 10% sample

Filter strategies:
- "learning_zone": Use the learning zone framework (recommended)
- "all": Keep all correct solutions (legacy)
- "best": Keep shortest correct solution per problem (legacy)
- "top_k": Keep top K correct solutions per problem (legacy)

Usage:
    python filter.py --generations_path ./outputs/rest/generations/generations.jsonl \
        --output_path ./outputs/rest/filtered/filtered.json \
        --filter_strategy learning_zone
"""

import argparse
import json
import os
import re
import random
import logging
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

from config import FilterConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Learning Zone Categories
# =============================================================================

@dataclass
class LearningZoneConfig:
    """Configuration for learning zone filtering."""
    # Category boundaries (for 16 samples)
    too_hard_max: int = 0           # 0 correct
    challenging_max: int = 3        # 1-3 correct
    learning_zone_max: int = 11     # 4-11 correct
    almost_mastered_max: int = 14   # 12-14 correct
    # 15-16 = mastered

    # How many solutions to keep per category
    challenging_keep: int = -1      # -1 = keep all
    learning_zone_keep: int = 3     # keep top 3
    almost_mastered_keep: int = 1   # keep best 1
    mastered_keep: int = 1          # keep best 1 (if included)

    # Weights for each category (for oversampling)
    challenging_weight: float = 1.0
    learning_zone_weight: float = 1.5
    almost_mastered_weight: float = 0.5
    mastered_weight: float = 0.1

    # Whether to include mastered problems
    include_mastered: bool = False
    mastered_sample_rate: float = 0.1  # 10% sampling

    # Whether to ONLY include learning_zone problems (skip challenging/almost_mastered)
    learning_zone_only: bool = False


def categorize_problem(correct_count: int, total_count: int, config: LearningZoneConfig) -> str:
    """Categorize a problem based on correct count.

    Args:
        correct_count: Number of correct solutions
        total_count: Total number of solutions generated
        config: Learning zone configuration

    Returns:
        Category name: "too_hard", "challenging", "learning_zone",
                       "almost_mastered", or "mastered"
    """
    if correct_count == 0:
        return "too_hard"
    elif correct_count <= config.challenging_max:
        return "challenging"
    elif correct_count <= config.learning_zone_max:
        return "learning_zone"
    elif correct_count <= config.almost_mastered_max:
        return "almost_mastered"
    else:
        return "mastered"


# =============================================================================
# Answer Extraction and Matching
# =============================================================================

def extract_answer(
    text: str,
    answer_pattern: str = r"<answer>(.*?)</answer>",
    fallback_patterns: List[str] = None,
) -> Optional[str]:
    """Extract answer from generated text.

    Tries primary pattern first, then fallback patterns.

    Args:
        text: Generated text
        answer_pattern: Primary regex pattern for answer extraction
        fallback_patterns: List of fallback patterns to try

    Returns:
        Extracted answer or None if not found
    """
    if fallback_patterns is None:
        fallback_patterns = [
            r"\(([A-D])\)",  # Match (A), (B), etc.
            r"^([A-D])\.",   # Match A., B., etc. at start
            r"answer is[:\s]*(.+?)(?:\.|$)",  # Match "answer is X"
            r"(?:correct|right)\s+(?:answer|option)\s+is[:\s]*(.+?)(?:\.|$)",
        ]

    # Try primary pattern
    match = re.search(answer_pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try fallback patterns
    for pattern in fallback_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()

    # Last resort: look for option letters in the last line
    lines = text.strip().split("\n")
    last_line = lines[-1] if lines else ""

    # Check for standalone option letter
    option_match = re.search(r"\b([A-D])\b", last_line)
    if option_match:
        return option_match.group(1)

    return None


def normalize_answer(answer: str, case_sensitive: bool = False) -> str:
    """Normalize answer for comparison.

    Args:
        answer: Answer string
        case_sensitive: Whether to preserve case

    Returns:
        Normalized answer
    """
    normalized = answer.strip()

    # Remove common prefixes
    normalized = re.sub(r"^\s*\([A-Da-d]\)\s*", "", normalized)
    normalized = re.sub(r"^\s*[A-Da-d]\.\s*", "", normalized)

    if not case_sensitive:
        normalized = normalized.lower()

    return normalized.strip()


def check_answer_match(
    predicted: str,
    ground_truth: str,
    case_sensitive: bool = False,
) -> bool:
    """Check if predicted answer matches ground truth.

    Handles various formats:
    - Exact match
    - Option letter match (A, B, C, D)
    - Content match (ignoring option prefix)

    Args:
        predicted: Predicted answer
        ground_truth: Ground truth answer
        case_sensitive: Whether comparison is case-sensitive

    Returns:
        True if answers match
    """
    pred_norm = normalize_answer(predicted, case_sensitive)
    gt_norm = normalize_answer(ground_truth, case_sensitive)

    # Exact match after normalization
    if pred_norm == gt_norm:
        return True

    # Extract just option letters for comparison
    pred_letter = re.search(r"^([A-Da-d])$", pred_norm)
    gt_letter = re.search(r"^\(([A-Da-d])\)", ground_truth)

    if pred_letter and gt_letter:
        return pred_letter.group(1).upper() == gt_letter.group(1).upper()

    # Check if ground truth option letter matches predicted
    if gt_letter:
        gt_option = gt_letter.group(1).upper()
        if pred_norm.upper() == gt_option:
            return True

    # Check content containment
    if len(gt_norm) > 2 and gt_norm in pred_norm:
        return True

    if len(pred_norm) > 2 and pred_norm in gt_norm:
        return True

    return False


# =============================================================================
# Candidate Evaluation
# =============================================================================

def evaluate_candidates(
    candidates: List,
    ground_truth: str,
    answer_pattern: str = None,  # Kept for backwards compatibility
    fallback_patterns: List[str] = None,  # Kept for backwards compatibility
    case_sensitive: bool = False,
) -> List[Tuple[int, Dict, str, bool]]:
    """Evaluate all candidates for a problem.

    Uses pre-parsed answer_prediction field from generation output.
    The generation process already extracts answers, so we just use those directly.

    Format expected:
    {
        "raw_output": "<think>...</think>\nANSWER: Double bass",
        "thinking_prediction": "...",
        "answer_text": "ANSWER: Double bass",
        "answer_prediction": "Double bass"  <- We use this directly
    }

    Returns:
        List of (index, candidate_dict, extracted_answer, is_correct)
    """
    results = []

    for idx, candidate in enumerate(candidates):
        # Handle both old (string) and new (dict) formats
        if isinstance(candidate, dict):
            # New structured format - use pre-parsed answer_prediction directly
            candidate_dict = candidate
            extracted = candidate.get("answer_prediction", "")

            # answer_prediction should already be clean from generation
            # No need to re-extract since generate.py already did this
        else:
            # Old/legacy format - string candidate (fallback only)
            candidate_dict = {
                "raw_output": candidate,
                "thinking_prediction": "",
                "answer_text": candidate,
                "answer_prediction": ""
            }
            # Only extract for legacy format
            extracted = extract_answer(candidate, answer_pattern or r"<answer>(.*?)</answer>", fallback_patterns)

        if not extracted:
            results.append((idx, candidate_dict, "", False))
            continue

        is_correct = check_answer_match(extracted, ground_truth, case_sensitive)
        results.append((idx, candidate_dict, extracted, is_correct))

    return results


# =============================================================================
# Solution Filtering
# =============================================================================

def _build_solution(idx: int, cand_dict: Dict, extracted_answer: str) -> Dict:
    """Build solution dict with structured fields."""
    if isinstance(cand_dict, dict):
        return {
            "index": idx,
            "candidate": cand_dict.get("raw_output", ""),
            "thinking_prediction": cand_dict.get("thinking_prediction", ""),
            "answer_text": cand_dict.get("answer_text", ""),
            "answer_prediction": cand_dict.get("answer_prediction", ""),
            "extracted_answer": extracted_answer,
        }
    else:
        return {
            "index": idx,
            "candidate": str(cand_dict),
            "thinking_prediction": "",
            "answer_text": "",
            "answer_prediction": "",
            "extracted_answer": extracted_answer,
        }


def get_solution_length(item: Tuple) -> int:
    """Get length for sorting - prefer concise answers."""
    cand_dict = item[1]
    if isinstance(cand_dict, dict):
        return len(cand_dict.get("raw_output", ""))
    return len(str(cand_dict))


def filter_by_category(
    correct_solutions: List[Tuple[int, Dict, str]],
    category: str,
    lz_config: LearningZoneConfig,
) -> Tuple[List[Dict], float]:
    """Filter solutions based on category using Learning Zone strategy.

    Args:
        correct_solutions: List of (index, candidate_dict, extracted_answer)
        category: Problem category
        lz_config: Learning zone configuration

    Returns:
        Tuple of (filtered_solutions, weight)
    """
    if not correct_solutions:
        return [], 0.0

    # Sort by length (prefer concise)
    sorted_correct = sorted(correct_solutions, key=get_solution_length)

    if category == "too_hard":
        # Should not happen, but handle gracefully
        return [], 0.0

    elif category == "challenging":
        # Skip if learning_zone_only is enabled
        if lz_config.learning_zone_only:
            return [], 0.0
        # Keep all correct solutions
        keep_count = len(sorted_correct) if lz_config.challenging_keep == -1 else lz_config.challenging_keep
        solutions = sorted_correct[:keep_count]
        weight = lz_config.challenging_weight

    elif category == "learning_zone":
        # Keep top K (default 3)
        keep_count = min(lz_config.learning_zone_keep, len(sorted_correct))
        solutions = sorted_correct[:keep_count]
        weight = lz_config.learning_zone_weight

    elif category == "almost_mastered":
        # Skip if learning_zone_only is enabled
        if lz_config.learning_zone_only:
            return [], 0.0
        # Keep best 1
        keep_count = min(lz_config.almost_mastered_keep, len(sorted_correct))
        solutions = sorted_correct[:keep_count]
        weight = lz_config.almost_mastered_weight

    elif category == "mastered":
        # Skip or sample
        if not lz_config.include_mastered:
            return [], 0.0
        if random.random() > lz_config.mastered_sample_rate:
            return [], 0.0
        keep_count = min(lz_config.mastered_keep, len(sorted_correct))
        solutions = sorted_correct[:keep_count]
        weight = lz_config.mastered_weight

    else:
        raise ValueError(f"Unknown category: {category}")

    return [_build_solution(idx, cand, ans) for idx, cand, ans in solutions], weight


def filter_correct_solutions(
    evaluated: List[Tuple[int, Dict, str, bool]],
    strategy: str = "top_k",
    top_k: int = 3,
) -> List[Dict]:
    """Filter correct solutions based on legacy strategy.

    Args:
        evaluated: List of (index, candidate_dict, extracted_answer, is_correct)
        strategy: "all", "best", or "top_k"
        top_k: Number of solutions to keep for top_k strategy

    Returns:
        List of selected solutions with structured output
    """
    correct = [(idx, cand_dict, ans) for idx, cand_dict, ans, is_correct in evaluated if is_correct]

    if not correct:
        return []

    if strategy == "all":
        return [_build_solution(idx, cand_dict, ans) for idx, cand_dict, ans in correct]

    elif strategy == "best":
        # Keep shortest correct solution
        best = min(correct, key=get_solution_length)
        return [_build_solution(best[0], best[1], best[2])]

    elif strategy == "top_k":
        # Keep top K (by length - prefer concise)
        sorted_correct = sorted(correct, key=get_solution_length)[:top_k]
        return [_build_solution(idx, cand_dict, ans) for idx, cand_dict, ans in sorted_correct]

    else:
        raise ValueError(f"Unknown filter strategy: {strategy}")


# =============================================================================
# Analytics
# =============================================================================

def compute_analytics(problems_data: List[Dict]) -> Dict:
    """Compute comprehensive analytics on generation results.

    Args:
        problems_data: List of problem dicts with evaluation results

    Returns:
        Analytics dictionary
    """
    analytics = {
        "summary": {},
        "category_distribution": {},
        "correct_count_histogram": defaultdict(int),
        "correct_rate_buckets": {
            "0%": 0,
            "1-25%": 0,
            "26-50%": 0,
            "51-75%": 0,
            "76-99%": 0,
            "100%": 0,
        },
        "problems_by_category": {
            "too_hard": [],
            "challenging": [],
            "learning_zone": [],
            "almost_mastered": [],
            "mastered": [],
        },
    }

    total_problems = len(problems_data)
    total_candidates = 0
    total_correct = 0
    correct_rates = []

    for problem in problems_data:
        correct_count = problem["correct_count"]
        total_count = problem["total_count"]
        category = problem["category"]

        total_candidates += total_count
        total_correct += correct_count

        # Histogram of correct counts
        analytics["correct_count_histogram"][correct_count] += 1

        # Correct rate buckets
        correct_rate = (correct_count / total_count * 100) if total_count > 0 else 0
        correct_rates.append(correct_rate)

        if correct_rate == 0:
            analytics["correct_rate_buckets"]["0%"] += 1
        elif correct_rate <= 25:
            analytics["correct_rate_buckets"]["1-25%"] += 1
        elif correct_rate <= 50:
            analytics["correct_rate_buckets"]["26-50%"] += 1
        elif correct_rate <= 75:
            analytics["correct_rate_buckets"]["51-75%"] += 1
        elif correct_rate < 100:
            analytics["correct_rate_buckets"]["76-99%"] += 1
        else:
            analytics["correct_rate_buckets"]["100%"] += 1

        # Problems by category
        analytics["problems_by_category"][category].append(problem["id"])

    # Summary statistics
    analytics["summary"] = {
        "total_problems": total_problems,
        "total_candidates": total_candidates,
        "total_correct": total_correct,
        "overall_accuracy": round(total_correct / total_candidates * 100, 2) if total_candidates > 0 else 0,
        "mean_correct_rate": round(sum(correct_rates) / len(correct_rates), 2) if correct_rates else 0,
        "median_correct_rate": round(sorted(correct_rates)[len(correct_rates)//2], 2) if correct_rates else 0,
    }

    # Category distribution
    for category in ["too_hard", "challenging", "learning_zone", "almost_mastered", "mastered"]:
        count = len(analytics["problems_by_category"][category])
        analytics["category_distribution"][category] = {
            "count": count,
            "percentage": round(count / total_problems * 100, 2) if total_problems > 0 else 0,
        }

    # Convert histogram keys to strings for JSON
    analytics["correct_count_histogram"] = dict(analytics["correct_count_histogram"])

    return analytics


def print_analytics(analytics: Dict):
    """Print analytics in a formatted way."""

    logger.info("")
    logger.info("=" * 70)
    logger.info("                    GENERATION ANALYTICS")
    logger.info("=" * 70)

    # Summary
    summary = analytics["summary"]
    logger.info("")
    logger.info("SUMMARY")
    logger.info("-" * 70)
    logger.info(f"  Total problems:        {summary['total_problems']}")
    logger.info(f"  Total candidates:      {summary['total_candidates']}")
    logger.info(f"  Total correct:         {summary['total_correct']}")
    logger.info(f"  Overall accuracy:      {summary['overall_accuracy']}%")
    logger.info(f"  Mean correct rate:     {summary['mean_correct_rate']}%")
    logger.info(f"  Median correct rate:   {summary['median_correct_rate']}%")

    # Category distribution
    logger.info("")
    logger.info("LEARNING ZONE CATEGORIES")
    logger.info("-" * 70)
    logger.info(f"  {'Category':<20} {'Count':>10} {'Percentage':>12}    Description")
    logger.info(f"  {'-'*20} {'-'*10} {'-'*12}    {'-'*20}")

    category_info = {
        "too_hard": "0 correct - SKIP",
        "challenging": "1-3 correct - Keep all",
        "learning_zone": "4-11 correct - PRIORITIZE!",
        "almost_mastered": "12-14 correct - Downsample",
        "mastered": "15-16 correct - Skip/10%",
    }

    for category in ["too_hard", "challenging", "learning_zone", "almost_mastered", "mastered"]:
        dist = analytics["category_distribution"][category]
        logger.info(f"  {category:<20} {dist['count']:>10} {dist['percentage']:>11}%    {category_info[category]}")

    # Correct rate buckets
    logger.info("")
    logger.info("CORRECT RATE DISTRIBUTION")
    logger.info("-" * 70)

    buckets = analytics["correct_rate_buckets"]
    total = sum(buckets.values())

    for bucket, count in buckets.items():
        pct = round(count / total * 100, 1) if total > 0 else 0
        bar = "█" * int(pct / 2)
        logger.info(f"  {bucket:>8}: {count:>6} ({pct:>5}%) {bar}")

    # Histogram (condensed)
    logger.info("")
    logger.info("CORRECT COUNT HISTOGRAM (per problem)")
    logger.info("-" * 70)

    histogram = analytics["correct_count_histogram"]
    max_count = max(histogram.values()) if histogram else 1

    for i in range(17):  # 0 to 16
        count = histogram.get(i, histogram.get(str(i), 0))
        bar_len = int(count / max_count * 40) if max_count > 0 else 0
        bar = "█" * bar_len
        logger.info(f"  {i:>2} correct: {count:>6} {bar}")

    logger.info("")
    logger.info("=" * 70)


# =============================================================================
# Main Processing
# =============================================================================

def process_generations(config: FilterConfig, lz_config: LearningZoneConfig = None) -> Dict:
    """Process all generations and filter correct solutions.

    Returns:
        Dict with filtered data, statistics, and analytics
    """
    if lz_config is None:
        lz_config = LearningZoneConfig()

    # Load generations
    logger.info(f"Loading generations from {config.generations_path}")
    generations = []
    with open(config.generations_path, "r", encoding="utf-8") as f:
        for line in f:
            generations.append(json.loads(line))

    logger.info(f"Loaded {len(generations)} problems with generations")

    # Statistics
    stats = {
        "total_problems": len(generations),
        "total_candidates": 0,
        "total_correct": 0,
        "problems_with_correct": 0,
        "problems_without_correct": 0,
        "filtered_samples": 0,
        "by_category": {
            "too_hard": {"count": 0, "samples": 0},
            "challenging": {"count": 0, "samples": 0},
            "learning_zone": {"count": 0, "samples": 0},
            "almost_mastered": {"count": 0, "samples": 0},
            "mastered": {"count": 0, "samples": 0},
        }
    }

    # Process each problem - first pass: evaluate all
    problems_data = []

    for gen in generations:
        problem_id = gen["id"]
        candidates = gen.get("candidates", [])
        ground_truth = gen.get("ground_truth", "")
        total_count = len(candidates)

        stats["total_candidates"] += total_count

        # Evaluate candidates
        evaluated = evaluate_candidates(
            candidates=candidates,
            ground_truth=ground_truth,
            answer_pattern=config.answer_pattern,
            fallback_patterns=config.fallback_patterns,
            case_sensitive=config.case_sensitive,
        )

        # Count correct
        correct_solutions = [(idx, cand, ans) for idx, cand, ans, is_correct in evaluated if is_correct]
        num_correct = len(correct_solutions)
        stats["total_correct"] += num_correct

        if num_correct > 0:
            stats["problems_with_correct"] += 1
        else:
            stats["problems_without_correct"] += 1

        # Categorize problem
        category = categorize_problem(num_correct, total_count, lz_config)
        stats["by_category"][category]["count"] += 1

        problems_data.append({
            "id": problem_id,
            "gen": gen,
            "evaluated": evaluated,
            "correct_solutions": correct_solutions,
            "correct_count": num_correct,
            "total_count": total_count,
            "category": category,
        })

    # Compute analytics
    analytics = compute_analytics(problems_data)

    # Print analytics
    print_analytics(analytics)

    # Second pass: filter based on strategy
    filtered_data = []
    problems_without_correct = []

    use_learning_zone = config.filter_strategy == "learning_zone"

    for problem in problems_data:
        problem_id = problem["id"]
        gen = problem["gen"]
        category = problem["category"]
        correct_solutions = problem["correct_solutions"]
        ground_truth = gen.get("ground_truth", "")

        if problem["correct_count"] == 0:
            problems_without_correct.append(problem_id)
            continue

        # Filter solutions
        if use_learning_zone:
            filtered, weight = filter_by_category(correct_solutions, category, lz_config)
        else:
            filtered = filter_correct_solutions(
                evaluated=problem["evaluated"],
                strategy=config.filter_strategy,
                top_k=config.top_k,
            )
            weight = 1.0

        if not filtered:
            continue

        # Create training samples
        for sol in filtered:
            sample = {
                "id": f"{problem_id}_sol{sol['index']}",
                "original_id": problem_id,
                "sound": gen.get("sound", ""),
                "audio_dir": gen.get("audio_dir", ""),  # Support combined datasets
                "question": gen.get("question", ""),
                "answer": ground_truth,
                "response": sol["candidate"],
                "thinking_prediction": sol.get("thinking_prediction", ""),
                "answer_text": sol.get("answer_text", ""),
                "answer_prediction": sol.get("answer_prediction", ""),
                "extracted_answer": sol.get("extracted_answer", ""),
                "category": category,
                "weight": weight,
            }
            filtered_data.append(sample)
            stats["filtered_samples"] += 1
            stats["by_category"][category]["samples"] += 1

    # Log filtering statistics
    logger.info("")
    logger.info("=" * 70)
    logger.info("                    FILTERING RESULTS")
    logger.info("=" * 70)
    logger.info(f"  Filter strategy:       {config.filter_strategy}")
    if use_learning_zone and lz_config.learning_zone_only:
        logger.info(f"  Learning zone only:    YES (skipping challenging/almost_mastered)")
    logger.info(f"  Total problems:        {stats['total_problems']}")
    logger.info(f"  Problems with correct: {stats['problems_with_correct']} ({100*stats['problems_with_correct']/stats['total_problems']:.1f}%)")
    logger.info(f"  Problems skipped:      {stats['problems_without_correct']} ({100*stats['problems_without_correct']/stats['total_problems']:.1f}%)")
    logger.info("")
    logger.info("  SAMPLES BY CATEGORY:")
    for category in ["too_hard", "challenging", "learning_zone", "almost_mastered", "mastered"]:
        cat_stats = stats["by_category"][category]
        logger.info(f"    {category:<20}: {cat_stats['count']:>4} problems -> {cat_stats['samples']:>4} samples")
    logger.info("")
    logger.info(f"  Total filtered samples: {stats['filtered_samples']}")
    logger.info("=" * 70)

    return {
        "filtered_data": filtered_data,
        "stats": stats,
        "analytics": analytics,
        "problems_without_correct": problems_without_correct,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Filter correct solutions with Learning Zone approach")

    parser.add_argument("--generations_path", type=str, required=True,
                        help="Path to generations JSONL file")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to output filtered JSON file")

    # Filter settings
    parser.add_argument("--filter_strategy", type=str, default="learning_zone",
                        choices=["learning_zone", "all", "best", "top_k"],
                        help="Filter strategy (learning_zone recommended)")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of solutions to keep per problem (for top_k/learning_zone)")

    # Learning Zone settings
    parser.add_argument("--learning_zone_only", action="store_true",
                        help="Only include learning_zone problems (4-11 correct), skip challenging/almost_mastered")
    parser.add_argument("--include_mastered", action="store_true",
                        help="Include mastered problems (15-16 correct) with 10% sampling")
    parser.add_argument("--mastered_sample_rate", type=float, default=0.1,
                        help="Sampling rate for mastered problems (default 0.1 = 10%)")

    # Answer extraction
    parser.add_argument("--answer_pattern", type=str, default=r"<answer>(.*?)</answer>",
                        help="Regex pattern for answer extraction")
    parser.add_argument("--case_sensitive", action="store_true",
                        help="Case-sensitive answer matching")

    # Random seed for reproducibility
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Build configs
    config = FilterConfig(
        generations_path=args.generations_path,
        output_path=args.output_path,
        filter_strategy=args.filter_strategy,
        top_k=args.top_k,
        answer_pattern=args.answer_pattern,
        case_sensitive=args.case_sensitive,
    )

    lz_config = LearningZoneConfig(
        include_mastered=args.include_mastered,
        mastered_sample_rate=args.mastered_sample_rate,
        learning_zone_keep=args.top_k,
        learning_zone_only=args.learning_zone_only,
    )

    # Process
    result = process_generations(config, lz_config)

    # Save filtered data
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(result["filtered_data"], f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(result['filtered_data'])} filtered samples to {args.output_path}")

    # Save statistics
    stats_path = args.output_path.replace(".json", "_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(result["stats"], f, indent=2)
    logger.info(f"Saved statistics to {stats_path}")

    # Save analytics
    analytics_path = args.output_path.replace(".json", "_analytics.json")
    with open(analytics_path, "w", encoding="utf-8") as f:
        json.dump(result["analytics"], f, indent=2)
    logger.info(f"Saved analytics to {analytics_path}")

    # Save problems without correct answers (for analysis)
    if result["problems_without_correct"]:
        no_correct_path = args.output_path.replace(".json", "_no_correct.json")
        with open(no_correct_path, "w", encoding="utf-8") as f:
            json.dump(result["problems_without_correct"], f, indent=2)
        logger.info(f"Saved {len(result['problems_without_correct'])} problem IDs without correct solutions to {no_correct_path}")

    # Save problems by category (for detailed analysis)
    by_category_path = args.output_path.replace(".json", "_by_category.json")
    with open(by_category_path, "w", encoding="utf-8") as f:
        json.dump(result["analytics"]["problems_by_category"], f, indent=2)
    logger.info(f"Saved problems by category to {by_category_path}")


if __name__ == "__main__":
    main()
