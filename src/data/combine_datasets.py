"""
Combine CountingQA and MusicBench datasets into train/test splits.

- Samples 20% of CountingQA to avoid data imbalance
- Uses all MusicBench data
- Creates train/test splits (80/20)

Usage:
    python combine_datasets.py
    python combine_datasets.py --countingqa_sample_ratio 0.2 --test_ratio 0.2
"""

import argparse
import json
import os
import random
from typing import List, Dict, Tuple

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

COUNTINGQA_JSON = os.path.join(SCRIPT_DIR, "countingqa_audioskills/CountingQA_MCQ.json")
COUNTINGQA_AUDIO_DIR = "countingqa_audioskills/counting_audios"

MUSICBENCH_JSON = os.path.join(SCRIPT_DIR, "musicbench_audioskills/MusicBench_clean.json")
MUSICBENCH_AUDIO_DIR = "musicbench_audioskills/audio"


def load_json(path: str) -> List[Dict]:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def add_dataset_info(samples: List[Dict], dataset_name: str, audio_dir: str) -> List[Dict]:
    """Add dataset info to each sample."""
    result = []
    for sample in samples:
        updated = sample.copy()
        updated["dataset"] = dataset_name
        updated["audio_dir"] = audio_dir
        result.append(updated)
    return result


def train_test_split(data: List[Dict], test_ratio: float, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """Split data into train and test sets."""
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * (1 - test_ratio))
    train = shuffled[:split_idx]
    test = shuffled[split_idx:]

    return train, test


def main():
    parser = argparse.ArgumentParser(description="Combine datasets with train/test split")
    parser.add_argument("--countingqa_sample_ratio", type=float, default=0.2,
                        help="Ratio of CountingQA to sample (default: 0.2 = 20%%)")
    parser.add_argument("--test_ratio", type=float, default=0.2,
                        help="Ratio of data for test set (default: 0.2 = 20%%)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: same as script)")
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = args.output_dir or SCRIPT_DIR

    print("=" * 60)
    print("COMBINING DATASETS")
    print("=" * 60)
    print(f"CountingQA sample ratio: {args.countingqa_sample_ratio:.0%}")
    print(f"Test ratio: {args.test_ratio:.0%}")
    print(f"Random seed: {args.seed}")
    print()

    # Load CountingQA and sample
    countingqa_full = load_json(COUNTINGQA_JSON)
    print(f"CountingQA total: {len(countingqa_full)} samples")

    sample_size = int(len(countingqa_full) * args.countingqa_sample_ratio)
    countingqa_data = random.sample(countingqa_full, sample_size)
    print(f"CountingQA sampled: {len(countingqa_data)} samples ({args.countingqa_sample_ratio:.0%})")

    # Load MusicBench (use all)
    musicbench_data = load_json(MUSICBENCH_JSON)
    print(f"MusicBench: {len(musicbench_data)} samples (100%)")

    # Add dataset info
    countingqa_data = add_dataset_info(countingqa_data, "countingqa", COUNTINGQA_AUDIO_DIR)
    musicbench_data = add_dataset_info(musicbench_data, "musicbench", MUSICBENCH_AUDIO_DIR)

    # Create train/test splits for each dataset
    countingqa_train, countingqa_test = train_test_split(countingqa_data, args.test_ratio, args.seed)
    musicbench_train, musicbench_test = train_test_split(musicbench_data, args.test_ratio, args.seed + 1)

    print()
    print("Split sizes:")
    print(f"  CountingQA train: {len(countingqa_train)}, test: {len(countingqa_test)}")
    print(f"  MusicBench train: {len(musicbench_train)}, test: {len(musicbench_test)}")

    # Combine and shuffle
    train_data = countingqa_train + musicbench_train
    test_data = countingqa_test + musicbench_test

    random.shuffle(train_data)
    random.shuffle(test_data)

    print()
    print(f"Combined train: {len(train_data)} samples")
    print(f"Combined test: {len(test_data)} samples")
    print(f"Total: {len(train_data) + len(test_data)} samples")

    # Save
    train_path = os.path.join(output_dir, "combined_train.json")
    test_path = os.path.join(output_dir, "combined_test.json")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved train set to: {train_path}")

    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    print(f"Saved test set to: {test_path}")

    # Print statistics
    print()
    print("=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    for split_name, split_data in [("train", train_data), ("test", test_data)]:
        countingqa_count = sum(1 for s in split_data if s["dataset"] == "countingqa")
        musicbench_count = sum(1 for s in split_data if s["dataset"] == "musicbench")
        print(f"{split_name}: {len(split_data)} total | "
              f"CountingQA: {countingqa_count} ({100*countingqa_count/len(split_data):.1f}%) | "
              f"MusicBench: {musicbench_count} ({100*musicbench_count/len(split_data):.1f}%)")

    # Print sample entry
    print()
    print("Sample entry from combined dataset:")
    sample = train_data[0]
    print(json.dumps({k: v for k, v in sample.items() if k != "conversations"}, indent=2))


if __name__ == "__main__":
    main()
