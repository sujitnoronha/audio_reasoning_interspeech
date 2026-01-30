"""
Clean and augment MusicBench dataset.

- Extracts question and answer keys from conversations
- Removes option letter prefix from answers (e.g., "(B) Vivace" -> "Vivace")
- Only includes samples where audio file exists
"""

import argparse
import json
import os
import re
import logging
from typing import List, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_answer_text(answer: str) -> str:
    """Remove option letter prefix from answer.

    Examples:
        "(B) Vivace" -> "Vivace"
        "(A) Lento" -> "Lento"
        "Vivace" -> "Vivace" (no change if no prefix)
    """
    # Pattern to match option prefix like "(A) ", "(B) ", etc.
    pattern = r'^\s*\([A-Da-d]\)\s*'
    cleaned = re.sub(pattern, '', answer).strip()
    return cleaned if cleaned else answer


def clean_sample(sample: Dict, audio_dir: str) -> Optional[Dict]:
    """Clean a single sample.

    Returns:
        Cleaned sample dict with question and answer keys, or None if audio doesn't exist.
    """
    # Check if audio file exists
    audio_path = os.path.join(audio_dir, sample["sound"])
    if not os.path.exists(audio_path):
        return None

    # Extract question and answer from conversations
    question = ""
    answer_raw = ""

    for conv in sample["conversations"]:
        if conv["from"] == "human":
            question = conv["value"]
        elif conv["from"] == "gpt":
            answer_raw = conv["value"]

    # Clean the answer - remove option letter prefix
    answer = extract_answer_text(answer_raw)

    # Create cleaned sample
    cleaned_sample = {
        "id": sample["id"],
        "sound": sample["sound"],
        "duration": sample.get("duration", 0),
        "question": question,
        "answer": answer,
        "conversations": sample["conversations"]  # Keep original conversations
    }

    return cleaned_sample


def main():
    parser = argparse.ArgumentParser(description="Clean MusicBench dataset")
    parser.add_argument("--input_json", type=str, default="MusicBench.json",
                        help="Path to input MusicBench.json")
    parser.add_argument("--output_json", type=str, default="MusicBench_clean.json",
                        help="Path to output cleaned JSON")
    parser.add_argument("--audio_dir", type=str, default="audio",
                        help="Directory containing audio files")

    args = parser.parse_args()

    # Resolve paths relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_json = os.path.join(script_dir, args.input_json) if not os.path.isabs(args.input_json) else args.input_json
    output_json = os.path.join(script_dir, args.output_json) if not os.path.isabs(args.output_json) else args.output_json
    audio_dir = os.path.join(script_dir, args.audio_dir) if not os.path.isabs(args.audio_dir) else args.audio_dir

    # Load input dataset
    logger.info(f"Loading dataset from {input_json}")
    with open(input_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    logger.info(f"Total samples in input: {len(dataset)}")

    # Process samples
    cleaned_samples = []
    missing_audio_count = 0

    for sample in dataset:
        cleaned = clean_sample(sample, audio_dir)
        if cleaned is not None:
            cleaned_samples.append(cleaned)
        else:
            missing_audio_count += 1

    # Save output
    logger.info(f"Saving {len(cleaned_samples)} samples to {output_json}")
    logger.info(f"Skipped {missing_audio_count} samples due to missing audio files")

    os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(cleaned_samples, f, indent=2, ensure_ascii=False)

    logger.info("Done!")


if __name__ == "__main__":
    main()
