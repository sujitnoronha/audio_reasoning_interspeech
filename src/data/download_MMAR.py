#!/usr/bin/env python3
"""
Download MMAR dataset for Audio Reasoning Challenge.

Usage:
    python download_mmar.py --output_dir ./data

This downloads:
    - MMAR-meta.json (metadata with questions, choices, audio paths)
    - mmar-audio.tar.gz (2.98 GB of audio files, auto-extracted)
"""

import argparse
import os
import shutil
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "BoJack/MMAR"


def download_mmar(output_dir: str, extract_audio: bool = True):
    """Download MMAR dataset from HuggingFace."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading MMAR dataset to {output_path}")
    print("-" * 50)

    # Download metadata
    print("Downloading MMAR-meta.json...")
    meta_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="MMAR-meta.json",
        repo_type="dataset",
    )
    dest_meta = output_path / "MMAR-meta.json"
    shutil.copy(meta_path, dest_meta)
    print(f"  Saved to: {dest_meta}")

    # Download audio archive
    print("\nDownloading mmar-audio.tar.gz (2.98 GB)...")
    audio_archive_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="mmar-audio.tar.gz",
        repo_type="dataset",
    )
    print(f"  Downloaded to cache: {audio_archive_path}")

    # Extract audio files
    if extract_audio:
        print("\nExtracting audio files...")
        with tarfile.open(audio_archive_path, "r:gz") as tar:
            tar.extractall(path=output_path)

        audio_dir = output_path / "audio"
        num_files = len(list(audio_dir.glob("*.wav")))
        print(f"  Extracted {num_files} audio files to: {audio_dir}")

    print("-" * 50)
    print("Download complete!")
    print(f"\nDataset structure:")
    print(f"  {output_path}/")
    print(f"  ├── MMAR-meta.json")
    print(f"  └── audio/")
    print(f"      └── *.wav ({num_files} files)")

    print(f"\nTo run inference:")
    print(f"  python infer_qwen25_omni.py \\")
    print(f"      --dataset_metadata {dest_meta} \\")
    print(f"      --dataset_audio_prefix {output_path}/ \\")
    print(f"      --output_path predictions.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Download MMAR dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Output directory for dataset (default: ./data)",
    )
    parser.add_argument(
        "--no_extract",
        action="store_true",
        help="Skip extracting audio archive",
    )
    args = parser.parse_args()

    download_mmar(args.output_dir, extract_audio=not args.no_extract)


if __name__ == "__main__":
    main()
