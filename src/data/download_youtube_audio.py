#!/usr/bin/env python3
"""
Download YouTube audio clips for ReasonAQA dataset.

Extracts YouTube IDs from the dataset JSON files and downloads audio in WAV format.

Usage:
    python download_youtube_audio.py --input test.json --output ./audio
    python download_youtube_audio.py --input train.json val.json test.json --output ./audio
"""

import argparse
import json
import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def extract_youtube_ids(json_files: list[str]) -> dict[str, set[str]]:
    """
    Extract YouTube IDs from JSON files, grouped by subfolder.

    Returns:
        Dict mapping subfolder path to set of YouTube IDs
    """
    ids_by_folder = {}

    for json_file in json_files:
        print(f"Processing {json_file}...")
        with open(json_file, 'r') as f:
            data = json.load(f)

        for item in data:
            for key in ['filepath1', 'filepath2']:
                filepath = item.get(key, '')
                if not filepath:
                    continue

                # Parse path like "AudioCapsLarger/test/Y7fmOlUlwoNg.wav"
                path = Path(filepath)
                folder = str(path.parent)  # e.g., "AudioCapsLarger/test"
                filename = path.stem        # e.g., "Y7fmOlUlwoNg"

                if folder not in ids_by_folder:
                    ids_by_folder[folder] = set()
                ids_by_folder[folder].add(filename)

    return ids_by_folder


def download_audio(video_id: str, output_path: Path, sample_rate: int = 16000) -> bool:
    """
    Download YouTube video and extract audio as WAV.

    Args:
        video_id: YouTube video ID
        output_path: Full path for output WAV file
        sample_rate: Audio sample rate (default 16kHz for speech models)

    Returns:
        True if successful, False otherwise
    """
    if output_path.exists():
        return True  # Already downloaded

    url = f"https://www.youtube.com/watch?v={video_id}"

    # Use yt-dlp to download and convert to WAV
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--postprocessor-args", f"ffmpeg:-ar {sample_rate} -ac 1",
        "--output", str(output_path.with_suffix('.%(ext)s')),
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        url
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def download_folder(folder: str, video_ids: set[str], output_base: Path,
                    max_workers: int = 4) -> tuple[int, int]:
    """
    Download all videos for a folder.

    Returns:
        Tuple of (successful_count, failed_count)
    """
    output_dir = output_base / folder
    output_dir.mkdir(parents=True, exist_ok=True)

    successful = 0
    failed = 0
    failed_ids = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for video_id in video_ids:
            output_path = output_dir / f"{video_id}.wav"
            future = executor.submit(download_audio, video_id, output_path)
            futures[future] = video_id

        with tqdm(total=len(futures), desc=f"Downloading {folder}") as pbar:
            for future in as_completed(futures):
                video_id = futures[future]
                try:
                    if future.result():
                        successful += 1
                    else:
                        failed += 1
                        failed_ids.append(video_id)
                except Exception:
                    failed += 1
                    failed_ids.append(video_id)
                pbar.update(1)

    # Log failed downloads
    if failed_ids:
        failed_log = output_dir / "failed_downloads.txt"
        with open(failed_log, 'w') as f:
            f.write('\n'.join(failed_ids))
        print(f"  Failed IDs logged to: {failed_log}")

    return successful, failed


def main():
    parser = argparse.ArgumentParser(
        description="Download YouTube audio for ReasonAQA dataset"
    )
    parser.add_argument(
        "--input", "-i",
        nargs="+",
        required=True,
        help="Input JSON file(s) containing filepath1/filepath2 fields"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./audio",
        help="Output directory for downloaded audio (default: ./audio)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of parallel downloads (default: 4)"
    )
    parser.add_argument(
        "--sample-rate", "-sr",
        type=int,
        default=16000,
        help="Audio sample rate in Hz (default: 16000)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just print stats, don't download"
    )
    args = parser.parse_args()

    output_base = Path(args.output)

    # Extract all YouTube IDs
    print("Extracting YouTube IDs from JSON files...")
    ids_by_folder = extract_youtube_ids(args.input)

    # Print summary
    total_ids = sum(len(ids) for ids in ids_by_folder.values())
    print(f"\nFound {total_ids} unique video IDs across {len(ids_by_folder)} folders:")
    for folder, ids in sorted(ids_by_folder.items()):
        print(f"  {folder}: {len(ids)} videos")

    if args.dry_run:
        print("\n[Dry run - no downloads performed]")
        return

    # Check for yt-dlp
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\nError: yt-dlp not found. Install with: pip install yt-dlp")
        return

    # Download each folder
    print(f"\nDownloading to: {output_base.absolute()}")
    print("-" * 60)

    total_success = 0
    total_failed = 0

    for folder, ids in sorted(ids_by_folder.items()):
        success, failed = download_folder(
            folder, ids, output_base,
            max_workers=args.workers
        )
        total_success += success
        total_failed += failed
        print(f"  {folder}: {success} successful, {failed} failed")

    print("-" * 60)
    print(f"Total: {total_success} successful, {total_failed} failed")


if __name__ == "__main__":
    main()
