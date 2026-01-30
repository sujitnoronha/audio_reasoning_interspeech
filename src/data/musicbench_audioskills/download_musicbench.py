#!/usr/bin/env python3
"""
Download YouTube audio clips for MusicBench dataset.

Downloads the first N seconds (based on 'duration' field) from each YouTube video.
ID format: "CZZSQ6rd8h0_7" -> YouTube ID is "CZZSQ6rd8h0"

Usage:
    python download_musicbench.py --dry-run
    python download_musicbench.py --workers 4
    python download_musicbench.py --limit 10  # Test with 10 downloads
"""

import argparse
import json
import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def get_youtube_id(musicbench_id: str) -> str:
    """
    Extract YouTube ID from MusicBench ID.

    "CZZSQ6rd8h0_7" -> "CZZSQ6rd8h0"
    """
    parts = musicbench_id.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return musicbench_id


def download_audio(youtube_id: str, output_path: Path,
                   duration: float = 10.0, sample_rate: int = 16000) -> bool:
    """
    Download first N seconds from a YouTube video as WAV.

    Args:
        youtube_id: YouTube video ID
        output_path: Output WAV file path (must end in .wav)
        duration: Duration to download in seconds
        sample_rate: Audio sample rate

    Returns:
        True if successful, False otherwise
    """
    if output_path.exists():
        return True  # Already downloaded

    output_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={youtube_id}"

    # Base path without extension - yt-dlp will add the actual extension
    temp_base = output_path.with_suffix('')

    try:
        # Step 1: Download audio with yt-dlp (let it choose format)
        download_cmd = [
            "yt-dlp",
            "--extract-audio",
            "--audio-quality", "0",
            "--output", str(temp_base) + ".%(ext)s",
            "--no-playlist",
            "--quiet",
            "--no-warnings",
            url
        ]

        subprocess.run(download_cmd, capture_output=True, text=True, timeout=180)

        # Find the downloaded file (could be .webm, .m4a, .opus, etc.)
        temp_files = list(output_path.parent.glob(f"{temp_base.name}.*"))
        temp_files = [f for f in temp_files if f.suffix != '.wav']  # Exclude any existing wav

        if not temp_files:
            return False

        temp_file = temp_files[0]

        # Step 2: Convert to WAV with ffmpeg
        convert_cmd = [
            "ffmpeg",
            "-y",
            "-i", str(temp_file),
            "-t", str(duration),
            "-ar", str(sample_rate),
            "-ac", "1",
            "-f", "wav",
            str(output_path)
        ]

        subprocess.run(convert_cmd, capture_output=True, text=True, timeout=60)

        # Cleanup temp file
        temp_file.unlink(missing_ok=True)

        return output_path.exists()

    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        # Cleanup any leftover temp files
        for ext in ['.webm', '.m4a', '.opus', '.ogg', '.mp3']:
            temp = output_path.with_suffix(ext)
            if temp.exists():
                temp.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Download MusicBench audio clips from YouTube"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="MusicBench.json",
        help="Input JSON file (default: MusicBench.json)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="audio",
        help="Output directory for audio files (default: ./audio)"
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
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of downloads (for testing)"
    )
    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    input_path = script_dir / args.input if not Path(args.input).is_absolute() else Path(args.input)
    output_dir = script_dir / args.output if not Path(args.output).is_absolute() else Path(args.output)

    # Load dataset
    print(f"Loading {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)

    print(f"Total samples: {len(data)}")

    # Parse all IDs
    downloads = []
    for item in data:
        musicbench_id = item['id']
        duration = item.get('duration', 10.0)

        youtube_id = get_youtube_id(musicbench_id)
        # Name file with the ID (e.g., CZZSQ6rd8h0_7.wav)
        output_path = output_dir / f"{musicbench_id}.wav"

        downloads.append({
            'youtube_id': youtube_id,
            'duration': duration,
            'output_path': output_path,
            'musicbench_id': musicbench_id
        })

    # Get unique YouTube videos
    unique_videos = set(d['youtube_id'] for d in downloads)
    print(f"Unique YouTube videos: {len(unique_videos)}")

    # Check already downloaded
    already_downloaded = sum(1 for d in downloads if d['output_path'].exists())
    print(f"Already downloaded: {already_downloaded}")
    print(f"Remaining: {len(downloads) - already_downloaded}")

    if args.dry_run:
        print("\n[Dry run - no downloads performed]")
        # Show first few examples
        print("\nFirst 5 samples:")
        for d in downloads[:5]:
            print(f"  {d['musicbench_id']} -> {d['youtube_id']} ({d['duration']}s)")
        return

    # Check for yt-dlp
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\nError: yt-dlp not found. Install with: pip install yt-dlp")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to only non-downloaded
    to_download = [d for d in downloads if not d['output_path'].exists()]

    if args.limit:
        to_download = to_download[:args.limit]

    print(f"\nDownloading {len(to_download)} clips to: {output_dir}")
    print("-" * 60)

    successful = 0
    failed = 0
    failed_ids = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for d in to_download:
            future = executor.submit(
                download_audio,
                d['youtube_id'],
                d['output_path'],
                d['duration'],
                args.sample_rate
            )
            futures[future] = d

        with tqdm(total=len(futures), desc="Downloading") as pbar:
            for future in as_completed(futures):
                d = futures[future]
                try:
                    if future.result():
                        successful += 1
                    else:
                        failed += 1
                        failed_ids.append(d['musicbench_id'])
                except Exception as e:
                    failed += 1
                    failed_ids.append(d['musicbench_id'])
                pbar.update(1)

    # Log failed downloads
    if failed_ids:
        failed_log = output_dir / "failed_downloads.txt"
        with open(failed_log, 'w') as f:
            f.write('\n'.join(failed_ids))
        print(f"\nFailed IDs logged to: {failed_log}")

    print("-" * 60)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total in output dir: {len(list(output_dir.glob('*.wav')))}")


if __name__ == "__main__":
    main()
