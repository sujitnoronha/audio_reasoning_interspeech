#!/usr/bin/env python3
"""
Download TACoS (Textually Annotated Cooking Scenes) captioning dataset.

Uses the Hugging Face datasets library to download from gijs/tacos-captioning.

Usage:
    python download_tacos.py --dry-run
    python download_tacos.py
    python download_tacos.py --output ./data
"""

import argparse
import json
from pathlib import Path
from datasets import load_dataset, Audio


def main():
    parser = argparse.ArgumentParser(
        description="Download TACoS captioning dataset from Hugging Face"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for dataset files (default: script directory)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just print dataset info, don't save"
    )
    args = parser.parse_args()

    # Resolve output path - default to script directory
    script_dir = Path(__file__).parent
    if args.output is None:
        output_dir = script_dir
    elif Path(args.output).is_absolute():
        output_dir = Path(args.output)
    else:
        output_dir = script_dir / args.output

    print("Loading TACoS captioning dataset from Hugging Face...")
    print("Dataset: gijs/tacos-captioning")
    print("-" * 60)

    # Load dataset without decoding audio (avoids librosa/soundfile dependency)
    ds = load_dataset("gijs/tacos-captioning")

    # Disable audio decoding to avoid needing librosa/soundfile
    for split_name in ds:
        if "audio" in ds[split_name].features:
            ds[split_name] = ds[split_name].cast_column("audio", Audio(decode=False))

    # Print dataset info
    print(f"\nDataset structure:")
    print(ds)

    for split_name in ds:
        split_data = ds[split_name]
        print(f"\n{split_name} split:")
        print(f"  Samples: {len(split_data)}")
        print(f"  Features: {list(split_data.features.keys())}")

        if len(split_data) > 0:
            print(f"\n  Example (first sample):")
            example = split_data[0]
            for key, value in example.items():
                if isinstance(value, dict) and "bytes" in value:
                    print(f"    {key}: <audio bytes>")
                elif isinstance(value, str) and len(value) > 100:
                    print(f"    {key}: {value[:100]}...")
                else:
                    print(f"    {key}: {value}")

    if args.dry_run:
        print("\n[Dry run - no files saved]")
        return

    # Create output directory and audio subdirectory
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Save each split as JSON and extract audio files
    for split_name in ds:
        split_data = ds[split_name]
        output_path = output_dir / f"tacos_{split_name}.json"
        print(f"\nSaving {split_name} split to {output_path}...")

        # Convert to list of dicts, extract audio files
        data = []
        for item in split_data:
            item_dict = dict(item)

            # Extract audio bytes to file
            if "audio" in item_dict and isinstance(item_dict["audio"], dict):
                audio_bytes = item_dict["audio"].get("bytes")
                file_name = item_dict.get("file_name", f"{len(data)}.mp3")

                if audio_bytes:
                    audio_path = audio_dir / file_name
                    with open(audio_path, "wb") as af:
                        af.write(audio_bytes)
                    # Store relative path in JSON
                    item_dict["audio"] = f"audio/{file_name}"
                else:
                    item_dict["audio"] = None

            data.append(item_dict)

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"  Saved {len(data)} samples")
        print(f"  Audio files saved to: {audio_dir}")

    print("\n" + "-" * 60)
    print("Download complete!")
    print(f"Files saved to: {output_dir}")


if __name__ == "__main__":
    main()
