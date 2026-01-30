"""
ReST (Reinforced Self-Training) training module.

Three-phase training pipeline:
1. GENERATE: Create K candidate solutions using vLLM serve
2. FILTER: Evaluate and keep only correct solutions
3. TRAIN: SFT on filtered data with LoRA

Usage:
    # Start vLLM server first:
    vllm serve Qwen/Qwen3-Omni-30B-A3B-Thinking --port 8901 --host 127.0.0.1 \
        --dtype bfloat16 --max-model-len 32768 --allowed-local-media-path / -tp 4

    # Run full ReST pipeline:
    python run_rest.py --data_path path/to/data.json --audio_dir path/to/audio

    # Or run individual phases:
    python generate.py --data_path data.json --audio_dir audio/ --num_samples 16
    python filter.py --generations_path generations.jsonl --output_path filtered.json
    python train.py --train_data_path filtered.json --audio_dir audio/
"""

from .config import (
    GenerationConfig,
    FilterConfig,
    TrainConfig,
    ReSTrainingConfig,
)

__all__ = [
    "GenerationConfig",
    "FilterConfig",
    "TrainConfig",
    "ReSTrainingConfig",
]
