"""
ReST (Reinforced Self-Training) orchestration script.

Runs the full ReST training loop:
1. GENERATE: Create K candidate solutions per problem using vLLM
2. FILTER: Evaluate and keep only correct solutions
3. TRAIN: SFT on filtered data

Iterates multiple times with decreasing learning rate.

Usage:
    # First start vLLM server (in another terminal):
    vllm serve Qwen/Qwen3-Omni-30B-A3B-Thinking --port 8901 --host 127.0.0.1 \
        --dtype bfloat16 --max-model-len 32768 --allowed-local-media-path / -tp 4

    # Then run ReST training:
    python run_rest.py --data_path ../data/countingqa/CountingQA_MCQ.json \
        --audio_dir ../data/countingqa/counting_audios \
        --num_iterations 3
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import logging
from datetime import datetime
from typing import List, Optional

from config import ReSTrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_generation(
    data_path: str,
    audio_dir: str,
    output_dir: str,
    vllm_base_url: str,
    model_name: str,
    num_samples: int,
    temperature: float,
    resume: bool = False,
):
    """Run Phase 1: Candidate Generation."""
    logger.info("=" * 60)
    logger.info("PHASE 1: CANDIDATE GENERATION")
    logger.info("=" * 60)

    cmd = [
        sys.executable, "generate.py",
        "--data_path", data_path,
        "--audio_dir", audio_dir,
        "--output_dir", output_dir,
        "--vllm_base_url", vllm_base_url,
        "--model_name", model_name,
        "--num_samples", str(num_samples),
        "--temperature", str(temperature),
    ]

    if resume:
        cmd.append("--resume")

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

    if result.returncode != 0:
        raise RuntimeError(f"Generation failed with return code {result.returncode}")

    return os.path.join(output_dir, "generations.jsonl")


def run_filter(
    generations_path: str,
    output_path: str,
    filter_strategy: str,
    top_k: int,
):
    """Run Phase 2: Evaluation and Filtering."""
    logger.info("=" * 60)
    logger.info("PHASE 2: EVALUATION AND FILTERING")
    logger.info("=" * 60)

    cmd = [
        sys.executable, "filter.py",
        "--generations_path", generations_path,
        "--output_path", output_path,
        "--filter_strategy", filter_strategy,
        "--top_k", str(top_k),
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

    if result.returncode != 0:
        raise RuntimeError(f"Filtering failed with return code {result.returncode}")

    return output_path


def run_training(
    train_data_path: str,
    audio_dir: str,
    output_dir: str,
    model_name_or_path: str,
    learning_rate: float,
    num_epochs: int,
    models_dir: str,
    lora_targets: str = "all",
):
    """Run Phase 3: SFT Training."""
    logger.info("=" * 60)
    logger.info("PHASE 3: SFT TRAINING")
    logger.info("=" * 60)

    cmd = [
        sys.executable, "train.py",
        "--train_data_path", train_data_path,
        "--audio_dir", audio_dir,
        "--output_dir", output_dir,
        "--model_name_or_path", model_name_or_path,
        "--learning_rate", str(learning_rate),
        "--num_epochs", str(num_epochs),
        "--models_dir", models_dir,
        "--lora_targets", lora_targets,
        "--use_4bit",
        "--use_flash_attention",
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with return code {result.returncode}")


def run_rest_iteration(
    iteration: int,
    config: ReSTrainingConfig,
    current_model: str,
) -> str:
    """Run a single ReST iteration.

    Returns:
        Path to trained model for next iteration
    """
    logger.info("#" * 60)
    logger.info(f"REST ITERATION {iteration + 1}/{config.num_iterations}")
    logger.info("#" * 60)

    # Create iteration output directory
    iter_dir = os.path.join(config.output_base_dir, f"iteration_{iteration + 1}")
    os.makedirs(iter_dir, exist_ok=True)

    # Get learning rate for this iteration
    lr = config.learning_rates[iteration] if iteration < len(config.learning_rates) else config.learning_rates[-1]
    logger.info(f"Learning rate for iteration {iteration + 1}: {lr}")

    # Phase 1: Generate candidates
    gen_output_dir = os.path.join(iter_dir, "generations")
    generations_path = run_generation(
        data_path=config.data_path,
        audio_dir=config.audio_dir,
        output_dir=gen_output_dir,
        vllm_base_url=config.vllm_base_url,
        model_name=current_model,
        num_samples=config.num_samples_per_problem,
        temperature=config.temperature,
    )

    # Phase 2: Filter correct solutions
    filtered_path = os.path.join(iter_dir, "filtered", "filtered.json")
    os.makedirs(os.path.dirname(filtered_path), exist_ok=True)
    run_filter(
        generations_path=generations_path,
        output_path=filtered_path,
        filter_strategy=config.filter_strategy,
        top_k=config.top_k,
    )

    # Check if we have enough training data
    with open(filtered_path, "r") as f:
        filtered_data = json.load(f)

    if len(filtered_data) == 0:
        logger.warning(f"No correct solutions found in iteration {iteration + 1}!")
        logger.warning("Skipping training for this iteration.")
        return current_model

    logger.info(f"Filtered data contains {len(filtered_data)} training samples")

    # Phase 3: Train on filtered data
    train_output_dir = os.path.join(iter_dir, "training")
    run_training(
        train_data_path=filtered_path,
        audio_dir=config.audio_dir,
        output_dir=train_output_dir,
        model_name_or_path=current_model,
        learning_rate=lr,
        num_epochs=config.num_epochs,
        models_dir=config.models_dir,
        lora_targets=config.lora_targets,
    )

    # Find the saved model (latest in models dir)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, config.models_dir)

    # Get latest model by timestamp
    model_dirs = [d for d in os.listdir(models_dir) if d.startswith("rest_")]
    if model_dirs:
        latest_model = sorted(model_dirs)[-1]
        new_model_path = os.path.join(models_dir, latest_model)
        logger.info(f"New model saved at: {new_model_path}")
        return new_model_path

    return current_model


def main():
    parser = argparse.ArgumentParser(description="ReST Training Pipeline")

    # Data settings
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to training data JSON")
    parser.add_argument("--audio_dir", type=str, required=True,
                        help="Directory containing audio files")
    parser.add_argument("--output_base_dir", type=str, default="./outputs/rest",
                        help="Base output directory")
    parser.add_argument("--models_dir", type=str, default="../../models",
                        help="Directory to save models")

    # Model settings
    parser.add_argument("--base_model", type=str,
                        default="Qwen/Qwen3-Omni-30B-A3B-Thinking",
                        help="Base model name or path")

    # ReST settings
    parser.add_argument("--num_iterations", type=int, default=3,
                        help="Number of ReST iterations")
    parser.add_argument("--num_samples", type=int, default=16,
                        help="Samples per problem (K)")
    parser.add_argument("--num_epochs", type=int, default=2,
                        help="Training epochs per ReST iteration")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Sampling temperature")
    parser.add_argument("--filter_strategy", type=str, default="top_k",
                        choices=["all", "best", "top_k"],
                        help="Filter strategy")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Top K for filtering")

    # vLLM settings
    parser.add_argument("--vllm_base_url", type=str,
                        default="http://127.0.0.1:8901/v1",
                        help="vLLM server URL")

    # LoRA settings
    parser.add_argument("--lora_targets", type=str, default="all",
                        choices=["all", "attention", "moe"],
                        help="LoRA target modules: 'all' (attention+MoE), 'attention' (q/k/v/o only), 'moe' (gate/up/down only)")

    # Learning rate schedule
    parser.add_argument("--learning_rates", type=float, nargs="+",
                        default=[2e-5, 1e-5, 5e-6],
                        help="Learning rates per iteration")

    args = parser.parse_args()

    # Build config
    config = ReSTrainingConfig(
        num_iterations=args.num_iterations,
        learning_rates=args.learning_rates,
        data_path=args.data_path,
        audio_dir=args.audio_dir,
        output_base_dir=args.output_base_dir,
        models_dir=args.models_dir,
        base_model=args.base_model,
        num_epochs=args.num_epochs,
        num_samples_per_problem=args.num_samples,
        temperature=args.temperature,
        filter_strategy=args.filter_strategy,
        top_k=args.top_k,
        lora_targets=args.lora_targets,
        vllm_base_url=args.vllm_base_url,
    )

    # Create output directory
    os.makedirs(config.output_base_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(config.output_base_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({
            "num_iterations": config.num_iterations,
            "learning_rates": config.learning_rates,
            "data_path": config.data_path,
            "audio_dir": config.audio_dir,
            "base_model": config.base_model,
            "num_samples_per_problem": config.num_samples_per_problem,
            "temperature": config.temperature,
            "filter_strategy": config.filter_strategy,
            "top_k": config.top_k,
        }, f, indent=2)

    # Run ReST iterations
    current_model = config.base_model
    logger.info(f"Starting ReST training with {config.num_iterations} iterations")
    logger.info(f"Base model: {current_model}")

    for iteration in range(config.num_iterations):
        try:
            current_model = run_rest_iteration(iteration, config, current_model)
        except Exception as e:
            logger.error(f"Error in iteration {iteration + 1}: {e}")
            raise

    logger.info("=" * 60)
    logger.info("REST TRAINING COMPLETE!")
    logger.info(f"Final model: {current_model}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
