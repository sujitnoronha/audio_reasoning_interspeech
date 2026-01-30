"""
GRPO (Group Relative Policy Optimization) training script for audio reasoning models.

Uses HuggingFace TRL GRPOTrainer with:
- 4-bit quantization
- LoRA PEFT finetuning
- Model merging at the end
"""

import argparse
import json
import os
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable

import torch
from torch.utils.data import Dataset
from transformers import HfArgumentParser
from trl import GRPOConfig, GRPOTrainer
from qwen_omni_utils import process_mm_info

from utils import (
    load_model_and_processor,
    setup_peft_model,
    merge_and_save_model,
    get_lora_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

USE_AUDIO_IN_VIDEO = True


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen3-Omni-30B",
        metadata={"help": "Path to pretrained model or model identifier (Qwen3-Omni-MoE thinking model)"}
    )
    use_4bit: bool = field(
        default=True,
        metadata={"help": "Whether to use 4-bit quantization"}
    )
    use_flash_attention: bool = field(
        default=False,
        metadata={"help": "Whether to use flash attention"}
    )


@dataclass
class DataArguments:
    train_data_path: str = field(
        metadata={"help": "Path to training data JSON file"}
    )
    eval_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to evaluation data JSON file"}
    )
    audio_dir: str = field(
        default="",
        metadata={"help": "Directory containing audio files"}
    )
    max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"}
    )
    max_prompt_length: int = field(
        default=1024,
        metadata={"help": "Maximum prompt length"}
    )
    max_completion_length: int = field(
        default=512,
        metadata={"help": "Maximum completion length for generation"}
    )


@dataclass
class LoRAArguments:
    lora_r: int = field(
        default=64,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=128,
        metadata={"help": "LoRA alpha scaling factor"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"}
    )


@dataclass
class GRPOTrainingArguments:
    output_dir: str = field(
        default="./outputs/grpo_training",
        metadata={"help": "Output directory for checkpoints"}
    )
    models_dir: str = field(
        default="../../models",
        metadata={"help": "Directory to save final merged model"}
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per device for training"}
    )
    per_device_eval_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per device for evaluation"}
    )
    gradient_accumulation_steps: int = field(
        default=8,
        metadata={"help": "Number of gradient accumulation steps"}
    )
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "Learning rate (lower for RL)"}
    )
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Warmup ratio"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Logging steps"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Save checkpoint every N steps"}
    )
    eval_steps: int = field(
        default=500,
        metadata={"help": "Evaluation steps"}
    )
    save_total_limit: int = field(
        default=3,
        metadata={"help": "Maximum number of checkpoints to keep"}
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Use bf16 precision"}
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Use gradient checkpointing"}
    )
    # GRPO specific
    num_generations: int = field(
        default=4,
        metadata={"help": "Number of generations per prompt for GRPO"}
    )
    beta: float = field(
        default=0.1,
        metadata={"help": "KL penalty coefficient"}
    )
    temperature: float = field(
        default=0.7,
        metadata={"help": "Sampling temperature for generation"}
    )


class GRPOAudioDataset(Dataset):
    """Dataset for GRPO training with audio prompts."""

    def __init__(
        self,
        data_path: str,
        processor,
        audio_dir: str = "",
        max_prompt_length: int = 1024,
    ):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.processor = processor
        self.audio_dir = audio_dir
        self.max_prompt_length = max_prompt_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, Any]:
        sample = self.data[idx]

        # Get audio path
        audio_path = os.path.join(self.audio_dir, sample["sound"])

        # Build question
        question = sample.get("question", "")
        answer = sample.get("answer", "")

        # If question not in sample, extract from conversations
        if not question:
            for conv in sample.get("conversations", []):
                if conv["from"] == "human":
                    question = conv["value"]
                elif conv["from"] == "gpt":
                    answer = conv["value"]

        # Clean question
        clean_question = question.replace("<sound>", "").strip()

        # For GRPO, we need prompt and reference answer
        return {
            "prompt": clean_question,
            "audio_path": audio_path,
            "answer": answer,  # Ground truth for reward computation
            "id": sample.get("id", str(idx)),
        }


def create_reward_function(processor) -> Callable:
    """Create reward function for GRPO training.

    The reward function evaluates generated answers against ground truth.
    For MCQ tasks, it checks if the generated answer matches the correct option.
    """

    def compute_reward(
        prompts: List[str],
        completions: List[str],
        ground_truths: List[str],
    ) -> List[float]:
        """Compute rewards for generated completions.

        Args:
            prompts: List of input prompts
            completions: List of generated completions
            ground_truths: List of correct answers

        Returns:
            List of reward scores (0 or 1 for MCQ)
        """
        rewards = []

        for completion, ground_truth in zip(completions, ground_truths):
            # Normalize both strings for comparison
            completion_clean = completion.strip().lower()
            ground_truth_clean = ground_truth.strip().lower()

            # Check for exact match
            if completion_clean == ground_truth_clean:
                rewards.append(1.0)
                continue

            # Check if ground truth is contained in completion
            if ground_truth_clean in completion_clean:
                rewards.append(0.8)
                continue

            # Check for option letter match (e.g., "(A)" or "A")
            option_pattern = r'\(([A-Da-d])\)'
            gt_match = re.search(option_pattern, ground_truth)
            comp_match = re.search(option_pattern, completion)

            if gt_match and comp_match:
                if gt_match.group(1).lower() == comp_match.group(1).lower():
                    rewards.append(0.9)
                    continue

            # No match
            rewards.append(0.0)

        return rewards

    return compute_reward


def format_prompt_for_grpo(sample: Dict, processor) -> str:
    """Format a sample into a prompt string for GRPO."""
    # Create conversation format for prompt only (no answer)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": sample["audio_path"]},
                {"type": "text", "text": sample["prompt"]},
            ],
        },
    ]

    # Apply chat template with generation prompt
    prompt = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )

    return prompt


def train_grpo(
    model_args: ModelArguments,
    data_args: DataArguments,
    lora_args: LoRAArguments,
    training_args: GRPOTrainingArguments,
):
    """Main GRPO training function."""

    logger.info("Loading model and processor...")
    model, processor = load_model_and_processor(
        model_args.model_name_or_path,
        use_4bit=model_args.use_4bit,
        use_flash_attention=model_args.use_flash_attention,
    )

    logger.info("Setting up LoRA...")
    lora_config = get_lora_config(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
    )
    model = setup_peft_model(model, lora_config)

    logger.info("Loading datasets...")
    train_dataset = GRPOAudioDataset(
        data_path=data_args.train_data_path,
        processor=processor,
        audio_dir=data_args.audio_dir,
        max_prompt_length=data_args.max_prompt_length,
    )

    eval_dataset = None
    if data_args.eval_data_path:
        eval_dataset = GRPOAudioDataset(
            data_path=data_args.eval_data_path,
            processor=processor,
            audio_dir=data_args.audio_dir,
            max_prompt_length=data_args.max_prompt_length,
        )

    # Create reward function
    reward_fn = create_reward_function(processor)

    # GRPO Config
    grpo_config = GRPOConfig(
        output_dir=training_args.output_dir,
        num_train_epochs=training_args.num_train_epochs,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        learning_rate=training_args.learning_rate,
        warmup_ratio=training_args.warmup_ratio,
        logging_steps=training_args.logging_steps,
        save_steps=training_args.save_steps,
        eval_steps=training_args.eval_steps if eval_dataset else None,
        save_total_limit=training_args.save_total_limit,
        bf16=training_args.bf16,
        gradient_checkpointing=training_args.gradient_checkpointing,
        # GRPO specific
        num_generations=training_args.num_generations,
        beta=training_args.beta,
        max_completion_length=data_args.max_completion_length,
        max_prompt_length=data_args.max_prompt_length,
        remove_unused_columns=False,
    )

    # Create GRPO Trainer
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        reward_funcs=reward_fn,
    )

    logger.info("Starting GRPO training...")
    trainer.train()

    # Save final checkpoint
    trainer.save_model(training_args.output_dir)

    # Merge and save final model
    logger.info("Merging LoRA adapters and saving final model...")
    models_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        training_args.models_dir
    )
    os.makedirs(models_dir, exist_ok=True)

    final_model_path = merge_and_save_model(
        model=model,
        processor=processor,
        output_dir=models_dir,
        training_style="grpo",
        model_name=model_args.model_name_or_path,
    )

    logger.info(f"Training complete! Final model saved to: {final_model_path}")

    return final_model_path


def main():
    parser = HfArgumentParser((
        ModelArguments,
        DataArguments,
        LoRAArguments,
        GRPOTrainingArguments,
    ))

    model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()

    train_grpo(model_args, data_args, lora_args, training_args)


if __name__ == "__main__":
    main()
