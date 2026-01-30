"""
ReST (Reinforced Self-Training) training script for audio reasoning models.

Uses HuggingFace TRL and Trainer with:
- 4-bit quantization
- LoRA PEFT finetuning
- Model merging at the end
"""

import argparse
import json
import os
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch
from torch.utils.data import Dataset
from transformers import (
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    DataCollatorWithPadding,
)
from trl import SFTTrainer, SFTConfig
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
class ReSTrainingArguments(TrainingArguments):
    output_dir: str = field(
        default="./outputs/rest_training",
        metadata={"help": "Output directory for checkpoints"}
    )
    models_dir: str = field(
        default="../../models",
        metadata={"help": "Directory to save final merged model"}
    )
    num_train_epochs: float = field(
        default=3.0,
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
        default=2e-5,
        metadata={"help": "Learning rate"}
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
    optim: str = field(
        default="paged_adamw_8bit",
        metadata={"help": "Optimizer to use"}
    )
    # ReST specific
    rest_iterations: int = field(
        default=3,
        metadata={"help": "Number of ReST iterations (Grow-Improve cycles)"}
    )
    filter_threshold: float = field(
        default=0.5,
        metadata={"help": "Threshold for filtering correct responses in ReST"}
    )


class AudioReasoningDataset(Dataset):
    """Dataset for audio reasoning tasks."""

    def __init__(
        self,
        data_path: str,
        processor,
        audio_dir: str = "",
        max_length: int = 2048,
    ):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.processor = processor
        self.audio_dir = audio_dir
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, Any]:
        sample = self.data[idx]

        # Get audio path
        audio_path = os.path.join(self.audio_dir, sample["sound"])

        # Build conversation
        question = sample.get("question", "")
        answer = sample.get("answer", "")

        # If question not in sample, extract from conversations
        if not question:
            for conv in sample.get("conversations", []):
                if conv["from"] == "human":
                    question = conv["value"]
                elif conv["from"] == "gpt":
                    answer = conv["value"]

        # Create conversation format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": question.replace("<sound>", "").strip()},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer},
                ],
            },
        ]

        # Process multimedia
        audios, images, videos = process_mm_info(
            conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO
        )

        # Apply chat template
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )

        # Tokenize
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
        )

        # Squeeze batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Create labels (same as input_ids for causal LM)
        inputs["labels"] = inputs["input_ids"].clone()

        return inputs


class AudioDataCollator:
    """Data collator for audio reasoning datasets."""

    def __init__(self, processor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Stack all tensors
        batch = {}
        for key in features[0].keys():
            if isinstance(features[0][key], torch.Tensor):
                batch[key] = torch.stack([f[key] for f in features])
            else:
                batch[key] = [f[key] for f in features]

        return batch


def train_rest(
    model_args: ModelArguments,
    data_args: DataArguments,
    lora_args: LoRAArguments,
    training_args: ReSTrainingArguments,
):
    """Main ReST training loop."""

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
    train_dataset = AudioReasoningDataset(
        data_path=data_args.train_data_path,
        processor=processor,
        audio_dir=data_args.audio_dir,
        max_length=data_args.max_length,
    )

    eval_dataset = None
    if data_args.eval_data_path:
        eval_dataset = AudioReasoningDataset(
            data_path=data_args.eval_data_path,
            processor=processor,
            audio_dir=data_args.audio_dir,
            max_length=data_args.max_length,
        )

    data_collator = AudioDataCollator(processor, max_length=data_args.max_length)

    # ReST Training Loop
    for iteration in range(training_args.rest_iterations):
        logger.info(f"=== ReST Iteration {iteration + 1}/{training_args.rest_iterations} ===")

        # Update output directory for this iteration
        iteration_output_dir = os.path.join(
            training_args.output_dir, f"iteration_{iteration + 1}"
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=iteration_output_dir,
                num_train_epochs=training_args.num_train_epochs,
                per_device_train_batch_size=training_args.per_device_train_batch_size,
                per_device_eval_batch_size=training_args.per_device_eval_batch_size,
                gradient_accumulation_steps=training_args.gradient_accumulation_steps,
                learning_rate=training_args.learning_rate,
                warmup_ratio=training_args.warmup_ratio,
                logging_steps=training_args.logging_steps,
                save_steps=training_args.save_steps,
                eval_steps=training_args.eval_steps if eval_dataset else None,
                eval_strategy="steps" if eval_dataset else "no",
                save_total_limit=training_args.save_total_limit,
                bf16=training_args.bf16,
                gradient_checkpointing=training_args.gradient_checkpointing,
                optim=training_args.optim,
                remove_unused_columns=False,
                dataloader_pin_memory=False,
            ),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        # Train
        logger.info(f"Starting training for iteration {iteration + 1}...")
        trainer.train()

        # Save iteration checkpoint
        trainer.save_model(iteration_output_dir)
        logger.info(f"Saved iteration {iteration + 1} checkpoint to {iteration_output_dir}")

        # TODO: In full ReST, here you would:
        # 1. Generate responses with current model
        # 2. Filter based on correctness (using filter_threshold)
        # 3. Update training data with filtered samples
        # For now, we just continue training on same data

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
        training_style="rest",
        model_name=model_args.model_name_or_path,
    )

    logger.info(f"Training complete! Final model saved to: {final_model_path}")

    return final_model_path


def main():
    parser = HfArgumentParser((
        ModelArguments,
        DataArguments,
        LoRAArguments,
        ReSTrainingArguments,
    ))

    model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()

    train_rest(model_args, data_args, lora_args, training_args)


if __name__ == "__main__":
    main()
