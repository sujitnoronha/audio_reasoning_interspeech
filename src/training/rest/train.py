"""
Phase 3: SFT Training on filtered correct solutions.

Fine-tunes the model on its own correct solutions using LoRA PEFT
with 4-bit quantization.

Usage:
    python train.py --train_data_path ./outputs/rest/filtered/filtered.json \
        --audio_dir ../data/countingqa/counting_audios \
        --output_dir ./outputs/rest/training/iter1 \
        --learning_rate 2e-5 --num_epochs 2
"""

import argparse
import json
import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime

import torch
from torch.utils.data import Dataset
from transformers import (
    TrainingArguments,
    Trainer,
    TrainerCallback,
    HfArgumentParser,
    Qwen3OmniMoeThinkerForConditionalGeneration,  # Use Thinker for text-only training
    Qwen3OmniMoeProcessor,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from qwen_omni_utils import process_mm_info

from config import TrainConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

USE_AUDIO_IN_VIDEO = True


def get_bnb_config() -> BitsAndBytesConfig:
    """Get BitsAndBytes config for 4-bit quantization."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def get_lora_config(
    r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    lora_targets: str = "all",
) -> LoraConfig:
    """Get LoRA configuration for PEFT.

    Args:
        lora_targets: Which modules to target.
            "all" = attention + MoE experts (default, current behavior)
            "attention" = attention only (q/k/v/o_proj)
            "moe" = MoE expert layers only (gate/up/down_proj)
    """
    if lora_targets == "attention":
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif lora_targets == "moe":
        target_modules = ["gate_proj", "up_proj", "down_proj"]
    else:  # "all"
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        # Don't set task_type for multimodal models - it causes forward signature issues
    )


def load_model_and_processor(
    model_name_or_path: str,
    use_4bit: bool = True,
    use_flash_attention: bool = True,
):
    """Load model and processor with 4-bit quantization."""
    logger.info(f"Loading model from {model_name_or_path}")

    processor = Qwen3OmniMoeProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )

    quantization_config = get_bnb_config() if use_4bit else None

    # Use Thinker model for text-only training (has proper forward signature)
    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        model_name_or_path,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if use_flash_attention else None,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Enable input gradients for LoRA training (required for PEFT)
    model.enable_input_require_grads()

    logger.info("Model loaded successfully")
    return model, processor


class ReSTrainDataset(Dataset):
    """Dataset for ReST training with filtered correct solutions."""

    def __init__(
        self,
        data_path: str,
        processor,
        audio_dir: str,
        max_length: int = 4096,
        max_samples: int = None,
    ):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        # Limit samples for testing
        if max_samples is not None and max_samples > 0:
            self.data = self.data[:max_samples]
            logger.info(f"Limited to {max_samples} samples for testing")

        self.processor = processor
        self.audio_dir = audio_dir
        self.max_length = max_length

        logger.info(f"Loaded {len(self.data)} training samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, Any]:
        sample = self.data[idx]

        # Get audio path - support per-sample audio_dir for combined datasets
        sample_audio_dir = sample.get("audio_dir", "")
        sound_file = sample["sound"]

        if sample_audio_dir:
            audio_path = os.path.join(self.audio_dir, sample_audio_dir, sound_file)
        else:
            # Infer audio_dir from filename pattern if not specified
            if sound_file.startswith("concat_"):
                # CountingQA dataset
                audio_path = os.path.join(self.audio_dir, "countingqa_audioskills/counting_audios", sound_file)
            else:
                # MusicBench dataset (YouTube IDs)
                audio_path = os.path.join(self.audio_dir, "musicbench_audioskills/audio", sound_file)

        # Get question and response (the correct solution)
        question = sample.get("question", "").replace("<sound>", "").strip()
        response = sample.get("response", "")

        # Build conversation for training - ONLY system and user messages
        # IMPORTANT: We manually append the assistant response to preserve <think> tags.
        # The Qwen3-Omni chat template has a bug where it strips <think> tags when
        # the user message uses array format (required for multimodal content).
        # See: docs/THINKING_TAGS_BUG.md for full explanation.
        conversation_without_assistant = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant that analyzes audio carefully. Think step by step before giving your final answer."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": question},
                ],
            },
        ]

        # Full conversation needed for process_mm_info (extracts audio from user message)
        full_conversation = conversation_without_assistant + [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response}],
            },
        ]

        # Process multimedia
        audios, images, videos = process_mm_info(
            full_conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO
        )

        # Apply chat template for system + user only, with add_generation_prompt=True
        # This gives us: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
        text = self.processor.apply_chat_template(
            conversation_without_assistant, tokenize=False, add_generation_prompt=True
        )

        # Manually append the assistant response WITH thinking tags preserved
        # The response contains: <think>...</think>\n\nANSWER: xxx
        text = text + response + "<|im_end|>\n"

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

        # Labels for causal LM - mask everything except assistant response
        labels = inputs["input_ids"].clone()

        # Find where assistant response starts by looking for the assistant marker
        # The chat template includes "<|im_start|>assistant\n" before the response
        input_ids = inputs["input_ids"]

        # Get the tokenized assistant marker
        assistant_marker = self.processor.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        marker_len = len(assistant_marker)

        # Find the position of the assistant marker
        assistant_start = -1
        for i in range(len(input_ids) - marker_len + 1):
            if input_ids[i:i+marker_len].tolist() == assistant_marker:
                # Found the marker, response starts after it
                assistant_start = i + marker_len
                break

        # Mask all tokens before assistant response with -100 (ignore in loss)
        if assistant_start > 0:
            labels[:assistant_start] = -100

        # Also mask padding tokens
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        inputs["labels"] = labels

        return inputs


class AudioDataCollator:
    """Data collator for audio datasets."""

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {}
        for key in features[0].keys():
            values = [f[key] for f in features]
            if isinstance(values[0], torch.Tensor):
                batch[key] = torch.stack(values)
            else:
                batch[key] = values
        return batch


class DetailedLoggingCallback(TrainerCallback):
    """Custom callback for detailed training progress logging."""

    def __init__(self, total_steps: int, log_every: int = 1):
        self.total_steps = total_steps
        self.log_every = log_every
        self.start_time = None
        self.step_times = []

    def on_train_begin(self, args, state, control, **kwargs):
        import time
        self.start_time = time.time()
        logger.info(f"="*60)
        logger.info(f"TRAINING STARTED")
        logger.info(f"Total steps: {self.total_steps}")
        logger.info(f"Epochs: {args.num_train_epochs}")
        logger.info(f"Batch size: {args.per_device_train_batch_size}")
        logger.info(f"Gradient accumulation: {args.gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
        logger.info(f"Learning rate: {args.learning_rate}")
        logger.info(f"="*60)

    def on_step_end(self, args, state, control, **kwargs):
        import time
        current_step = state.global_step

        if current_step % self.log_every == 0:
            # Calculate progress
            progress = (current_step / self.total_steps) * 100 if self.total_steps > 0 else 0

            # Calculate time estimates
            elapsed = time.time() - self.start_time
            if current_step > 0:
                time_per_step = elapsed / current_step
                remaining_steps = self.total_steps - current_step
                eta = remaining_steps * time_per_step
                eta_str = f"{eta/60:.1f}min" if eta > 60 else f"{eta:.0f}s"
            else:
                eta_str = "calculating..."

            # Get current loss from logs
            loss = state.log_history[-1].get('loss', 'N/A') if state.log_history else 'N/A'
            if isinstance(loss, float):
                loss = f"{loss:.4f}"

            logger.info(
                f"Step {current_step}/{self.total_steps} ({progress:.1f}%) | "
                f"Loss: {loss} | "
                f"Epoch: {state.epoch:.2f} | "
                f"ETA: {eta_str}"
            )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Log additional metrics
            metrics_to_show = ['loss', 'learning_rate', 'grad_norm']
            metrics_str = " | ".join(
                f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in logs.items() if k in metrics_to_show
            )
            if metrics_str:
                logger.info(f"Metrics: {metrics_str}")

    def on_epoch_end(self, args, state, control, **kwargs):
        import time
        elapsed = time.time() - self.start_time
        logger.info(f"="*40)
        logger.info(f"Epoch {int(state.epoch)} completed | Total time: {elapsed/60:.1f}min")
        logger.info(f"Saving checkpoint at end of epoch {int(state.epoch)}...")
        logger.info(f"="*40)
        # Trigger save at end of each epoch
        control.should_save = True
        return control

    def on_train_end(self, args, state, control, **kwargs):
        import time
        total_time = time.time() - self.start_time
        logger.info(f"="*60)
        logger.info(f"TRAINING COMPLETED")
        logger.info(f"Total steps: {state.global_step}")
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"Final loss: {state.log_history[-1].get('train_loss', 'N/A') if state.log_history else 'N/A'}")
        logger.info(f"="*60)


def train(
    model_name_or_path: str,
    train_data_path: str,
    audio_dir: str,
    output_dir: str,
    eval_data_path: Optional[str] = None,
    # Model settings
    use_4bit: bool = True,
    use_flash_attention: bool = True,
    # LoRA settings
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    lora_targets: str = "all",
    # Training settings
    num_epochs: int = 2,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    # Saving
    save_steps: int = 500,
    save_total_limit: int = 3,
    models_dir: str = "../../models",
    # Logging
    logging_dir: Optional[str] = None,
    # Testing
    max_samples: int = None,
    # Sequence length
    max_length: int = 4096,
    # Resume
    resume_from_checkpoint: Optional[str] = None,
):
    """Main training function."""

    # Load model and processor
    model, processor = load_model_and_processor(
        model_name_or_path,
        use_4bit=use_4bit,
        use_flash_attention=use_flash_attention,
    )

    # Setup LoRA for QLoRA training
    logger.info("Setting up LoRA...")
    lora_config = get_lora_config(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_targets=lora_targets,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load datasets
    logger.info("Loading training data...")
    train_dataset = ReSTrainDataset(
        data_path=train_data_path,
        processor=processor,
        audio_dir=audio_dir,
        max_samples=max_samples,
    )

    eval_dataset = None
    if eval_data_path:
        eval_dataset = ReSTrainDataset(
            data_path=eval_data_path,
            processor=processor,
            audio_dir=audio_dir,
            max_samples=max_samples,
        )

    data_collator = AudioDataCollator()

    # Set up logging directory for TensorBoard
    if logging_dir is None:
        logging_dir = os.path.join(output_dir, "logs")

    # Calculate total training steps for progress tracking
    num_training_samples = len(train_dataset)
    steps_per_epoch = num_training_samples // (batch_size * gradient_accumulation_steps)
    total_steps = steps_per_epoch * num_epochs
    logger.info(f"Training samples: {num_training_samples}, Steps per epoch: {steps_per_epoch}, Total steps: {total_steps}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        logging_dir=logging_dir,
        logging_steps=1,  # Log every step for detailed tracking
        logging_first_step=True,
        save_strategy="steps",
        save_steps=save_steps,
        save_on_each_node=True,
        eval_steps=save_steps if eval_dataset else None,
        eval_strategy="steps" if eval_dataset else "no",
        save_total_limit=save_total_limit,
        load_best_model_at_end=False,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to="tensorboard",
    )

    logger.info(f"TensorBoard logs will be saved to: {logging_dir}")

    # Create custom callback for detailed logging
    logging_callback = DetailedLoggingCallback(
        total_steps=total_steps,
        log_every=1,  # Log every step
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[logging_callback],
    )

    # Train
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint if resume_from_checkpoint else None)

    # Generate save path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_clean = model_name_or_path.replace("/", "_").replace("-", "_")
    save_name = f"rest_{model_name_clean}_{timestamp}"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    final_save_path = os.path.join(script_dir, models_dir, save_name)
    os.makedirs(final_save_path, exist_ok=True)

    # Save LoRA adapter (don't merge with 4-bit models - causes compatibility issues)
    logger.info(f"Saving LoRA adapter to {final_save_path}")
    model.save_pretrained(final_save_path)
    processor.save_pretrained(final_save_path)

    logger.info(f"Training complete! LoRA adapter saved to {final_save_path}")
    logger.info(f"To use: load base model + adapter with PeftModel.from_pretrained()")

    return final_save_path


def main():
    parser = argparse.ArgumentParser(description="Phase 3: SFT Training")

    # Data settings
    parser.add_argument("--train_data_path", type=str, required=True,
                        help="Path to filtered training data JSON")
    parser.add_argument("--eval_data_path", type=str, default=None,
                        help="Path to evaluation data JSON")
    parser.add_argument("--audio_dir", type=str, required=True,
                        help="Directory containing audio files")
    parser.add_argument("--output_dir", type=str, default="./outputs/rest/training",
                        help="Output directory for checkpoints")
    parser.add_argument("--models_dir", type=str, default="../../models",
                        help="Directory to save final merged model")

    # Model settings
    parser.add_argument("--model_name_or_path", type=str,
                        default="Qwen/Qwen3-Omni-30B-A3B-Thinking",
                        help="Model name or path")
    parser.add_argument("--use_4bit", action="store_true", default=True,
                        help="Use 4-bit quantization")
    parser.add_argument("--use_flash_attention", action="store_true", default=False,
                        help="Use flash attention (requires flash_attn package)")

    # LoRA settings
    parser.add_argument("--lora_r", type=int, default=64,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--lora_targets", type=str, default="all",
                        choices=["all", "attention", "moe"],
                        help="LoRA target modules: 'all' (attention+MoE), 'attention' (q/k/v/o only), 'moe' (gate/up/down only)")

    # Training settings
    parser.add_argument("--num_epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio")

    # Saving
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save every N steps")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Maximum checkpoints to keep")

    # Logging
    parser.add_argument("--logging_dir", type=str, default=None,
                        help="TensorBoard logging directory (default: output_dir/logs)")

    # Testing
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit training to first N samples (for testing)")

    # Resume
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from (e.g., ./outputs/checkpoint-546)")

    args = parser.parse_args()

    train(
        model_name_or_path=args.model_name_or_path,
        train_data_path=args.train_data_path,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        eval_data_path=args.eval_data_path,
        use_4bit=args.use_4bit,
        use_flash_attention=args.use_flash_attention,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=args.lora_targets,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        models_dir=args.models_dir,
        logging_dir=args.logging_dir,
        max_samples=args.max_samples,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )


if __name__ == "__main__":
    main()
