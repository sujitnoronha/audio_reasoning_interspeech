"""
Training module for audio reasoning models.

Provides training scripts and utilities for:
- ReST (Reinforced Self-Training)
- GRPO (Group Relative Policy Optimization)

All training uses:
- 4-bit quantization (BitsAndBytes)
- LoRA PEFT for efficient finetuning
- Model merging for final deployment
"""

from .utils import (
    get_bnb_config,
    get_lora_config,
    load_model_and_processor,
    setup_peft_model,
    merge_and_save_model,
    get_model_save_path,
)

from .data_utils import (
    AudioReasoningDataset,
    PromptOnlyDataset,
    AudioDataCollator,
    load_dataset_splits,
    create_dataloader,
    filter_samples_by_audio,
)

__all__ = [
    # Model utilities
    "get_bnb_config",
    "get_lora_config",
    "load_model_and_processor",
    "setup_peft_model",
    "merge_and_save_model",
    "get_model_save_path",
    # Data utilities
    "AudioReasoningDataset",
    "PromptOnlyDataset",
    "AudioDataCollator",
    "load_dataset_splits",
    "create_dataloader",
    "filter_samples_by_audio",
]
