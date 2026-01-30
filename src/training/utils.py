"""
Common utilities for training scripts.

Includes model loading with 4-bit quantization, LoRA configuration,
and model merging utilities.

Uses Qwen3-Omni-MoE (thinking model) for training.
"""

import os
from datetime import datetime
from typing import Optional

import torch
from transformers import (
    BitsAndBytesConfig,
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,
)


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
    target_modules: Optional[list] = None,
) -> LoraConfig:
    """Get LoRA configuration for PEFT.

    Args:
        r: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: Dropout probability for LoRA layers
        target_modules: List of module names to apply LoRA to.
                       If None, uses default modules for Qwen3-Omni-MoE models.
    """
    if target_modules is None:
        # Default target modules for Qwen3-Omni-MoE (thinking model)
        # Includes attention projections and MoE expert layers
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            # MoE expert layers
            "gate_proj",
            "up_proj",
            "down_proj",
            # Shared expert (if present)
            "shared_expert.gate_proj",
            "shared_expert.up_proj",
            "shared_expert.down_proj",
        ]

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_model_and_processor(
    model_name_or_path: str,
    use_4bit: bool = True,
    use_flash_attention: bool = False,
):
    """Load Qwen3-Omni-MoE model and processor with optional 4-bit quantization.

    Args:
        model_name_or_path: Path to model or HuggingFace model ID
        use_4bit: Whether to load model in 4-bit quantization
        use_flash_attention: Whether to use flash attention

    Returns:
        Tuple of (model, processor)
    """
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_name_or_path)

    # Configure quantization if needed
    quantization_config = get_bnb_config() if use_4bit else None

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_name_or_path,
        quantization_config=quantization_config,
        torch_dtype="auto" if not use_4bit else None,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if use_flash_attention else None,
        device_map="auto",
    )

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    return model, processor


def setup_peft_model(
    model,
    lora_config: Optional[LoraConfig] = None,
):
    """Setup PEFT model with LoRA.

    Args:
        model: Base model
        lora_config: LoRA configuration. If None, uses default config.

    Returns:
        PEFT model with LoRA adapters
    """
    if lora_config is None:
        lora_config = get_lora_config()

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def merge_and_save_model(
    model: PeftModel,
    processor,
    output_dir: str,
    training_style: str,
    model_name: str,
):
    """Merge LoRA adapters into base model and save.

    Args:
        model: PEFT model with trained LoRA adapters
        processor: Model processor/tokenizer
        output_dir: Base output directory (e.g., 'models/')
        training_style: Training method name (e.g., 'rest', 'grpo')
        model_name: Base model name

    Returns:
        Path to saved merged model
    """
    # Create output path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_clean = model_name.replace("/", "_").replace("-", "_")
    save_name = f"{training_style}_{model_name_clean}_{timestamp}"
    save_path = os.path.join(output_dir, save_name)

    print(f"Merging LoRA adapters into base model...")

    # Merge LoRA weights into base model
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to {save_path}...")

    # Save merged model and processor
    merged_model.save_pretrained(save_path)
    processor.save_pretrained(save_path)

    print(f"Model saved successfully to {save_path}")

    return save_path


def get_model_save_path(
    output_dir: str,
    training_style: str,
    model_name: str,
) -> str:
    """Generate model save path with timestamp.

    Args:
        output_dir: Base output directory
        training_style: Training method name
        model_name: Base model name

    Returns:
        Full path for saving model
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_clean = model_name.replace("/", "_").replace("-", "_")
    save_name = f"{training_style}_{model_name_clean}_{timestamp}"
    return os.path.join(output_dir, save_name)
