#!/usr/bin/env python3
"""
Inference script for ReST-trained models with LoRA adapters.

Usage:
    python inference.py \
        --base_model /path/to/base/model \
        --adapter_path /path/to/lora/adapter \
        --audio_path /path/to/audio.wav \
        --question "What instrument is playing?"
"""

import argparse
import logging
import torch
from transformers import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel
from qwen_omni_utils import process_mm_info

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


def load_model_with_adapter(
    base_model_path: str,
    adapter_path: str,
    use_4bit: bool = True,
    use_flash_attention: bool = False,
):
    """Load base model with LoRA adapter."""
    logger.info(f"Loading base model from {base_model_path}")

    processor = Qwen3OmniMoeProcessor.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )

    quantization_config = get_bnb_config() if use_4bit else None

    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        base_model_path,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if use_flash_attention else None,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    logger.info(f"Loading LoRA adapter from {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    logger.info("Model loaded successfully")
    return model, processor


def prepare_inputs(processor, audio_path: str, question: str):
    """Prepare inputs for the model."""
    conversation = [
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

    # Process multimedia
    audios, images, videos = process_mm_info(
        conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO
    )

    # Apply chat template
    text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )

    # Process inputs
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
    )

    return inputs


@torch.no_grad()
def generate_response(
    model,
    processor,
    audio_path: str,
    question: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """Generate response for an audio question."""
    inputs = prepare_inputs(processor, audio_path, question)

    # Move to model device and convert float tensors to bfloat16
    processed_inputs = {}
    for k, v in inputs.items():
        if v.dtype == torch.float32:
            processed_inputs[k] = v.to(device=model.device, dtype=torch.bfloat16)
        else:
            processed_inputs[k] = v.to(model.device)
    inputs = processed_inputs

    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )

    # Decode response (skip input tokens)
    input_len = inputs["input_ids"].shape[1]
    response = processor.tokenizer.decode(
        outputs[0][input_len:],
        skip_special_tokens=True
    )

    return response


def main():
    parser = argparse.ArgumentParser(description="Inference with LoRA adapter")

    parser.add_argument("--base_model", type=str, required=True,
                        help="Path to base model")
    parser.add_argument("--adapter_path", type=str, required=True,
                        help="Path to LoRA adapter")
    parser.add_argument("--audio_path", type=str, required=True,
                        help="Path to audio file")
    parser.add_argument("--question", type=str, required=True,
                        help="Question about the audio")
    parser.add_argument("--use_4bit", action="store_true", default=True,
                        help="Use 4-bit quantization")
    parser.add_argument("--use_flash_attention", action="store_true", default=False,
                        help="Use flash attention")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")

    args = parser.parse_args()

    # Load model
    model, processor = load_model_with_adapter(
        base_model_path=args.base_model,
        adapter_path=args.adapter_path,
        use_4bit=args.use_4bit,
        use_flash_attention=args.use_flash_attention,
    )

    # Generate response
    logger.info(f"Processing audio: {args.audio_path}")
    logger.info(f"Question: {args.question}")

    response = generate_response(
        model=model,
        processor=processor,
        audio_path=args.audio_path,
        question=args.question,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print("\n" + "="*60)
    print("RESPONSE:")
    print("="*60)
    print(response)
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
