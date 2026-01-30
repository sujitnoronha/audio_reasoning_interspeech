#!/usr/bin/env python3
"""
Evaluation script to compare baseline vs fine-tuned model on test data.

Usage:
    # Evaluate baseline only
    python evaluate.py \
        --base_model /path/to/base/model \
        --test_data /path/to/test.json \
        --audio_dir /path/to/audio \
        --output_dir ./eval_results

    # Evaluate fine-tuned model
    python evaluate.py \
        --base_model /path/to/base/model \
        --adapter_path /path/to/adapter \
        --test_data /path/to/test.json \
        --audio_dir /path/to/audio \
        --output_dir ./eval_results

    # Evaluate both for comparison
    python evaluate.py \
        --base_model /path/to/base/model \
        --adapter_path /path/to/adapter \
        --test_data /path/to/test.json \
        --audio_dir /path/to/audio \
        --output_dir ./eval_results \
        --evaluate_both
"""

import argparse
import json
import os
import re
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from tqdm import tqdm

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


def load_model(
    base_model_path: str,
    adapter_path: Optional[str] = None,
    use_4bit: bool = True,
    use_flash_attention: bool = True,
):
    """Load model with optional LoRA adapter."""
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

    if adapter_path:
        logger.info(f"Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        logger.info("Adapter loaded successfully")
    else:
        logger.info("Using base model without adapter")

    model.eval()
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

    audios, images, videos = process_mm_info(
        conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO
    )

    text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )

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


def prepare_batch_inputs(processor, batch_data: List[Dict]):
    """Prepare batched inputs for the model."""
    all_texts = []
    all_audios = []
    all_images = []
    all_videos = []

    for item in batch_data:
        audio_path = item["audio_path"]
        question = item["question"]

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

        audios, images, videos = process_mm_info(
            conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO
        )

        text = processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

        all_texts.append(text)
        all_audios.append(audios[0] if audios else None)
        all_images.extend(images if images else [])
        all_videos.extend(videos if videos else [])

    # Process batch
    inputs = processor(
        text=all_texts,
        audio=all_audios,
        images=all_images if all_images else None,
        videos=all_videos if all_videos else None,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
    )

    return inputs


def extract_answer(response: str, options: List[str]) -> Tuple[str, str]:
    """
    Extract the answer from model response.
    Returns (extracted_letter, extracted_answer_text)
    """
    response_lower = response.lower().strip()

    # Try to find explicit option letter like (A), (B), etc.
    letter_match = re.search(r'\(([A-D])\)', response)
    if letter_match:
        letter = letter_match.group(1).upper()
        return letter, f"({letter})"

    # Try to find standalone letter at beginning
    letter_match = re.search(r'^([A-D])[\.\:\s]', response.strip())
    if letter_match:
        letter = letter_match.group(1).upper()
        return letter, f"({letter})"

    # Try to match the answer text directly
    for i, opt in enumerate(options):
        opt_clean = opt.lower().strip()
        if opt_clean in response_lower:
            letter = chr(ord('A') + i)
            return letter, opt

    # Check for common answer patterns
    answer_patterns = [
        r'the answer is[:\s]+\(?([A-D])\)?',
        r'correct answer is[:\s]+\(?([A-D])\)?',
        r'answer[:\s]+\(?([A-D])\)?',
        r'option[:\s]+\(?([A-D])\)?',
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, response_lower)
        if match:
            letter = match.group(1).upper()
            return letter, f"({letter})"

    return "", response[:100]  # Return first 100 chars if no match


def parse_options_from_question(question: str) -> List[str]:
    """Parse answer options from question text."""
    options = []
    pattern = r'\(([A-D])\)\s*([^\n\(]+)'
    matches = re.findall(pattern, question)
    for letter, text in matches:
        options.append(text.strip())
    return options


def get_correct_letter(answer: str, question: str) -> str:
    """Get the correct option letter from the answer."""
    # Check if answer is already a letter format like "(A)"
    match = re.search(r'\(([A-D])\)', answer)
    if match:
        return match.group(1)

    # Otherwise find which option matches the answer text
    options = parse_options_from_question(question)
    answer_lower = answer.lower().strip()
    for i, opt in enumerate(options):
        if opt.lower().strip() == answer_lower or answer_lower in opt.lower():
            return chr(ord('A') + i)

    return ""


@torch.no_grad()
def generate_response(
    model,
    processor,
    audio_path: str,
    question: str,
    max_new_tokens: int = 512,
    temperature: float = 0.1,  # Low temperature for more deterministic answers
) -> str:
    """Generate response for an audio question."""
    inputs = prepare_inputs(processor, audio_path, question)

    # Move to model device and convert dtypes
    processed_inputs = {}
    for k, v in inputs.items():
        if v.dtype == torch.float32:
            processed_inputs[k] = v.to(device=model.device, dtype=torch.bfloat16)
        else:
            processed_inputs[k] = v.to(model.device)
    inputs = processed_inputs

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )

    input_len = inputs["input_ids"].shape[1]
    response = processor.tokenizer.decode(
        outputs[0][input_len:],
        skip_special_tokens=True
    )

    return response


@torch.no_grad()
def generate_batch_responses(
    model,
    processor,
    batch_data: List[Dict],
    max_new_tokens: int = 512,
    temperature: float = 0.1,
) -> List[str]:
    """Generate responses for a batch of audio questions."""
    inputs = prepare_batch_inputs(processor, batch_data)

    # Move to model device and convert dtypes
    processed_inputs = {}
    for k, v in inputs.items():
        if v.dtype == torch.float32:
            processed_inputs[k] = v.to(device=model.device, dtype=torch.bfloat16)
        else:
            processed_inputs[k] = v.to(model.device)
    inputs = processed_inputs

    # Get input lengths for each sample in batch
    input_lens = (inputs["input_ids"] != processor.tokenizer.pad_token_id).sum(dim=1)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )

    # Decode each response
    responses = []
    for i in range(len(batch_data)):
        input_len = input_lens[i].item()
        response = processor.tokenizer.decode(
            outputs[i][input_len:],
            skip_special_tokens=True
        )
        responses.append(response)

    return responses


def evaluate_model(
    model,
    processor,
    test_data: List[Dict],
    audio_dir: str,
    model_name: str = "model",
    max_samples: Optional[int] = None,
    batch_size: int = 1,
) -> Dict[str, Any]:
    """Evaluate model on test data."""
    if max_samples:
        test_data = test_data[:max_samples]

    results = []
    correct = 0
    total = 0
    errors = 0

    # Track per-dataset accuracy
    dataset_stats = {}

    logger.info(f"Evaluating {model_name} on {len(test_data)} samples (batch_size={batch_size})...")

    # Prepare all samples with audio paths
    prepared_samples = []
    for sample in test_data:
        sample_id = sample.get("id", "")
        sound_file = sample.get("sound", "")
        question = sample.get("question", "").replace("<sound>", "").strip()
        answer = sample.get("answer", "")
        dataset = sample.get("dataset", "unknown")
        sample_audio_dir = sample.get("audio_dir", "")

        # Construct audio path
        if sample_audio_dir:
            audio_path = os.path.join(audio_dir, sample_audio_dir, sound_file)
        elif sound_file.startswith("concat_"):
            audio_path = os.path.join(audio_dir, "countingqa_audioskills/counting_audios", sound_file)
        else:
            audio_path = os.path.join(audio_dir, "musicbench_audioskills/audio", sound_file)

        # Check if audio exists
        if not os.path.exists(audio_path):
            logger.warning(f"Audio not found: {audio_path}")
            errors += 1
            continue

        prepared_samples.append({
            "sample_id": sample_id,
            "audio_path": audio_path,
            "question": question,
            "answer": answer,
            "dataset": dataset,
            "original_question": sample.get("question", ""),
        })

    # Process in batches
    num_batches = (len(prepared_samples) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc=f"Evaluating {model_name}"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(prepared_samples))
        batch = prepared_samples[start_idx:end_idx]

        try:
            if batch_size == 1:
                # Single sample - use original function
                item = batch[0]
                responses = [generate_response(model, processor, item["audio_path"], item["question"])]
            else:
                # Batch processing
                responses = generate_batch_responses(model, processor, batch)

            # Process responses
            for i, item in enumerate(batch):
                response = responses[i]
                options = parse_options_from_question(item["original_question"])
                predicted_letter, predicted_text = extract_answer(response, options)
                correct_letter = get_correct_letter(item["answer"], item["original_question"])

                is_correct = predicted_letter == correct_letter

                if is_correct:
                    correct += 1

                total += 1

                # Track per-dataset stats
                dataset = item["dataset"]
                if dataset not in dataset_stats:
                    dataset_stats[dataset] = {"correct": 0, "total": 0}
                dataset_stats[dataset]["total"] += 1
                if is_correct:
                    dataset_stats[dataset]["correct"] += 1

                results.append({
                    "id": item["sample_id"],
                    "dataset": dataset,
                    "question": item["question"][:200],
                    "ground_truth": item["answer"],
                    "correct_letter": correct_letter,
                    "predicted_letter": predicted_letter,
                    "predicted_text": predicted_text,
                    "full_response": response[:500],
                    "is_correct": is_correct,
                })

        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {e}")
            errors += len(batch)
            continue

    # Calculate overall accuracy
    accuracy = correct / total if total > 0 else 0

    # Calculate per-dataset accuracy
    for ds in dataset_stats:
        ds_acc = dataset_stats[ds]["correct"] / dataset_stats[ds]["total"] if dataset_stats[ds]["total"] > 0 else 0
        dataset_stats[ds]["accuracy"] = ds_acc

    summary = {
        "model_name": model_name,
        "total_samples": total,
        "correct": correct,
        "accuracy": accuracy,
        "errors": errors,
        "dataset_stats": dataset_stats,
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"{model_name} Results:")
    logger.info(f"  Overall Accuracy: {accuracy:.2%} ({correct}/{total})")
    for ds, stats in dataset_stats.items():
        logger.info(f"  {ds}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    logger.info(f"  Errors: {errors}")
    logger.info(f"{'='*60}\n")

    return {"summary": summary, "results": results}


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline vs fine-tuned model")

    parser.add_argument("--base_model", type=str, required=True,
                        help="Path to base model")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to LoRA adapter (for fine-tuned model)")
    parser.add_argument("--test_data", type=str, required=True,
                        help="Path to test data JSON")
    parser.add_argument("--audio_dir", type=str, required=True,
                        help="Base directory containing audio files")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="Output directory for results")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum samples to evaluate (for testing)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for evaluation (default: 4)")
    parser.add_argument("--evaluate_both", action="store_true",
                        help="Evaluate both baseline and fine-tuned models")
    parser.add_argument("--use_4bit", action="store_true", default=True,
                        help="Use 4-bit quantization")
    parser.add_argument("--use_flash_attention", action="store_true", default=True,
                        help="Use flash attention for faster inference")

    args = parser.parse_args()

    # Load test data
    logger.info(f"Loading test data from {args.test_data}")
    with open(args.test_data, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    logger.info(f"Loaded {len(test_data)} test samples")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = {}

    if args.evaluate_both:
        # Evaluate baseline first
        logger.info("\n" + "="*60)
        logger.info("EVALUATING BASELINE MODEL")
        logger.info("="*60)

        model, processor = load_model(
            args.base_model,
            adapter_path=None,
            use_4bit=args.use_4bit,
            use_flash_attention=args.use_flash_attention,
        )

        baseline_results = evaluate_model(
            model, processor, test_data, args.audio_dir,
            model_name="baseline",
            max_samples=args.max_samples,
            batch_size=args.batch_size,
        )
        all_results["baseline"] = baseline_results

        # Clear memory
        del model
        torch.cuda.empty_cache()

        # Evaluate fine-tuned
        if args.adapter_path:
            logger.info("\n" + "="*60)
            logger.info("EVALUATING FINE-TUNED MODEL")
            logger.info("="*60)

            model, processor = load_model(
                args.base_model,
                adapter_path=args.adapter_path,
                use_4bit=args.use_4bit,
                use_flash_attention=args.use_flash_attention,
            )

            finetuned_results = evaluate_model(
                model, processor, test_data, args.audio_dir,
                model_name="finetuned",
                max_samples=args.max_samples,
                batch_size=args.batch_size,
            )
            all_results["finetuned"] = finetuned_results

            # Print comparison
            logger.info("\n" + "="*60)
            logger.info("COMPARISON SUMMARY")
            logger.info("="*60)
            baseline_acc = baseline_results["summary"]["accuracy"]
            finetuned_acc = finetuned_results["summary"]["accuracy"]
            improvement = finetuned_acc - baseline_acc
            logger.info(f"Baseline Accuracy:   {baseline_acc:.2%}")
            logger.info(f"Fine-tuned Accuracy: {finetuned_acc:.2%}")
            logger.info(f"Improvement:         {improvement:+.2%}")
            logger.info("="*60)

    else:
        # Evaluate single model (baseline or fine-tuned)
        model_name = "finetuned" if args.adapter_path else "baseline"

        model, processor = load_model(
            args.base_model,
            adapter_path=args.adapter_path,
            use_4bit=args.use_4bit,
            use_flash_attention=args.use_flash_attention,
        )

        results = evaluate_model(
            model, processor, test_data, args.audio_dir,
            model_name=model_name,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
        )
        all_results[model_name] = results

    # Save results
    output_file = os.path.join(args.output_dir, f"eval_results_{timestamp}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
