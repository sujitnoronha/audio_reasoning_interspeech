"""
Phase 1: Candidate Generation using HuggingFace Transformers.

Alternative to vLLM for when vLLM doesn't support the model.
Generates K candidate solutions per problem using the thinking model.

Usage:
    python generate_hf.py --data_path ../data/countingqa/CountingQA_MCQ.json \
        --audio_dir ../data/countingqa/counting_audios \
        --num_samples 16 --temperature 0.9
"""

import argparse
import json
import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

import torch
from transformers import (
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
)
from qwen_omni_utils import process_mm_info
from tqdm import tqdm

from config import GenerationConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

USE_AUDIO_IN_VIDEO = True


def parse_thinking_and_answer(text: str) -> Tuple[str, str]:
    """Parse thinking content and final answer from model output.

    The model outputs in format: <think>reasoning...</think>final answer

    Handles edge cases:
    - Truncated output (no closing </think> tag)
    - Missing <think> tags entirely

    Args:
        text: Raw model output

    Returns:
        Tuple of (thinking_prediction, answer_prediction)
    """
    thinking = ""
    answer = ""

    # Try to extract thinking content from <think>...</think> tags
    think_pattern = r'<think>(.*?)</think>'
    think_match = re.search(think_pattern, text, re.DOTALL)

    if think_match:
        thinking = think_match.group(1).strip()
        # Answer is everything after </think>
        after_think = text.split('</think>', 1)
        if len(after_think) > 1:
            answer = after_think[1].strip()
        else:
            answer = ""
    else:
        # No complete </think> tag - check if there's an opening <think>
        if '<think>' in text:
            # Truncated output - extract what's after <think> as thinking
            after_open = text.split('<think>', 1)
            if len(after_open) > 1:
                thinking = after_open[1].strip()
            answer = ""  # No answer available if truncated
        else:
            # No thinking tags found - entire text is the answer
            answer = text.strip()

    return thinking, answer


def extract_final_answer(answer_text: str) -> str:
    """Extract the final answer choice text from the answer.

    Looks for patterns like:
    - "ANSWER: Bell ringing" -> "Bell ringing" (preferred format)
    - "The answer is (A) Bell ringing" -> "Bell ringing"
    - "correct answer is (A) Bell ringing" -> "Bell ringing"

    Args:
        answer_text: The answer portion after thinking

    Returns:
        Extracted answer choice text (e.g., "Bell ringing") or empty string if not found
    """
    if not answer_text:
        return ""

    # First, try the structured "ANSWER: <text>" format (most reliable)
    answer_pattern = r'ANSWER:\s*(.+?)(?:\n|$)'
    match = re.search(answer_pattern, answer_text, re.IGNORECASE)
    if match:
        answer = match.group(1).strip().rstrip('.')
        # Remove any leading (A), (B), etc. if present
        answer = re.sub(r'^\([A-D]\)\s*', '', answer)
        if answer:
            return answer

    # Fallback patterns for other formats
    answer_with_text_patterns = [
        # "The answer is (A) Bell ringing" or "answer is (A) Bell ringing"
        r'(?:the\s+)?(?:correct\s+)?answer\s+(?:is|should\s+be)[:\s]*\([A-D]\)\s*([A-Za-z][A-Za-z\s]+)',
        # "Therefore, the correct answer is (A) Bell ringing"
        r'(?:therefore|thus|so)[,\s]+(?:the\s+)?(?:correct\s+)?(?:answer|choice)\s+(?:is|should\s+be)[:\s]*\([A-D]\)\s*([A-Za-z][A-Za-z\s]+)',
        # Just "(A) Bell ringing" near the end
        r'\([A-D]\)\s+([A-Za-z][A-Za-z\s]+?)(?:\.|\s*$)',
    ]

    for pattern in answer_with_text_patterns:
        match = re.search(pattern, answer_text, re.IGNORECASE)
        if match:
            answer_choice = match.group(1).strip().rstrip('.')
            if answer_choice:
                return answer_choice

    # Return empty string if no answer choice found
    return ""


def load_model(model_name_or_path: str, flash_attention: bool = True):
    """Load Qwen3-Omni model and processor."""
    logger.info(f"Loading model from {model_name_or_path}")

    processor = Qwen3OmniMoeProcessor.from_pretrained(model_name_or_path)

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if flash_attention else None,
        device_map="auto",
    )
    model.eval()

    logger.info("Model loaded successfully")
    return model, processor


SYSTEM_PROMPT = """You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.

You are an expert at analyzing audio content and answering questions about what you hear. You have exceptional abilities in:
- Counting sounds, events, and occurrences in audio
- Identifying different sound sources and their characteristics
- Understanding temporal relationships and sequences in audio
- Detecting patterns and anomalies in audio content

When answering questions:
1. Listen to the audio carefully and thoroughly
2. Think step-by-step about what you hear
3. Consider all relevant audio details before answering
4. Provide your reasoning in <think>...</think> tags
5. After </think>, state your final answer in this EXACT format:
   ANSWER: <full answer text from the choices>

Example format:
<think>I hear a dog barking followed by a cat meowing...</think>
ANSWER: Dog barking"""


def build_conversation(sample: Dict, audio_dir: str) -> List[Dict]:
    """Build conversation for a sample."""
    audio_filename = sample.get("sound", "")
    # Support combined dataset format with per-sample audio_dir
    sample_audio_dir = sample.get("audio_dir", "")
    if sample_audio_dir:
        audio_path = os.path.join(audio_dir, sample_audio_dir, audio_filename)
    else:
        audio_path = os.path.join(audio_dir, audio_filename)

    question = sample.get("question", "")
    if not question:
        for conv in sample.get("conversations", []):
            if conv.get("from") == "human":
                question = conv.get("value", "")
                break

    clean_question = question.replace("<sound>", "").strip()

    # Add detailed instructions with structured output format
    enhanced_question = (
        clean_question
        + "\n\nIMPORTANT INSTRUCTIONS:"
        + "\n1. Think step-by-step and analyze the audio carefully in <think>...</think> tags."
        + "\n2. Your answer must be 100% correct - this is a critical evaluation."
        + "\n3. After </think>, you MUST output your final answer in this EXACT format:"
        + "\n   ANSWER: <full answer text from the choices>"
        + "\n\nExample: If the correct choice is '(A) Bell ringing', output:"
        + "\nANSWER: Bell ringing"
    )

    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": enhanced_question},
            ],
        },
    ]

    return conversation


def generate_candidates(
    model,
    processor,
    sample: Dict,
    audio_dir: str,
    num_samples: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> List[Dict]:
    """Generate K candidate solutions for a single sample.

    Returns:
        List of parsed candidate dicts with raw_output, thinking_prediction,
        answer_text, and answer_prediction fields.
    """

    conversation = build_conversation(sample, audio_dir)

    # Process multimedia
    audios, images, videos = process_mm_info(
        conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO
    )

    # Apply chat template
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
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

    inputs = inputs.to(model.device).to(model.dtype)

    candidates = []

    # Generate multiple samples
    with torch.no_grad():
        for _ in range(num_samples):
            try:
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    num_beams=1,
                    return_audio=False,
                    use_audio_in_video=USE_AUDIO_IN_VIDEO,
                )

                # Decode
                raw_output = processor.batch_decode(
                    output_ids[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0].strip()

                # Parse thinking and answer
                thinking, answer_text = parse_thinking_and_answer(raw_output)
                final_answer = extract_final_answer(answer_text)

                candidates.append({
                    "raw_output": raw_output,
                    "thinking_prediction": thinking,
                    "answer_text": answer_text,
                    "answer_prediction": final_answer,
                })

            except Exception as e:
                logger.warning(f"Generation failed: {e}")
                continue

    return candidates


def load_dataset(data_path: str, start_idx: int = 0, end_idx: Optional[int] = None) -> List[Dict]:
    """Load dataset from JSON file."""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if end_idx is None:
        end_idx = len(data)

    return data[start_idx:end_idx]


def load_existing_generations(output_path: str) -> Dict[str, Dict]:
    """Load existing generations for resume."""
    if not os.path.exists(output_path):
        return {}

    existing = {}
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            existing[record["id"]] = record

    return existing


def save_generation(output_path: str, record: Dict):
    """Append a single generation record to JSONL file."""
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Generate candidates using HuggingFace")

    # Data settings
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to dataset JSON file")
    parser.add_argument("--audio_dir", type=str, required=True,
                        help="Directory containing audio files")
    parser.add_argument("--output_dir", type=str, default="./outputs/rest/generations",
                        help="Output directory for generations")

    # Model settings
    parser.add_argument("--model_name_or_path", type=str,
                        default="Qwen/Qwen3-Omni-30B-A3B-Thinking",
                        help="Model name or path")
    parser.add_argument("--flash_attention", action="store_true", default=True,
                        help="Use flash attention")

    # Generation settings
    parser.add_argument("--num_samples", type=int, default=16,
                        help="Number of samples per problem (K)")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p sampling")
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="Maximum tokens to generate")

    # Processing settings
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start index in dataset")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="End index in dataset")

    # Resume
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing generations")

    args = parser.parse_args()

    # Setup output
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "generations.jsonl")

    # Load dataset
    logger.info(f"Loading dataset from {args.data_path}")
    dataset = load_dataset(args.data_path, args.start_idx, args.end_idx)
    logger.info(f"Loaded {len(dataset)} samples")

    # Load existing if resuming
    existing = {}
    if args.resume:
        existing = load_existing_generations(output_path)
        logger.info(f"Resuming: found {len(existing)} existing generations")

    # Filter already processed
    samples_to_process = [s for s in dataset if s.get("id") not in existing]
    logger.info(f"Samples to process: {len(samples_to_process)}")

    if not samples_to_process:
        logger.info("No samples to process!")
        return

    # Load model
    model, processor = load_model(args.model_name_or_path, args.flash_attention)

    # Process samples
    total_candidates = 0

    for sample in tqdm(samples_to_process, desc="Generating"):
        try:
            candidates = generate_candidates(
                model=model,
                processor=processor,
                sample=sample,
                audio_dir=args.audio_dir,
                num_samples=args.num_samples,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )

            result = {
                "id": sample.get("id", ""),
                "sound": sample.get("sound", ""),
                "question": sample.get("question", ""),
                "ground_truth": sample.get("answer", ""),
                "candidates": candidates,
                "num_candidates": len(candidates),
            }

            save_generation(output_path, result)
            total_candidates += len(candidates)

            # Clear cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error processing sample {sample.get('id')}: {e}")
            continue

    logger.info(f"Generation complete!")
    logger.info(f"Total samples: {len(samples_to_process)}")
    logger.info(f"Total candidates: {total_candidates}")
    logger.info(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
