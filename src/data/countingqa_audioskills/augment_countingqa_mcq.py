"""
Data augmentation script for CountingQA dataset.

Converts free-form Q&A into multiple choice questions by generating
plausible distractors using Qwen 2.5 Omni model with audio context.
"""

import argparse
import json
import os
import random
import logging
from typing import Optional, List, Dict
from dataclasses import dataclass

# Conditional imports for model (not needed in dry-run mode)
torch = None
Qwen3OmniMoeForConditionalGeneration = None
Qwen3OmniMoeProcessor = None
process_mm_info = None

def _import_model_deps():
    """Import model dependencies lazily."""
    global torch, Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor, process_mm_info
    if torch is None:
        import torch as _torch
        torch = _torch
        from transformers import Qwen3OmniMoeForConditionalGeneration as _Model
        from transformers import Qwen3OmniMoeProcessor as _Processor
        from qwen_omni_utils import process_mm_info as _pmm
        Qwen3OmniMoeForConditionalGeneration = _Model
        Qwen3OmniMoeProcessor = _Processor
        process_mm_info = _pmm

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

USE_AUDIO_IN_VIDEO = True


@dataclass
class AugmentConfig:
    input_json: str
    output_json: str
    audio_dir: str
    model_name_or_path: str = "Qwen/Qwen3-Omni-30B"
    batch_size: int = 1
    start_idx: int = 0
    end_idx: Optional[int] = None
    save_every: int = 100
    resume: bool = False
    flash_attention: bool = False
    max_new_tokens: int = 256
    num_distractors: int = 3


def load_model(model_name_or_path: str, flash_attention: bool = False):
    """Load Qwen3-Omni-MoE (thinking model) and processor."""
    _import_model_deps()

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


def create_distractor_prompt(question: str, correct_answer: str) -> str:
    """Create a prompt for generating short correct answer and plausible distractors."""
    return f"""You are helping create a multiple choice question for an audio understanding task.

Original question: {question}
Correct answer: {correct_answer}

Your task:
1. Convert the correct answer to a SHORT form (2-5 words max)
2. Generate exactly 3 plausible but INCORRECT short alternatives

IMPORTANT - Keep ALL options SHORT and concise:
- For counting questions: just use "Two times", "Three times", etc. (NOT full sentences)
- For "which sound" questions: just use the sound name like "Dog barking", "Bell ringing" (NOT full sentences)
- For sequence questions: just the key element like "Car horn", "Footsteps"

Examples of GOOD short options:
- "Three times" (for counting)
- "Dog barking" (for sound identification)
- "Bell ringing" (for sequence)

Examples of BAD long options (DO NOT generate these):
- "The bell ringing happens three times in this audio."
- "The last sound in the sequence is dog barking."

Output ONLY a JSON object with this exact format:
{{"correct": "<short correct answer>", "distractors": ["<distractor1>", "<distractor2>", "<distractor3>"]}}

Example output:
{{"correct": "Two times", "distractors": ["Three times", "Five times", "Once"]}}"""


def generate_mcq_options_batch(
    model,
    processor,
    samples: List[Dict],
    audio_dir: str,
    max_new_tokens: int = 256,
) -> List[tuple[str, List[str]]]:
    """Generate short correct answer and distractors for a batch of samples using audio context.

    This performs true batch inference - all samples are processed in a single forward pass.

    Returns:
        List of tuples (short_correct, distractors) for each sample.
        Returns (None, []) for samples that failed.
    """
    _import_model_deps()

    if not samples:
        return []

    # Build conversations for all samples in the batch
    conversations = []
    for sample in samples:
        audio_path = os.path.join(audio_dir, sample["sound"])

        # Extract question and answer from conversations
        question = ""
        correct_answer = ""
        for conv in sample["conversations"]:
            if conv["from"] == "human":
                question = conv["value"].replace("<sound>", "").replace("\n", " ").strip()
            elif conv["from"] == "gpt":
                correct_answer = conv["value"]

        prompt = create_distractor_prompt(question, correct_answer)

        # Prepare conversation with audio context
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are an expert at creating SHORT, concise multiple choice options. Output a JSON object with a short correct answer and 3 short distractors (2-5 words each). Never generate full sentences - only short phrases."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        conversations.append(conversation)

    try:
        # Process multimedia info for all conversations at once
        audios, images, videos = process_mm_info(
            conversations, use_audio_in_video=USE_AUDIO_IN_VIDEO
        )

        # Apply chat template to all conversations
        texts = processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )

        # Process all inputs together
        inputs = processor(
            text=texts,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
        )

        inputs = inputs.to(model.device).to(model.dtype)

        # Batch generate
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                return_audio=False,
                use_audio_in_video=USE_AUDIO_IN_VIDEO,
            )

        # Decode all outputs
        generated_texts = processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        # Parse results for each sample
        all_results = []
        for generated_text in generated_texts:
            short_correct, distractors = parse_model_output(generated_text.strip())
            all_results.append((short_correct, distractors))

        return all_results

    except Exception as e:
        logger.warning(f"Error in batch inference: {e}")
        torch.cuda.empty_cache()
        # Return failure for all samples in batch
        return [(None, []) for _ in samples]


def parse_model_output(generated_text: str) -> tuple[str, List[str]]:
    """Parse short correct answer and distractors from model output.

    Returns:
        tuple of (short_correct, distractors). Returns (None, []) if parsing fails.
    """
    try:
        # Try to find JSON object in the response
        start_idx = generated_text.find("{")
        end_idx = generated_text.rfind("}") + 1

        if start_idx != -1 and end_idx > start_idx:
            json_str = generated_text[start_idx:end_idx]
            parsed = json.loads(json_str)

            if isinstance(parsed, dict):
                short_correct = parsed.get("correct", "").strip()
                distractors_raw = parsed.get("distractors", [])

                # Filter out empty strings and duplicates
                distractors = []
                seen = {short_correct.lower().strip()} if short_correct else set()
                for d in distractors_raw:
                    if isinstance(d, str) and d.strip():
                        d_lower = d.lower().strip()
                        if d_lower not in seen:
                            distractors.append(d.strip())
                            seen.add(d_lower)
                        if len(distractors) >= 3:
                            break

                if short_correct:
                    return short_correct, distractors

    except json.JSONDecodeError:
        logger.warning(f"Failed to parse JSON from model output: {generated_text[:100]}...")

    return None, []


def format_mcq(
    sample: Dict,
    short_correct: str,
    distractors: List[str],
) -> Dict:
    """Format sample as multiple choice question with randomized option order.

    Args:
        sample: Original sample dict
        short_correct: Short form of correct answer from LLM
        distractors: List of distractor options from LLM
    """

    # Extract original question
    original_question = ""
    for conv in sample["conversations"]:
        if conv["from"] == "human":
            original_question = conv["value"]
            break

    # Validate we have short_correct and enough distractors from LLM
    if not short_correct:
        logger.warning(f"Sample {sample['id']}: No short correct answer from LLM - skipping")
        return None

    # Ensure we have exactly 3 unique distractors
    unique_distractors = []
    seen = {short_correct.lower().strip()}
    for d in distractors:
        d_lower = d.lower().strip()
        if d_lower not in seen and d.strip():
            unique_distractors.append(d.strip())
            seen.add(d_lower)
        if len(unique_distractors) >= 3:
            break

    if len(unique_distractors) < 3:
        logger.warning(f"Sample {sample['id']}: Only {len(unique_distractors)} unique distractors from LLM - skipping")
        return None

    # Create options list with short correct answer and distractors
    options = [short_correct] + unique_distractors

    # Verify we have exactly 4 options
    assert len(options) == 4, f"Expected 4 options, got {len(options)}"

    # Shuffle options and track correct answer position
    random.shuffle(options)
    correct_idx = options.index(short_correct)
    option_labels = ["A", "B", "C", "D"]
    correct_label = option_labels[correct_idx]

    # Format options string
    options_str = "\n".join([f"({label}) {opt}" for label, opt in zip(option_labels, options)])

    # Update question to include options
    # Remove <sound> tag temporarily, add options, then add back
    clean_question = original_question.replace("<sound>", "").replace("\n", " ").strip()

    # Determine where <sound> was in the original question
    if original_question.strip().startswith("<sound>"):
        mcq_question = f"<sound>\n{clean_question}\n\nChoose the correct option:\n{options_str}"
    else:
        mcq_question = f"{clean_question}\n\nChoose the correct option:\n{options_str}\n<sound>"

    # Create new sample - preserve original conversations, only add question with options
    mcq_sample = {
        "id": sample["id"],
        "sound": sample["sound"],
        "duration": sample.get("duration", 0),
        "question": mcq_question,
        "answer": short_correct,  # Answer matches the option in the question
        "conversations": sample["conversations"]  # Keep original conversations unchanged
    }

    return mcq_sample


def load_existing_results(output_json: str) -> Dict[str, Dict]:
    """Load existing results for resume functionality."""
    if not os.path.exists(output_json):
        return {}

    try:
        with open(output_json, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            return {item["id"]: item for item in existing_data}
    except (json.JSONDecodeError, KeyError):
        return {}


def save_results(output_json: str, results: List[Dict]):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(results)} samples to {output_json}")


def main():
    parser = argparse.ArgumentParser(description="Augment CountingQA with MCQ distractors")
    parser.add_argument("--input_json", type=str, default="CountingQA.json",
                        help="Path to input CountingQA.json")
    parser.add_argument("--output_json", type=str, default="CountingQA_MCQ.json",
                        help="Path to output augmented JSON")
    parser.add_argument("--audio_dir", type=str, default="counting_audios",
                        help="Directory containing audio files")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-Omni-30B",
                        help="Qwen3-Omni-MoE model path")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for processing")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start index in dataset")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="End index in dataset (exclusive)")
    parser.add_argument("--save_every", type=int, default=100,
                        help="Save checkpoint every N samples")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")
    parser.add_argument("--flash_attention", action="store_true",
                        help="Use flash attention")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum new tokens for generation")
    parser.add_argument("--dry_run", action="store_true",
                        help="Test with fallback distractors only (no model)")

    args = parser.parse_args()

    # Resolve paths relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_json = os.path.join(script_dir, args.input_json) if not os.path.isabs(args.input_json) else args.input_json
    output_json = os.path.join(script_dir, args.output_json) if not os.path.isabs(args.output_json) else args.output_json
    audio_dir = os.path.join(script_dir, args.audio_dir) if not os.path.isabs(args.audio_dir) else args.audio_dir

    # Load input dataset
    logger.info(f"Loading dataset from {input_json}")
    with open(input_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Apply start/end indices
    end_idx = args.end_idx if args.end_idx is not None else len(dataset)
    dataset = dataset[args.start_idx:end_idx]
    logger.info(f"Processing {len(dataset)} samples (indices {args.start_idx} to {end_idx})")

    # Load existing results if resuming
    existing_results = {}
    if args.resume:
        existing_results = load_existing_results(output_json)
        logger.info(f"Loaded {len(existing_results)} existing results for resume")

    # Load model (required - no dry run mode without LLM)
    if args.dry_run:
        logger.error("Dry run mode is disabled. LLM is required for distractor generation.")
        return
    model, processor = load_model(args.model_name_or_path, args.flash_attention)

    # Process samples
    results = list(existing_results.values())
    processed_ids = set(existing_results.keys())

    # Filter out already processed samples
    samples_to_process = [s for s in dataset if s["id"] not in processed_ids]
    logger.info(f"Samples to process: {len(samples_to_process)}")

    skipped_count = 0
    batch_size = args.batch_size
    num_batches = (len(samples_to_process) + batch_size - 1) // batch_size

    logger.info(f"Processing {len(samples_to_process)} samples in {num_batches} batches (batch_size={batch_size})")

    for batch_idx in tqdm(range(num_batches), desc="Augmenting"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(samples_to_process))
        batch_samples = samples_to_process[start_idx:end_idx]

        try:
            # Generate short correct answers and distractors for the entire batch
            batch_results = generate_mcq_options_batch(
                model, processor, batch_samples, audio_dir, args.max_new_tokens
            )

            # Process each result in the batch
            for sample, (short_correct, distractors) in zip(batch_samples, batch_results):
                mcq_sample = format_mcq(sample, short_correct, distractors)
                if mcq_sample is None:
                    skipped_count += 1
                    continue
                results.append(mcq_sample)

            # Save checkpoint
            if (batch_idx + 1) % max(1, args.save_every // batch_size) == 0:
                save_results(output_json, results)

        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {e}")
            skipped_count += len(batch_samples)
            continue

    # Final save
    save_results(output_json, results)
    logger.info(f"Augmentation complete. Total samples: {len(results)}, Skipped: {skipped_count}")


if __name__ == "__main__":
    main()
