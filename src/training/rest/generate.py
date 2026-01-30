"""
Phase 1: Candidate Generation using vLLM serve.

Generates K candidate solutions per problem using the thinking model.
Uses vLLM serve API for fast parallel inference.

Usage:
    # First start vLLM server:
    vllm serve Qwen/Qwen3-Omni-30B-A3B-Thinking --port 8901 --host 127.0.0.1 \
        --dtype bfloat16 --max-model-len 32768 --allowed-local-media-path / -tp 4

    # Then run generation:
    python generate.py --data_path ../data/countingqa/CountingQA_MCQ.json \
        --audio_dir ../data/countingqa/counting_audios \
        --num_samples 16 --temperature 0.9
"""

import argparse
import asyncio
import json
import os
import re
import logging
from dataclasses import asdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import aiohttp
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

from config import GenerationConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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


class VLLMClient:
    """Async client for vLLM serve API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8901/v1",
        model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Thinking",
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.chat_endpoint = f"{self.base_url}/chat/completions"

    async def generate(
        self,
        messages: List[Dict],
        temperature: float = 0.9,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        n: int = 1,
        session: aiohttp.ClientSession = None,
    ) -> List[str]:
        """Generate completions using vLLM chat API.

        Args:
            messages: Chat messages in OpenAI format
            temperature: Sampling temperature
            top_p: Top-p sampling
            max_tokens: Maximum tokens to generate
            n: Number of completions to generate
            session: aiohttp session to reuse

        Returns:
            List of generated completions
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "n": n,
        }

        close_session = False
        if session is None:
            session = aiohttp.ClientSession()
            close_session = True

        try:
            async with session.post(
                self.chat_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"API error: {response.status} - {error_text}")
                    return []

                result = await response.json()
                completions = [
                    choice["message"]["content"]
                    for choice in result.get("choices", [])
                ]
                return completions

        except Exception as e:
            logger.error(f"Request failed: {e}")
            return []

        finally:
            if close_session:
                await session.close()


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


def build_messages(sample: Dict, audio_dir: str) -> List[Dict]:
    """Build chat messages for a sample.

    Args:
        sample: Sample dict with question, answer, sound fields
        audio_dir: Base directory for audio files (or full path if sample has no audio_dir field)

    Returns:
        Messages in vLLM chat format
    """
    # Get audio path (use file:// for local files)
    audio_filename = sample.get("sound", "")
    # Support combined dataset format with per-sample audio_dir
    sample_audio_dir = sample.get("audio_dir", "")
    if sample_audio_dir:
        audio_path = os.path.abspath(os.path.join(audio_dir, sample_audio_dir, audio_filename))
    else:
        audio_path = os.path.abspath(os.path.join(audio_dir, audio_filename))

    # Get question
    question = sample.get("question", "")
    if not question:
        # Extract from conversations
        for conv in sample.get("conversations", []):
            if conv.get("from") == "human":
                question = conv.get("value", "")
                break

    # Clean question (remove <sound> tag as we'll pass audio separately)
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

    # Build messages with audio
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": f"file://{audio_path}"}},
                {"type": "text", "text": enhanced_question},
            ]
        }
    ]

    return messages


async def generate_for_sample(
    client: VLLMClient,
    sample: Dict,
    audio_dir: str,
    num_samples: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    session: aiohttp.ClientSession,
) -> Dict:
    """Generate K candidate solutions for a single sample.

    Returns:
        Dict with sample info and generated candidates with parsed thinking/answer
    """
    messages = build_messages(sample, audio_dir)

    # Generate multiple completions
    completions = await client.generate(
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=num_samples,
        session=session,
    )

    # Parse each candidate to extract thinking and answer
    parsed_candidates = []
    for raw_output in completions:
        thinking, answer_text = parse_thinking_and_answer(raw_output)
        final_answer = extract_final_answer(answer_text)

        parsed_candidates.append({
            "raw_output": raw_output,
            "thinking_prediction": thinking,
            "answer_text": answer_text,
            "answer_prediction": final_answer,
        })

    return {
        "id": sample.get("id", ""),
        "sound": sample.get("sound", ""),
        "audio_dir": sample.get("audio_dir", ""),  # Preserve per-sample audio_dir for combined datasets
        "question": sample.get("question", ""),
        "ground_truth": sample.get("answer", ""),
        "candidates": parsed_candidates,
        "num_candidates": len(parsed_candidates),
    }


async def generate_batch(
    client: VLLMClient,
    samples: List[Dict],
    audio_dir: str,
    config: GenerationConfig,
) -> List[Dict]:
    """Generate candidates for a batch of samples concurrently."""

    async with aiohttp.ClientSession() as session:
        tasks = [
            generate_for_sample(
                client=client,
                sample=sample,
                audio_dir=audio_dir,
                num_samples=config.num_samples_per_problem,
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_tokens,
                session=session,
            )
            for sample in samples
        ]

        results = await tqdm_asyncio.gather(*tasks, desc="Generating")
        return results


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


async def main_async(config: GenerationConfig):
    """Main async generation loop."""

    # Setup output directory
    os.makedirs(config.output_dir, exist_ok=True)
    output_path = os.path.join(config.output_dir, "generations.jsonl")

    # Load dataset
    logger.info(f"Loading dataset from {config.data_path}")
    dataset = load_dataset(config.data_path, config.start_idx, config.end_idx)
    logger.info(f"Loaded {len(dataset)} samples")

    # Load existing generations if resuming
    existing = {}
    if config.resume:
        existing = load_existing_generations(output_path)
        logger.info(f"Resuming: found {len(existing)} existing generations")

    # Filter out already processed samples
    samples_to_process = [s for s in dataset if s.get("id") not in existing]
    logger.info(f"Samples to process: {len(samples_to_process)}")

    if not samples_to_process:
        logger.info("No samples to process!")
        return

    # Initialize vLLM client
    client = VLLMClient(
        base_url=config.vllm_base_url,
        model_name=config.model_name,
    )

    # Process in batches
    total_candidates = 0
    batch_size = config.batch_size

    for i in tqdm(range(0, len(samples_to_process), batch_size), desc="Batches"):
        batch = samples_to_process[i:i + batch_size]

        # Generate candidates for batch
        results = await generate_batch(client, batch, config.audio_dir, config)

        # Save results
        for result in results:
            save_generation(output_path, result)
            total_candidates += result["num_candidates"]

        # Log progress
        if (i // batch_size + 1) % 10 == 0:
            logger.info(
                f"Progress: {i + len(batch)}/{len(samples_to_process)} samples, "
                f"{total_candidates} total candidates"
            )

    logger.info(f"Generation complete!")
    logger.info(f"Total samples: {len(samples_to_process)}")
    logger.info(f"Total candidates: {total_candidates}")
    logger.info(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Generate candidates using vLLM")

    # Data settings
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to dataset JSON file")
    parser.add_argument("--audio_dir", type=str, required=True,
                        help="Directory containing audio files")
    parser.add_argument("--output_dir", type=str, default="./outputs/rest/generations",
                        help="Output directory for generations")

    # vLLM settings
    parser.add_argument("--vllm_base_url", type=str, default="http://127.0.0.1:8901/v1",
                        help="vLLM server base URL")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Thinking",
                        help="Model name for vLLM")

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
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Concurrent API requests")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start index in dataset")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="End index in dataset")

    # Resume
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing generations")

    args = parser.parse_args()

    # Build config
    config = GenerationConfig(
        vllm_base_url=args.vllm_base_url,
        model_name=args.model_name,
        num_samples_per_problem=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        data_path=args.data_path,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        resume=args.resume,
    )

    # Run async main
    asyncio.run(main_async(config))


if __name__ == "__main__":
    main()
