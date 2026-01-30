"""
Debug inference script for iterating on prompts and fixing answer extraction.
Run on error samples to quickly test improvements.

Usage (with adapter, no 4bit - recommended):
    python debug_inference.py \
        --error_samples_path bf16_errors_analysis.json \
        --adapter_path ../models/rest_Qwen_Qwen3_Omni_30B_A3B_Thinking_20260125_210524 \
        --num_samples 5 \
        --prompt_version v1

Filter by weak subcategory:
    python debug_inference.py \
        --adapter_path ../models/rest_Qwen_Qwen3_Omni_30B_A3B_Thinking_20260125_210524 \
        --filter_subcategory "Spatial Analysis" \
        --prompt_version v2

Test base model (no adapter):
    python debug_inference.py \
        --num_samples 3 \
        --prompt_version baseline
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
import os
import json
import re

import torch
from concurrent.futures import ThreadPoolExecutor
from transformers import (
    HfArgumentParser,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeProcessor,
)
from peft import PeftModel
from transformers import TextStreamer
from qwen_omni_utils import process_mm_info

USE_AUDIO_IN_VIDEO = True

# Default paths
DEFAULT_ADAPTER_PATH = "/home/ikulkar1/qwen_omni_finetune/audio_reasoning_interspeech/src/models/rest_Qwen_Qwen3_Omni_30B_A3B_Thinking_20260125_210524"
DEFAULT_ERROR_SAMPLES = "/home/ikulkar1/qwen_omni_finetune/audio_reasoning_interspeech/src/inference/bf16_errors_analysis.json"
DEFAULT_AUDIO_PREFIX = "/home/ikulkar1/qwen_omni_finetune/audio_reasoning_interspeech/src/data"


@dataclass
class DebugConfig:
    error_samples_path: str = field(
        default=DEFAULT_ERROR_SAMPLES,
        metadata={"help": "Path to error samples JSON file"}
    )
    qwen3_omni_model_name_or_path: str = field(
        default="Qwen/Qwen3-Omni-30B-A3B-Thinking",
        metadata={"help": "Base model path"}
    )
    adapter_path: Optional[str] = field(
        default=DEFAULT_ADAPTER_PATH,
        metadata={"help": "Path to LoRA adapter (set to empty string to skip)"}
    )
    dataset_audio_prefix: str = field(
        default=DEFAULT_AUDIO_PREFIX,
        metadata={"help": "Audio files prefix path"}
    )
    num_samples: int = field(
        default=5,
        metadata={"help": "Number of samples to test (0 = all)"}
    )
    max_new_tokens: int = field(
        default=6000,
        metadata={"help": "Max tokens to generate"}
    )
    filter_subcategory: Optional[str] = field(
        default=None,
        metadata={"help": "Filter to specific subcategory (e.g., 'Spatial Analysis', 'Counting and Statistics')"}
    )
    use_4bit: bool = field(
        default=False,
        metadata={"help": "Use 4-bit quantization (NOT recommended - use bf16 for better quality)"}
    )
    prompt_version: str = field(
        default="v1",
        metadata={"help": "Prompt version: v1-v7, v8 (balanced inference+counting), v9 (anti-Yes bias+strict choice), baseline"}
    )
    skip_adapter: bool = field(
        default=False,
        metadata={"help": "Skip loading adapter (test base model only)"}
    )
    verbose: bool = field(
        default=True,
        metadata={"help": "Show detailed output for each sample"}
    )
    # Generation parameters
    do_sample: bool = field(
        default=False,
        metadata={"help": "Enable sampling (non-deterministic). Set True with temperature for exploration."}
    )
    temperature: float = field(
        default=0.1,
        metadata={"help": "Sampling temperature. Lower=focused, higher=creative. Only used when do_sample=True."}
    )
    top_p: float = field(
        default=0.9,
        metadata={"help": "Nucleus sampling threshold. Only used when do_sample=True."}
    )
    top_k: int = field(
        default=40,
        metadata={"help": "Top-k sampling. Only used when do_sample=True."}
    )
    repetition_penalty: float = field(
        default=1.2,
        metadata={"help": "Repetition penalty. Higher=stronger penalty against loops. Try 1.3-1.5 to reduce circular reasoning. Default 1.2."}
    )
    no_repeat_ngram_size: int = field(
        default=0,
        metadata={"help": "Block repeating N-grams of this size. Set 4-5 to prevent repetitive loops. 0=disabled."}
    )
    num_beams: int = field(
        default=1,
        metadata={"help": "Beam search width. >1 explores multiple paths (slower). 1=greedy/sampling."}
    )
    length_penalty: float = field(
        default=1.0,
        metadata={"help": "Length penalty for beam search. <1 favors shorter, >1 favors longer outputs."}
    )
    batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for inference. >1 processes multiple samples per forward pass. Try 2-4 on A100 80GB."}
    )
    compile_model: bool = field(
        default=False,
        metadata={"help": "Use torch.compile for faster inference. First call is slow (compilation), subsequent calls faster."}
    )


def get_prompt_v1(question: str, choices: List[str]) -> tuple[str, str]:
    """Original finetuned prompt"""
    system = "You are a helpful assistant that analyzes audio carefully. Think step by step before giving your final answer."
    user = (
        question
        + "\n\nProvided choices:\n"
        + "\n".join(choices)
        + "\n\nIMPORTANT: You MUST think step-by-step and analyze the audio carefully. "
        + "Your answer must be 100% correct - this is a critical evaluation and incorrect answers are unacceptable. "
        + "Take your time to reason through all the audio details before selecting your final answer. "
        + "Your final answer MUST be exactly one of the provided choices."
    )
    return system, user


def get_prompt_v2(question: str, choices: List[str]) -> tuple[str, str]:
    """Improved prompt with clearer answer format instruction"""
    system = "You are a helpful assistant that analyzes audio carefully. Think step by step before giving your final answer."

    choices_formatted = "\n".join([f"- {c}" for c in choices])
    user = f"""{question}

Choices:
{choices_formatted}

Instructions:
1. Listen to the audio carefully
2. Think through your reasoning step by step
3. After your reasoning, output your final answer on a new line starting with "ANSWER: " followed by exactly one of the choices above

Your final answer must match one of the provided choices exactly."""
    return system, user


def get_prompt_v3(question: str, choices: List[str]) -> tuple[str, str]:
    """Minimal prompt - let model think naturally"""
    system = "You are an expert audio analyst. Listen carefully and provide accurate answers."

    choices_formatted = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
    user = f"""Question: {question}

Options:
{choices_formatted}

Think step by step, then give your final answer (just the choice text, nothing else)."""
    return system, user


def get_prompt_v4(question: str, choices: List[str]) -> tuple[str, str]:
    """Strict format prompt with clear answer delimiter"""
    system = "You are a helpful assistant that analyzes audio carefully. Think step by step before giving your final answer."

    choices_formatted = "\n".join([f"({chr(65+i)}) {c}" for i, c in enumerate(choices)])
    user = f"""{question}

Options:
{choices_formatted}

Instructions:
1. Listen to the audio carefully and analyze all relevant details
2. Reason through your analysis step by step in your thinking
3. After your reasoning, you MUST output your final answer in this exact format:

ANSWER: [exact choice text from above]

The answer must be the exact text of one of the options (not the letter)."""
    return system, user


def get_prompt_v5(question: str, choices: List[str]) -> tuple[str, str]:
    """Enhanced prompt with word limits and analysis guidance based on error patterns"""
    system = """You are an expert audio analyst. You must:
- Listen to the ENTIRE audio before answering
- Be precise with counts (count each occurrence carefully)
- Identify spatial positions accurately (left/right/center/background)
- Not assume positive emotions - analyze tone objectively
- Base answers ONLY on what you hear, not assumptions"""

    choices_formatted = "\n".join([f"- {c}" for c in choices])
    user = f"""{question}

Choices:
{choices_formatted}

Analysis checklist:
□ If counting: Count each instance explicitly (1, 2, 3...)
□ If spatial: Note specific left/right/center/background positions
□ If emotion: Consider ALL possibilities, don't default to positive
□ If music: Identify tempo, instruments, genre precisely

You may use up to 1000 words for reasoning. Be thorough but stay focused on the audio evidence.

End with: ANSWER: [exact choice text]"""
    return system, user


def get_prompt_v6(question: str, choices: List[str]) -> tuple[str, str]:
    """Concise prompt with strict output format"""
    system = "You are a precise audio analyst. Answer based only on what you hear. Be objective - do not assume positive emotions or round numbers."

    choices_formatted = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
    user = f"""{question}

{choices_formatted}

Rules:
- Count carefully if asked about quantities
- Note exact positions for spatial questions
- Consider negative/neutral emotions equally
- Reasoning can be up to 1000 words if needed

Format your response as:
THINKING: [your analysis]
ANSWER: [exact choice from above]"""
    return system, user


def get_prompt_v7(question: str, choices: List[str]) -> tuple[str, str]:
    """Step-by-step temporal analysis prompt"""
    system = """You are an expert audio analyst. Follow these rules strictly:

HOW TO ANALYZE:
- Go through the audio second by second, describing what you hear at each moment.
- Format: "0:00-0:03: [what you hear]", "0:03-0:06: [what you hear]", etc.
- Only describe sounds, words, and tones you ACTUALLY hear. Do NOT invent or fabricate content.
- After your second-by-second breakdown, state your conclusion.

CRITICAL RULES:
- Do NOT assume positive emotions. Analyze tone objectively - frustration, sarcasm, shock, and dissatisfaction are equally likely.
- Sarcasm: Positive words ("impressive", "great") with flat/dry tone = sarcasm, not genuine praise.
- Laughter can mean nervousness, shock, or awkwardness - not just happiness.
- Count precisely: list each instance (1, 2, 3...). "How many students" ≠ "how many people".
- Answer the EXACT question asked. Do not substitute a related question.
- Consider full conversational context, not just literal words.
- Listen for editing cuts (background noise changes, volume shifts) indicating montage.
- Do NOT repeat yourself or reason in circles. State each observation ONCE.
- When uncertain, pick the choice best supported by specific audio evidence.
- You MUST end with: ANSWER: [exact choice text]"""

    choices_formatted = "\n".join([f"- {c}" for c in choices])
    user = f"""{question}

Choices:
{choices_formatted}

Do all your second-by-second analysis inside your thinking. After thinking, respond ONLY with:
ANSWER: [exact choice text]

Do NOT repeat your analysis outside of thinking. Your visible response should be ONLY the ANSWER line."""
    return system, user


def get_prompt_v8(question: str, choices: List[str]) -> tuple[str, str]:
    """Balanced prompt: allows inference, tighter counting, strict output format"""
    system = """You are an expert audio analyst. Analyze the audio carefully before answering.

ANALYSIS APPROACH:
- Listen to the full audio first, then analyze what you heard.
- You may make reasonable inferences from tone, context, and conversational cues.
- Distinguish between what you directly hear and what you infer. Both are valid.

KEY GUIDELINES:
- Counting: If asked "how many", count ONLY events you are confident about. Listen for each distinct occurrence and number them (1, 2, 3...). If you are unsure whether something is a separate event, do NOT count it. Pick the answer choice closest to your confident count.
- Emotions: Analyze tone objectively. Sarcasm, frustration, and nervousness are common. Do not default to positive.
- Context: Consider what is implied by the conversation, not just literal words. People often speak indirectly.
- Sarcasm: Positive words with flat/dry/exaggerated tone often indicate sarcasm or dissatisfaction.
- Comparisons: When asked which is "best" or "better", focus on clarity, smoothness, and technical quality rather than personal preference.
- Answer the EXACT question asked.

HOW TO DECIDE:
- Evaluate each choice against your observations and inferences.
- Pick the choice with the strongest evidence. When uncertain, prefer the most likely interpretation.

OUTPUT FORMAT:
- Your response after thinking MUST be exactly one line: ANSWER: [exact choice text]
- Do NOT write any explanation, reasoning, or additional text outside of thinking.
- Do NOT repeat your analysis. Only output the ANSWER line."""

    choices_formatted = "\n".join([f"- {c}" for c in choices])
    user = f"""{question}

Choices:
{choices_formatted}

Think step by step inside <think> tags. Your visible response must be ONLY:
ANSWER: [exact choice text]"""
    return system, user


def get_prompt_v9(question: str, choices: List[str]) -> tuple[str, str]:
    """V9: Fixes v8 Content Analysis regressions.
    Changes from v8:
    - Added explicit anti-Yes bias for binary questions
    - Added 'MUST pick from choices only' constraint
    - Simplified analysis approach to reduce overthinking on factual questions
    - Added direct comprehension guidance
    """
    system = """You are an expert audio analyst. Analyze the audio carefully before answering.

ANALYSIS APPROACH:
- Listen to the full audio first, then analyze what you heard.
- You may make reasonable inferences from tone, context, and conversational cues.
- For factual questions (what happened, who said what, what is someone doing), trust the most direct evidence.

KEY GUIDELINES:
- Counting: If asked "how many", count ONLY events you are confident about. Number them (1, 2, 3...). Pick the choice closest to your confident count.
- Yes/No questions: Do NOT default to "Yes". Look for specific evidence SUPPORTING "Yes" before choosing it. If the evidence is ambiguous or absent, prefer "No".
- Emotions: Analyze tone objectively. Sarcasm, frustration, and nervousness are common. Do not default to positive.
- Context: Consider what is implied by the conversation, not just literal words.
- Sarcasm: Positive words with flat/dry/exaggerated tone often indicate sarcasm or dissatisfaction.
- Comparisons: When asked which is "best" or "better", focus on clarity, smoothness, and technical quality.
- Answer the EXACT question asked.

CRITICAL RULES:
- Your answer MUST be exactly one of the provided choices. Never combine multiple choices or invent your own.
- If a question asks you to pick a segment, person, or item, you MUST pick one — do not say "none" or "all" unless that is a listed choice.
- When uncertain, pick the choice with the strongest evidence.

OUTPUT FORMAT:
- Your response after thinking MUST be exactly one line: ANSWER: [exact choice text]
- Do NOT write any explanation, reasoning, or additional text outside of thinking.
- Do NOT repeat your analysis. Only output the ANSWER line."""

    choices_formatted = "\n".join([f"- {c}" for c in choices])
    user = f"""{question}

Choices:
{choices_formatted}

Think step by step inside <think> tags. Your visible response must be ONLY:
ANSWER: [exact choice text]"""
    return system, user


def get_prompt_baseline(question: str, choices: List[str]) -> tuple[Optional[str], str]:
    """Baseline style - no system prompt"""
    user = (
        question
        + "\n\nProvided choices:\n"
        + "\n".join(choices)
        + "\n\nIMPORTANT: You MUST think step-by-step and analyze the audio carefully. "
        + "Your answer must be 100% correct - this is a critical evaluation and incorrect answers are unacceptable. "
        + "Take your time to reason through all the audio details before selecting your final answer. "
        + "Your final answer MUST be exactly one of the provided choices."
    )
    return None, user


def extract_answer(full_text: str, choices: List[str]) -> tuple[str, str]:
    """
    Improved answer extraction that handles multiple formats.
    Returns (thinking, answer)
    """
    full_text = full_text.strip()

    # Method 1: Look for explicit ANSWER: pattern
    answer_patterns = [
        r'ANSWER:\s*(.+?)(?:\n|$)',
        r'FINAL ANSWER:\s*(.+?)(?:\n|$)',
        r'Final answer:\s*(.+?)(?:\n|$)',
        r'The answer is:\s*(.+?)(?:\n|$)',
        r'The correct answer is:\s*(.+?)(?:\n|$)',
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            thinking = full_text[:match.start()].strip()
            # Clean up answer
            answer = re.sub(r'^["\']|["\']$', '', answer)  # Remove quotes
            return thinking, answer

    # Method 2: Check if the last line is a choice
    lines = full_text.strip().split('\n')
    last_line = lines[-1].strip() if lines else ""

    # Check if last line matches any choice (case insensitive)
    for choice in choices:
        if last_line.lower() == choice.lower():
            thinking = '\n'.join(lines[:-1]).strip()
            return thinking, choice
        # Also check if last line contains the choice
        if choice.lower() in last_line.lower() and len(last_line) < len(choice) + 20:
            thinking = '\n'.join(lines[:-1]).strip()
            return thinking, choice

    # Method 3: Look for choice mentioned at the end
    # Search backwards for any choice mention
    text_lower = full_text.lower()
    best_match = None
    best_pos = -1

    for choice in choices:
        # Find last occurrence of choice
        pos = text_lower.rfind(choice.lower())
        if pos > best_pos:
            best_pos = pos
            best_match = choice

    if best_match and best_pos > len(full_text) * 0.5:  # Choice should be in second half
        # Check if it's at the end of a sentence or line
        after_match = full_text[best_pos + len(best_match):].strip()
        if len(after_match) < 50:  # Not too much text after
            thinking = full_text[:best_pos].strip()
            return thinking, best_match

    # Method 4: Fallback - return full text as answer (parsing failed)
    return "", full_text


def load_model(config: DebugConfig):
    """Load model with or without adapter"""
    print(f"Loading processor from {config.qwen3_omni_model_name_or_path}")
    processor = Qwen3OmniMoeProcessor.from_pretrained(config.qwen3_omni_model_name_or_path)

    print(f"Loading model (use_4bit={config.use_4bit})...")
    if config.use_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            config.qwen3_omni_model_name_or_path,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    else:
        # Use dtype="auto" like baseline - faster, more VRAM
        model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            config.qwen3_omni_model_name_or_path,
            dtype="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

    # Load adapter unless skip_adapter is set or no adapter_path provided
    if config.skip_adapter:
        print("Skipping adapter loading (testing base model)")
    elif config.adapter_path:
        print(f"Loading adapter from {config.adapter_path}")
        model = PeftModel.from_pretrained(model, config.adapter_path)
        if not config.use_4bit:
            print("Merging adapter into base model...")
            model = model.merge_and_unload()
            print("Adapter merged successfully - PEFT wrapper removed")
        else:
            print("WARNING: Cannot merge adapter with 4-bit quantized model. Keeping PEFT wrapper.")
    else:
        print("No adapter path provided - using base model only")

    model.eval()

    if config.compile_model:
        print("Compiling model with torch.compile (first inference will be slow)...")
        model = torch.compile(model, mode="reduce-overhead")
        print("Model compiled successfully")

    print(f"Model loaded and ready for inference")
    return model, processor


def preprocess_sample(processor, sample: Dict, config: DebugConfig):
    """Preprocess a sample (audio loading + tokenization) on CPU.
    This can run in a background thread while GPU generates for the current sample."""
    audio_path = os.path.join(config.dataset_audio_prefix, sample['audio_path'])
    question = sample['question']
    choices = sample['choices']

    prompt_funcs = {
        'v1': get_prompt_v1, 'v2': get_prompt_v2, 'v3': get_prompt_v3,
        'v4': get_prompt_v4, 'v5': get_prompt_v5, 'v6': get_prompt_v6,
        'v7': get_prompt_v7, 'v8': get_prompt_v8, 'v9': get_prompt_v9, 'baseline': get_prompt_baseline,
    }
    get_prompt = prompt_funcs.get(config.prompt_version, get_prompt_v1)
    system_prompt, user_prompt = get_prompt(question, choices)

    if system_prompt:
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": user_prompt},
            ]},
        ]
    else:
        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": user_prompt},
            ]},
        ]

    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False,
        enable_thinking=True if system_prompt else False,
    )
    inputs = processor(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO,
    )
    return inputs, system_prompt, user_prompt


def run_inference(
    model,
    processor,
    sample: Dict,
    config: DebugConfig,
    prefetched_inputs=None,
) -> Dict:
    """Run inference on a single sample with detailed output.
    If prefetched_inputs is provided, skip preprocessing (already done async)."""

    question = sample['question']
    choices = sample['choices']

    if prefetched_inputs is not None:
        inputs, system_prompt, user_prompt = prefetched_inputs
    else:
        inputs, system_prompt, user_prompt = preprocess_sample(processor, sample, config)

    inputs = inputs.to(model.device).to(model.dtype)
    input_len = inputs["input_ids"].shape[1]

    # Get EOS token for proper stopping
    eos_token_id = processor.tokenizer.eos_token_id

    # Generation kwargs - all configurable
    gen_kwargs = {
        **inputs,
        "max_new_tokens": config.max_new_tokens,
        "repetition_penalty": config.repetition_penalty,
        "do_sample": config.do_sample,
        "use_cache": True,
        "use_audio_in_video": USE_AUDIO_IN_VIDEO,
        "eos_token_id": eos_token_id,
        "pad_token_id": eos_token_id,
    }

    # Sampling parameters (only active when do_sample=True)
    if config.do_sample:
        gen_kwargs["temperature"] = config.temperature
        gen_kwargs["top_p"] = config.top_p
        gen_kwargs["top_k"] = config.top_k

    # Beam search (only active when num_beams > 1)
    if config.num_beams > 1:
        gen_kwargs["num_beams"] = config.num_beams
        gen_kwargs["length_penalty"] = config.length_penalty
        gen_kwargs["do_sample"] = False  # beam search doesn't mix with sampling

    # Block repeated n-grams
    if config.no_repeat_ngram_size > 0:
        gen_kwargs["no_repeat_ngram_size"] = config.no_repeat_ngram_size

    # Add stop strings to prevent repetition
    stop_strings = ["Human:", "\nHuman:", "<|im_end|>"]
    gen_kwargs["stop_strings"] = stop_strings
    gen_kwargs["tokenizer"] = processor.tokenizer

    # Log generation config
    if config.verbose:
        print(f"\n[GEN CONFIG] do_sample={config.do_sample}, temp={config.temperature}, "
              f"top_p={config.top_p}, top_k={config.top_k}, rep_penalty={config.repetition_penalty}, "
              f"no_repeat_ngram={config.no_repeat_ngram_size}, beams={config.num_beams}, "
              f"max_tokens={config.max_new_tokens}")

    # Set up streamer for real-time token output
    if config.verbose:
        print(f"\n{'='*60}")
        print(f"STREAMING OUTPUT:")
        print(f"{'='*60}")
        streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer

    # Generate
    with torch.inference_mode():
        output_ids = model.generate(**gen_kwargs)

    if config.verbose:
        print(f"\n{'='*60}")

    # Decode
    generated_ids = output_ids[0, input_len:]

    # Decode with and without special tokens
    raw_output = processor.decode(generated_ids, skip_special_tokens=False)
    clean_output = processor.decode(generated_ids, skip_special_tokens=True)

    # Extract thinking and answer using multiple methods (matching finetuned script)
    thinking = ""
    answer = ""

    # Method 1: Try regex-based <think>...</think> parsing from text (training format)
    think_match = re.search(r'<think>(.*?)</think>', clean_output, re.DOTALL)

    if think_match:
        # Found thinking tags in text
        thinking = think_match.group(1).strip()
        # Get everything after </think> as the answer
        answer_raw = clean_output[think_match.end():].strip()
        _, answer = extract_answer(answer_raw, choices)
    else:
        # Method 2: Try token-based thinking extraction
        try:
            # Find thinking end token (151668)
            output_list = generated_ids.tolist()
            index = len(output_list) - output_list[::-1].index(151668)
            thinking = processor.decode(output_list[:index], skip_special_tokens=True).strip()
            answer_raw = processor.decode(output_list[index:], skip_special_tokens=True).strip()
            # Clean thinking tags
            thinking = thinking.replace("<think>", "").replace("</think>", "").strip()
            # Extract answer from remaining text
            _, answer = extract_answer(answer_raw, choices)
        except ValueError:
            # Method 3: No thinking token found, use text-based extraction
            thinking, answer = extract_answer(clean_output, choices)

    # Clean up any "ANSWER:" prefix from the answer
    answer = re.sub(r'^ANSWER:\s*', '', answer, flags=re.IGNORECASE).strip()

    return {
        'raw_output': raw_output,
        'clean_output': clean_output,
        'thinking': thinking,
        'extracted_answer': answer,
        'prompt_used': f"System: {system_prompt}\n\nUser: {user_prompt}" if system_prompt else f"User: {user_prompt}",
    }


def prepare_batch(processor, samples: List[Dict], config: DebugConfig):
    """Prepare a batch of samples for parallel generation."""
    prompt_funcs = {
        'v1': get_prompt_v1, 'v2': get_prompt_v2, 'v3': get_prompt_v3,
        'v4': get_prompt_v4, 'v5': get_prompt_v5, 'v6': get_prompt_v6,
        'v7': get_prompt_v7, 'v8': get_prompt_v8, 'v9': get_prompt_v9, 'baseline': get_prompt_baseline,
    }
    get_prompt = prompt_funcs.get(config.prompt_version, get_prompt_v1)

    all_texts = []
    all_audios = []
    all_images = []
    all_videos = []

    for sample in samples:
        audio_path = os.path.join(config.dataset_audio_prefix, sample['audio_path'])
        system_prompt, user_prompt = get_prompt(sample['question'], sample['choices'])

        if system_prompt:
            conversation = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": user_prompt},
                ]},
            ]
        else:
            conversation = [
                {"role": "user", "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": user_prompt},
                ]},
            ]

        audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False,
            enable_thinking=True if system_prompt else False,
        )

        all_texts.append(text)
        all_audios.append(audios[0] if audios else None)
        if images:
            all_images.extend(images)
        if videos:
            all_videos.extend(videos)

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


def run_batch_inference(
    model, processor, samples: List[Dict], config: DebugConfig,
) -> List[Dict]:
    """Run inference on a batch of samples."""
    inputs = prepare_batch(processor, samples, config)
    inputs = inputs.to(model.device).to(model.dtype)
    input_ids = inputs["input_ids"]
    batch_size = input_ids.shape[0]
    eos_token_id = processor.tokenizer.eos_token_id

    gen_kwargs = {
        **inputs,
        "max_new_tokens": config.max_new_tokens,
        "repetition_penalty": config.repetition_penalty,
        "do_sample": config.do_sample,
        "use_cache": True,
        "use_audio_in_video": USE_AUDIO_IN_VIDEO,
        "eos_token_id": eos_token_id,
        "pad_token_id": eos_token_id,
    }

    if config.do_sample:
        gen_kwargs["temperature"] = config.temperature
        gen_kwargs["top_p"] = config.top_p
        gen_kwargs["top_k"] = config.top_k

    if config.num_beams > 1:
        gen_kwargs["num_beams"] = config.num_beams
        gen_kwargs["length_penalty"] = config.length_penalty
        gen_kwargs["do_sample"] = False

    if config.no_repeat_ngram_size > 0:
        gen_kwargs["no_repeat_ngram_size"] = config.no_repeat_ngram_size

    gen_kwargs["stop_strings"] = ["Human:", "\nHuman:", "<|im_end|>"]
    gen_kwargs["tokenizer"] = processor.tokenizer

    with torch.inference_mode():
        output_ids = model.generate(**gen_kwargs)

    # Decode each sample
    results = []
    for b in range(batch_size):
        generated_ids = output_ids[b, input_ids.shape[1]:]
        # Remove padding tokens
        generated_ids = generated_ids[generated_ids != eos_token_id]

        clean_output = processor.decode(generated_ids, skip_special_tokens=True)
        raw_output = processor.decode(generated_ids, skip_special_tokens=False)
        choices = samples[b]['choices']

        # Extract thinking + answer (same logic as single path)
        thinking = ""
        answer = ""

        think_match = re.search(r'<think>(.*?)</think>', clean_output, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            answer_raw = clean_output[think_match.end():].strip()
            _, answer = extract_answer(answer_raw, choices)
        else:
            output_list = generated_ids.tolist()
            try:
                index = len(output_list) - output_list[::-1].index(151668)
                thinking = processor.decode(output_list[:index], skip_special_tokens=True).strip()
                answer_raw = processor.decode(output_list[index:], skip_special_tokens=True).strip()
                thinking = thinking.replace("<think>", "").replace("</think>", "").strip()
                _, answer = extract_answer(answer_raw, choices)
            except ValueError:
                thinking, answer = extract_answer(clean_output, choices)

        answer = re.sub(r'^ANSWER:\s*', '', answer, flags=re.IGNORECASE).strip()

        results.append({
            'raw_output': raw_output,
            'clean_output': clean_output,
            'thinking': thinking,
            'extracted_answer': answer,
        })

    return results


def main():
    import time

    parser = HfArgumentParser([DebugConfig])
    config, = parser.parse_args_into_dataclasses()

    # Load error samples
    with open(config.error_samples_path, 'r') as f:
        error_samples = json.load(f)

    print(f"Loaded {len(error_samples)} error samples")

    # Filter by subcategory if specified
    if config.filter_subcategory:
        error_samples = [s for s in error_samples if s['sub_category'] == config.filter_subcategory]
        print(f"Filtered to {len(error_samples)} samples in '{config.filter_subcategory}'")

    # Limit samples
    if config.num_samples > 0:
        error_samples = error_samples[:config.num_samples]

    bs = config.batch_size
    print(f"Testing on {len(error_samples)} samples with prompt version: {config.prompt_version}, batch_size: {bs}")
    print("=" * 80)

    # Load model
    model, processor = load_model(config)

    # Run inference
    correct = 0
    total = 0
    results = []
    start_time = time.time()

    if bs > 1:
        # --- Batched path ---
        for batch_start in range(0, len(error_samples), bs):
            batch_samples = error_samples[batch_start:batch_start + bs]
            batch_end = batch_start + len(batch_samples)

            print(f"\n{'='*80}")
            print(f"BATCH [{batch_start+1}-{batch_end}/{len(error_samples)}]")
            for s in batch_samples:
                print(f"  {s['id']}: {s['question'][:60]}... | truth: {s['ground_truth']}")
            print("-" * 80)

            try:
                batch_results = run_batch_inference(model, processor, batch_samples, config)

                for sample, result in zip(batch_samples, batch_results):
                    is_correct = result['extracted_answer'].strip().lower() == sample['ground_truth'].strip().lower()
                    total += 1
                    if is_correct:
                        correct += 1

                    tag = "CORRECT" if is_correct else f"WRONG (expected: {sample['ground_truth']})"
                    print(f"  {sample['id']}: {result['extracted_answer']} >>> {tag}")

                    results.append({
                        'id': sample['id'],
                        'ground_truth': sample['ground_truth'],
                        'prediction': result['extracted_answer'],
                        'correct': is_correct,
                        'thinking': result['thinking'][:500],
                    })

                elapsed = time.time() - start_time
                rate = total / elapsed if elapsed > 0 else 0
                print(f"  [{total} done, {correct}/{total} correct, {rate:.2f} samples/sec]")

            except Exception as e:
                print(f"BATCH FAILED: {e}, falling back to single processing...")
                import traceback
                traceback.print_exc()
                torch.cuda.empty_cache()

                # Fallback: process one by one
                for sample in batch_samples:
                    try:
                        result = run_inference(model, processor, sample, config)
                        is_correct = result['extracted_answer'].strip().lower() == sample['ground_truth'].strip().lower()
                        total += 1
                        if is_correct:
                            correct += 1

                        tag = "CORRECT" if is_correct else f"WRONG (expected: {sample['ground_truth']})"
                        print(f"  (fallback) {sample['id']}: {result['extracted_answer']} >>> {tag}")

                        results.append({
                            'id': sample['id'],
                            'ground_truth': sample['ground_truth'],
                            'prediction': result['extracted_answer'],
                            'correct': is_correct,
                            'thinking': result['thinking'][:500],
                        })
                    except Exception as e2:
                        print(f"  ERROR on {sample['id']}: {e2}")

            # Save intermediate
            output_path = f"debug_results_{config.prompt_version}.json"
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)

    else:
        # --- Single sample path with async prefetching ---
        executor = ThreadPoolExecutor(max_workers=1)
        prefetch_future = None

        # Prefetch first sample
        if len(error_samples) > 0:
            prefetch_future = executor.submit(preprocess_sample, processor, error_samples[0], config)

        for i, sample in enumerate(error_samples):
            print(f"\n{'='*80}")
            print(f"[{i+1}/{len(error_samples)}] ID: {sample['id']}")
            print(f"Category: {sample['category']} / {sample['sub_category']}")
            print(f"Modality: {sample['modality']}")
            print(f"Question: {sample['question'][:100]}...")
            print(f"Choices: {sample['choices']}")
            print(f"Ground Truth: {sample['ground_truth']}")
            print(f"Baseline: {sample['baseline_prediction']} ({'correct' if sample['baseline_correct'] else 'wrong'})")
            print("-" * 80)

            try:
                # Get prefetched inputs for current sample
                prefetched = None
                if prefetch_future is not None:
                    prefetched = prefetch_future.result()
                    prefetch_future = None

                # Start prefetching NEXT sample while GPU generates current
                if i + 1 < len(error_samples):
                    prefetch_future = executor.submit(preprocess_sample, processor, error_samples[i + 1], config)

                result = run_inference(model, processor, sample, config, prefetched_inputs=prefetched)

                print(f"\n[RAW OUTPUT (first 500 chars)]:")
                print(result['raw_output'][:500])
                print(f"\n[EXTRACTED THINKING (first 300 chars)]:")
                print(result['thinking'][:300] if result['thinking'] else "(empty)")
                print(f"\n[EXTRACTED ANSWER]: {result['extracted_answer']}")

                is_correct = result['extracted_answer'].strip().lower() == sample['ground_truth'].strip().lower()
                total += 1
                if is_correct:
                    correct += 1
                    print(f"\n>>> CORRECT!")
                else:
                    print(f"\n>>> WRONG (expected: {sample['ground_truth']})")

                results.append({
                    'id': sample['id'],
                    'ground_truth': sample['ground_truth'],
                    'prediction': result['extracted_answer'],
                    'correct': is_correct,
                    'thinking': result['thinking'][:500],
                })

                output_path = f"debug_results_{config.prompt_version}.json"
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)

            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()

            print("-" * 80)

        executor.shutdown(wait=False)

    # Summary
    elapsed = time.time() - start_time
    rate = total / elapsed if elapsed > 0 else 0
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Prompt version: {config.prompt_version}")
    print(f"Batch size: {bs}")
    print(f"torch.compile: {config.compile_model}")
    print(f"Async prefetch: {bs == 1}")
    print(f"Correct: {correct}/{total} ({correct/total*100:.1f}%)" if total > 0 else "No samples processed")
    print(f"Time: {elapsed:.1f}s ({rate:.2f} samples/sec)")

    # Save results
    output_path = f"debug_results_{config.prompt_version}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
