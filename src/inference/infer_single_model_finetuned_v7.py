"""
Finetuned inference script with v7 prompt (second-by-second temporal analysis).
Processes one sample at a time, saves after each sample.

Usage (with adapter, BF16):
    python infer_single_model_finetuned_v7.py \
        --dataset_meta_path ../data/MMAR-meta.json \
        --dataset_audio_prefix ../data \
        --qwen3_omni_model_name_or_path Qwen/Qwen3-Omni-30B-A3B-Thinking \
        --adapter_path ../models/rest_Qwen_Qwen3_Omni_30B_A3B_Thinking_20260125_210524 \
        --output_dir outputs/v7_finetuned

Usage (with merged model):
    python infer_single_model_finetuned_v7.py \
        --dataset_meta_path ../data/MMAR-meta.json \
        --dataset_audio_prefix ../data \
        --merged_model_path ../models/merged_model \
        --output_dir outputs/v7_merged
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
import os
import json
import logging
import re

import torch
from transformers import (
    HfArgumentParser,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeProcessor,
)
from peft import PeftModel
from qwen_omni_utils import process_mm_info
from tqdm import tqdm

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

USE_AUDIO_IN_VIDEO = True

# --- v7 Prompt ---

V7_SYSTEM_PROMPT = """You are an expert audio analyst. Follow these rules strictly:

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

HOW TO DECIDE:
- After your analysis, evaluate EACH choice against your observations.
- For each choice, note what evidence supports or contradicts it.
- Pick the choice with the MOST supporting evidence and LEAST contradictions.
- Do NOT commit to an answer early. Finish all analysis first, then decide.
- You MUST end with: ANSWER: [exact choice text]"""


def build_v7_user_prompt(question: str, choices: List[str]) -> str:
    choices_formatted = "\n".join([f"- {c}" for c in choices])
    return f"""{question}

Choices:
{choices_formatted}

Do all your second-by-second analysis inside your thinking. After thinking, respond ONLY with:
ANSWER: [exact choice text]

Do NOT repeat your analysis outside of thinking. Your visible response should be ONLY the ANSWER line."""


@dataclass
class InferConfig:
    # Required
    dataset_meta_path: str = field(
        metadata={"help": "Path to a JSON metadata file with 'id', 'audio_path', 'question', 'choices' fields."}
    )
    # Model loading options (one of these must be provided)
    qwen3_omni_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Base model path (required when using --adapter_path)."}
    )
    adapter_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to LoRA adapter checkpoint."}
    )
    merged_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pre-merged model (no adapter needed)."}
    )
    # Dataset options
    dataset_audio_prefix: str = field(
        default="",
        metadata={"help": "Prefix prepended to every 'audio_path' in the metadata."},
    )
    dataset_start: int = field(
        default=0,
        metadata={"help": "Start index for dataset slice [start, end)."},
    )
    dataset_end: Optional[int] = field(
        default=None,
        metadata={"help": "End index for dataset slice [start, end)."},
    )
    # Model options
    flash_attention: bool = field(
        default=True, metadata={"help": "Use flash attention 2 (requires flash-attn package)."}
    )
    merge_adapter: bool = field(
        default=True,
        metadata={"help": "Merge LoRA adapter into base model for faster inference."},
    )
    save_merged_path: Optional[str] = field(
        default=None,
        metadata={"help": "Save merged model to this path for reuse."},
    )
    # Output options
    output_dir: str = field(
        default="outputs",
        metadata={"help": "Output directory for prediction.jsonl."},
    )
    # Generation parameters
    max_new_tokens: int = field(
        default=6000,
        metadata={"help": "Max tokens to generate."},
    )
    compile_model: bool = field(
        default=False,
        metadata={"help": "Use torch.compile for faster inference (slow first call, faster subsequent)."},
    )
    do_sample: bool = field(
        default=True,
        metadata={"help": "Enable sampling. Set False for greedy decoding."},
    )
    temperature: float = field(
        default=0.1,
        metadata={"help": "Sampling temperature. Only used when do_sample=True."},
    )
    top_p: float = field(
        default=0.9,
        metadata={"help": "Nucleus sampling threshold. Only used when do_sample=True."},
    )
    top_k: int = field(
        default=40,
        metadata={"help": "Top-k sampling. Only used when do_sample=True."},
    )
    repetition_penalty: float = field(
        default=1.2,
        metadata={"help": "Repetition penalty. Higher = stronger penalty against loops."},
    )
    no_repeat_ngram_size: int = field(
        default=0,
        metadata={"help": "Block repeating N-grams of this size. 0 = disabled."},
    )


# --- Model loading ---

def load_merged_model(merged_model_path: str, flash_attention: bool = False):
    processor = Qwen3OmniMoeProcessor.from_pretrained(merged_model_path)
    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        merged_model_path,
        dtype="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if flash_attention else None,
        device_map="auto",
    )
    model.eval()
    print(f"Loaded merged model from {merged_model_path}")
    return model, processor


def load_model_with_adapter(
    model_name_or_path: str,
    adapter_path: str,
    flash_attention: bool = False,
    merge_adapter: bool = True,
    save_merged_path: Optional[str] = None,
):
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_name_or_path)

    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        model_name_or_path,
        dtype="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if flash_attention else None,
        device_map="auto",
    )

    print(f"Loading LoRA adapter from {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    if merge_adapter:
        print("Merging adapter into base model...")
        model = model.merge_and_unload()
        print("Adapter merged successfully")

        if save_merged_path:
            print(f"Saving merged model to {save_merged_path}...")
            if hasattr(model, 'generation_config'):
                model.generation_config.do_sample = True
            model.save_pretrained(save_merged_path)
            processor.save_pretrained(save_merged_path)
            print(f"Merged model saved to {save_merged_path}")

    model.eval()
    print(f"Loaded Qwen3-Omni from {model_name_or_path} with adapter from {adapter_path}")
    return model, processor


# --- Input preparation ---

def prepare_input(processor: Qwen3OmniMoeProcessor, sample: Dict):
    """Prepare input for a single sample using v7 prompt."""
    audio_path = sample["audio_path"]
    user_prompt = build_v7_user_prompt(sample["question"], sample["choices"])

    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": V7_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)

    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False, enable_thinking=True
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


# --- Answer extraction ---

def extract_answer_from_text(text: str, choices: List[str]) -> str:
    """Extract the answer from model output, trying multiple patterns."""
    # Method 1: ANSWER: pattern
    answer_patterns = [
        r'ANSWER:\s*(.+?)(?:\n|$)',
        r'FINAL ANSWER:\s*(.+?)(?:\n|$)',
        r'Final answer:\s*(.+?)(?:\n|$)',
        r'The answer is:\s*(.+?)(?:\n|$)',
        r'The correct answer is:\s*(.+?)(?:\n|$)',
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            answer = re.sub(r'^["\']|["\']$', '', answer)
            return answer

    # Method 2: Last line matches a choice
    lines = text.strip().split('\n')
    last_line = lines[-1].strip() if lines else ""
    for choice in choices:
        if last_line.lower() == choice.lower():
            return choice
        if choice.lower() in last_line.lower() and len(last_line) < len(choice) + 20:
            return choice

    # Method 3: Choice mentioned near end
    text_lower = text.lower()
    best_match = None
    best_pos = -1
    for choice in choices:
        pos = text_lower.rfind(choice.lower())
        if pos > best_pos:
            best_pos = pos
            best_match = choice
    if best_match and best_pos > len(text) * 0.5:
        after_match = text[best_pos + len(best_match):].strip()
        if len(after_match) < 50:
            return best_match

    # Fallback
    return text.strip()


# --- Generation ---

def generate_single(
    model,
    processor: Qwen3OmniMoeProcessor,
    inputs,
    config: InferConfig,
    choices: List[str],
) -> tuple[str, str]:
    """Generate prediction for a single sample."""
    inputs_on_device = inputs.to(model.device).to(model.dtype)
    input_len = inputs_on_device["input_ids"].shape[1]
    eos_token_id = processor.tokenizer.eos_token_id

    gen_kwargs = {
        **inputs_on_device,
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

    if config.no_repeat_ngram_size > 0:
        gen_kwargs["no_repeat_ngram_size"] = config.no_repeat_ngram_size

    # Stop strings
    gen_kwargs["stop_strings"] = ["Human:", "\nHuman:", "<|im_end|>"]
    gen_kwargs["tokenizer"] = processor.tokenizer

    text_ids = model.generate(**gen_kwargs)

    # Decode
    generated_ids = text_ids[0, input_len:]
    full_text = processor.decode(generated_ids, skip_special_tokens=True).strip("\n")

    # Extract thinking and answer
    think_match = re.search(r'<think>(.*?)</think>', full_text, re.DOTALL)

    if think_match:
        thinking_content = think_match.group(1).strip()
        answer_raw = full_text[think_match.end():].strip()
    else:
        # Token-based thinking extraction (token 151668)
        output_ids = generated_ids.tolist()
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
            thinking_content = processor.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            thinking_content = thinking_content.replace("<think>", "").replace("</think>", "").strip()
            answer_raw = processor.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        except ValueError:
            thinking_content = ""
            answer_raw = full_text

    # Extract answer
    content = extract_answer_from_text(answer_raw, choices)
    content = re.sub(r'^ANSWER:\s*', '', content, flags=re.IGNORECASE).strip()

    return thinking_content, content


# --- Dataset loading ---

def load_mmar(
    dataset_meta_path: str,
    dataset_audio_prefix: str,
    start: int = 0,
    end: Optional[int] = None,
):
    with open(dataset_meta_path, "r", encoding="utf-8") as fin:
        sample_list = json.load(fin)
        sample_list = sample_list[start:end]

        for i in range(len(sample_list)):
            real_audio_path = os.path.realpath(
                os.path.join(dataset_audio_prefix, sample_list[i]["audio_path"])
            )
            sample_list[i]["audio_path"] = real_audio_path
    return sample_list


# --- Main inference ---

def infer(config: InferConfig):
    output_jsonl_path = os.path.join(config.output_dir, "prediction.jsonl")
    os.makedirs(config.output_dir, exist_ok=True)

    # Clear any existing file from previous runs to avoid duplicates
    if os.path.exists(output_jsonl_path):
        print(f"WARNING: {output_jsonl_path} already exists. Removing to avoid duplicates.")
        os.remove(output_jsonl_path)

    # Load model
    if config.merged_model_path:
        model, processor = load_merged_model(
            merged_model_path=config.merged_model_path,
            flash_attention=config.flash_attention,
        )
    else:
        if not config.adapter_path or not config.qwen3_omni_model_name_or_path:
            raise ValueError("Provide either --merged_model_path OR both --qwen3_omni_model_name_or_path and --adapter_path")
        model, processor = load_model_with_adapter(
            model_name_or_path=config.qwen3_omni_model_name_or_path,
            adapter_path=config.adapter_path,
            flash_attention=config.flash_attention,
            merge_adapter=config.merge_adapter,
            save_merged_path=config.save_merged_path,
        )

    # Optional: torch.compile for faster generation
    if config.compile_model:
        print("Compiling model with torch.compile (first call will be slow)...")
        model = torch.compile(model, mode="reduce-overhead")

    sample_list = load_mmar(
        dataset_meta_path=config.dataset_meta_path,
        dataset_audio_prefix=config.dataset_audio_prefix,
        start=config.dataset_start,
        end=config.dataset_end,
    )

    num_samples = len(sample_list)
    print(f"Processing {num_samples} samples (one at a time, saving after each)")
    print(f"Generation: do_sample={config.do_sample}, temp={config.temperature}, "
          f"rep_penalty={config.repetition_penalty}, max_tokens={config.max_new_tokens}")

    for i, sample in enumerate(tqdm(sample_list, desc="Inferring")):
        with torch.inference_mode():
            try:
                inputs = prepare_input(processor, sample)
                thinking_content, content = generate_single(
                    model, processor, inputs, config, sample["choices"]
                )

                result = {
                    "id": sample["id"],
                    "thinking_prediction": thinking_content,
                    "answer_prediction": content,
                }
                print(result)

                # Save immediately after each sample
                with open(output_jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()

            except Exception as e:
                logger.exception("Error on sample %s: %s", sample.get("id", i), str(e))
                torch.cuda.empty_cache()

    print(f"Done. Results saved to {output_jsonl_path}")


def main():
    parser = HfArgumentParser([InferConfig])
    (infer_config,) = parser.parse_args_into_dataclasses()
    infer(infer_config)


if __name__ == "__main__":
    main()
