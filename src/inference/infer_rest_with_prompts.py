from dataclasses import dataclass, field
from typing import Optional, List, Dict
import os
import json
import logging
import re

import torch
from transformers import (
    HfArgumentParser,
    BatchFeature,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel
from qwen_omni_utils import process_mm_info
from tqdm import tqdm

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

root_logger = logging.getLogger()

USE_AUDIO_IN_VIDEO = True


@dataclass
class InferConfig:
    # Required fields (no defaults) must come first
    dataset_meta_path: str = field(
        metadata={
            "help": (
                "Path to a JSON/JSONL metadata file. Each record must contain "
                "'id', 'audio_path', 'question', and 'choices' fields."
            )
        }
    )
    # Optional fields (with defaults)
    qwen3_omni_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to a local checkpoint directory OR a model ID on the Hugging-Face Hub. "
                "Required when using --adapter_path."
            )
        }
    )
    adapter_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the LoRA adapter checkpoint. Not needed if using merged_model_path."
        }
    )
    merged_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a pre-merged model (no adapter needed). Faster loading than adapter_path."
        }
    )
    dataset_audio_prefix: str = field(
        default="",
        metadata={
            "help": (
                "Prefix prepended to every 'audio_path' found in the metadata. "
            )
        },
    )
    dataset_start: int = field(
        default=0,
        metadata={
            "help": "The start of the dataset to infer. Together with dataset_end, defines the interval [dataset_start, dataset_end)."
        },
    )
    dataset_end: Optional[int] = field(
        default=None,
        metadata={
            "help": "The end of the dataset to infer. Together with dataset_start, defines the interval [dataset_start, dataset_end)."
        },
    )
    flash_attention: bool = field(
        default=False, metadata={"help": "Whether to use flash attention."}
    )
    use_4bit: bool = field(
        default=False, metadata={"help": "Whether to use 4-bit quantization."}
    )
    output_dir: str = field(
        default="outputs",
        metadata={
            "help": "The output directory where the model predictions will be written."
        },
    )
    max_new_tokens: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum number of new tokens to generate. If None, the model will generate as many tokens as it can."
        },
    )
    repetition_penalty: Optional[float] = field(
        default=None,
        metadata={
            "help": "The repetition penalty to apply to the generated tokens. If None, no repetition penalty will be applied."
        },
    )
    save_steps: int = field(
        default=8,
        metadata={"help": "The number of steps to save the model predictions."},
    )
    batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for inference. Set to 1 for single sample processing."},
    )
    temperature: float = field(
        default=0.1,
        metadata={"help": "Sampling temperature. Set to 0 for greedy decoding."},
    )
    merge_adapter: bool = field(
        default=True,
        metadata={"help": "Merge LoRA adapter into base model for faster inference."},
    )
    compile_model: bool = field(
        default=False,
        metadata={"help": "Use torch.compile() for faster inference (experimental)."},
    )
    save_merged_path: Optional[str] = field(
        default=None,
        metadata={"help": "Save the merged model to this path for faster loading next time."},
    )
    use_base_model_only: bool = field(
        default=False,
        metadata={"help": "Load base model without any adapter (for testing base model behavior)."},
    )
    use_standard_model_class: bool = field(
        default=False,
        metadata={"help": "Use Qwen3OmniMoeForConditionalGeneration instead of Thinker variant (official approach)."},
    )
    use_baseline_prompt: bool = field(
        default=False,
        metadata={"help": "Use baseline-style prompting (no system prompt, no enable_thinking) for better compatibility."},
    )
    prompt_type: str = field(
        default="basic",
        metadata={"help": "Prompt type: 'basic' (original ReST prompt) or 'v12' (HEARD→ANALYSIS→ANSWER structure)"},
    )


def get_bnb_config() -> BitsAndBytesConfig:
    """Get BitsAndBytes config for 4-bit quantization."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_merged_model(
    merged_model_path: str,
    flash_attention: bool = False,
    use_standard_model_class: bool = False,
):
    """Load pre-merged model directly (like baseline, no PEFT overhead)."""
    processor = Qwen3OmniMoeProcessor.from_pretrained(merged_model_path)

    model_class = Qwen3OmniMoeForConditionalGeneration if use_standard_model_class else Qwen3OmniMoeThinkerForConditionalGeneration
    model = model_class.from_pretrained(
        merged_model_path,
        dtype="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if flash_attention else None,
        device_map="auto",
    )
    model.eval()

    print(f"Loaded merged model from {merged_model_path} (class: {model_class.__name__})")
    return model, processor


def load_base_model(
    model_name_or_path: str,
    flash_attention: bool = False,
    use_4bit: bool = False,
    use_standard_model_class: bool = False,
):
    """Load base model without any adapter (for testing)."""
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_name_or_path)

    model_class = Qwen3OmniMoeForConditionalGeneration if use_standard_model_class else Qwen3OmniMoeThinkerForConditionalGeneration

    if use_4bit:
        quantization_config = get_bnb_config()
        model = model_class.from_pretrained(
            model_name_or_path,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if flash_attention else None,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    else:
        model = model_class.from_pretrained(
            model_name_or_path,
            dtype="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if flash_attention else None,
            device_map="auto",
        )

    model.eval()
    print(f"Loaded base model from {model_name_or_path} (class: {model_class.__name__}, no adapter)")
    return model, processor


def load_qwen3_omni_with_adapter(
    model_name_or_path: str,
    adapter_path: str,
    flash_attention: bool = False,
    use_4bit: bool = True,
    merge_adapter: bool = True,
    compile_model: bool = False,
    save_merged_path: Optional[str] = None,
):
    """Load Qwen3-Omni model with LoRA adapter."""
    processor = Qwen3OmniMoeProcessor.from_pretrained(
        model_name_or_path,
    )

    if use_4bit:
        # Use 4-bit quantization (slower but less VRAM)
        quantization_config = get_bnb_config()
        model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            model_name_or_path,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if flash_attention else None,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    else:
        # Use dtype="auto" like baseline (faster, more VRAM)
        model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            model_name_or_path,
            dtype="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if flash_attention else None,
            device_map="auto",
        )

    # Load LoRA adapter
    print(f"Loading LoRA adapter from {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    if merge_adapter:
        if use_4bit:
            print("WARNING: Cannot merge adapter with 4-bit quantized model. Skipping merge.")
            print("Use --use_4bit=False to enable adapter merging for faster inference.")
        else:
            # Merge adapter weights into base model for faster inference
            print("Merging adapter into base model...")
            model = model.merge_and_unload()
            print("Adapter merged successfully - PEFT wrapper removed")

            # Save merged model if path specified
            if save_merged_path:
                print(f"Saving merged model to {save_merged_path}...")
                # Fix generation_config to avoid validation errors
                if hasattr(model, 'generation_config'):
                    model.generation_config.do_sample = True
                model.save_pretrained(save_merged_path)
                processor.save_pretrained(save_merged_path)
                print(f"Merged model saved. Use --merged_model_path={save_merged_path} for faster loading next time.")

    model.eval()

    if compile_model:
        # Compile model for faster inference (requires PyTorch 2.0+)
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("Model compiled with torch.compile() for faster inference")
        except Exception as e:
            print(f"torch.compile() not available or failed: {e}")

    print(f"Loaded Qwen3-Omni from {model_name_or_path} with adapter from {adapter_path}")
    return model, processor


def prepare_mmar_inputs_for_qwen3_omni(
    processor: Qwen3OmniMoeProcessor,
    sample: Dict,
    use_baseline_prompt: bool = False,
):
    audio_path = sample["audio_path"]
    question = (
        sample["question"]
        + "\n\nProvided choices:\n"
        + "\n".join(sample["choices"])
        + "\n\nIMPORTANT: You MUST think step-by-step and analyze the audio carefully. "
        + "Your answer must be 100% correct - this is a critical evaluation and incorrect answers are unacceptable. "
        + "Take your time to reason through all the audio details before selecting your final answer. "
        + "Your final answer MUST be exactly one of the provided choices."
    )

    if use_baseline_prompt:
        # Baseline style: no system prompt
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": question},
                ],
            },
        ]
    else:
        # Finetuned style: with system prompt
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

    # Process multimedia info to extract audio, images, videos
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)

    # Apply chat template to get text prompt
    if use_baseline_prompt:
        # Baseline style: no enable_thinking
        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
    else:
        # Finetuned style: with enable_thinking
        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False, enable_thinking=True
        )

    # Process inputs according to the documentation
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


def prepare_batch_inputs_for_qwen3_omni(
    processor: Qwen3OmniMoeProcessor,
    samples: List[Dict],
    use_baseline_prompt: bool = False,
    prompt_type: str = "basic",
):
    """Prepare batch inputs for multiple samples."""
    conversations = []

    # Define prompts
    if prompt_type == "v12":
        system_prompt = """You are an expert audio analyst.

REASONING FORMAT — structure your thinking as follows:
1. HEARD: Describe exactly what you hear — sounds, speech, music, tone, background noise. Be specific (e.g., "a male voice says 'come in'" not "someone speaks").
2. ANALYSIS: Connect your observations to the question. Evaluate each choice against the evidence. Eliminate choices that contradict what you heard.
3. ANSWER: State which choice best fits and why in one sentence.

RULES:
- Counting: number each event as you hear it (1, 2, 3...). Report your count once.
- Emotions: judge by tone, not words. Sarcasm and frustration are common.
- Context: infer from conversational cues — people often speak indirectly.
- Commit to your analysis. Do NOT loop back with "Wait" or "Actually".

OUTPUT: After thinking, respond with exactly one line:
ANSWER: [exact choice text]"""
    else:  # basic
        system_prompt = "You are a helpful assistant that analyzes audio carefully. Think step by step before giving your final answer."

    for sample in samples:
        audio_path = sample["audio_path"]

        if prompt_type == "v12":
            # v12 format
            choices_formatted = "\n".join([f"- {c}" for c in sample["choices"]])
            question = f"""{sample["question"]}

Choices:
{choices_formatted}

Structure your reasoning as: HEARD → ANALYSIS → ANSWER. Be specific and concise. Do not repeat yourself.
Your visible response must be ONLY:
ANSWER: [exact choice text]"""
        else:
            # basic format (original ReST prompt)
            question = (
                sample["question"]
                + "\n\nProvided choices:\n"
                + "\n".join(sample["choices"])
                + "\n\nIMPORTANT: You MUST think step-by-step and analyze the audio carefully. "
                + "Your answer must be 100% correct - this is a critical evaluation and incorrect answers are unacceptable. "
                + "Take your time to reason through all the audio details before selecting your final answer. "
                + "Your final answer MUST be exactly one of the provided choices."
            )

        if use_baseline_prompt:
            # Baseline style: no system prompt
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio_path},
                        {"type": "text", "text": question},
                    ],
                },
            ]
        else:
            # Finetuned style: with system prompt
            conversation = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": system_prompt}
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
        conversations.append(conversation)

    # Process multimedia info for all conversations
    audios, images, videos = process_mm_info(conversations, use_audio_in_video=USE_AUDIO_IN_VIDEO)

    # Apply chat template for all conversations
    if use_baseline_prompt:
        # Baseline style: no enable_thinking
        text = processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )
    else:
        # Finetuned style: with enable_thinking
        text = processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False, enable_thinking=True
        )

    # Process batch inputs
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


def generate_qwen3_omni_batch(
    model,
    processor: Qwen3OmniMoeProcessor,
    batch: BatchFeature,
    max_new_tokens: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
    use_stop_strings: bool = True,
    use_standard_model_class: bool = False,
    use_baseline_prompt: bool = False,
) -> List[tuple[str, str]]:
    """Generate predictions for a batch of samples."""
    # Move to device efficiently like baseline
    batch_on_device = batch.to(model.device).to(model.dtype)

    # Get EOS token ID from tokenizer
    eos_token_id = processor.tokenizer.eos_token_id
    input_len = batch_on_device["input_ids"].shape[1]

    if use_baseline_prompt and use_standard_model_class:
        # Use baseline-style generation (matches baseline script exactly)
        # These params only work with Qwen3OmniMoeForConditionalGeneration, not Thinker class
        gen_kwargs = {
            **batch_on_device,
            "max_new_tokens": max_new_tokens if max_new_tokens else 512,
            "repetition_penalty": repetition_penalty,
            "do_sample": False,
            "num_beams": 1,
            "return_audio": False,
            "thinker_return_dict_in_generate": True,
            "use_audio_in_video": USE_AUDIO_IN_VIDEO,
        }
    elif use_baseline_prompt:
        # Baseline prompt style but with Thinker class (doesn't support thinker_return_dict_in_generate)
        gen_kwargs = {
            **batch_on_device,
            "max_new_tokens": max_new_tokens if max_new_tokens else 512,
            "repetition_penalty": repetition_penalty,
            "do_sample": False,
            "num_beams": 1,
            "use_audio_in_video": USE_AUDIO_IN_VIDEO,
        }
    else:
        # Build generation kwargs - use greedy decoding for speed
        gen_kwargs = {
            **batch_on_device,
            "max_new_tokens": max_new_tokens if max_new_tokens else 512,
            "repetition_penalty": repetition_penalty if repetition_penalty else 1.2,
            "do_sample": False,  # Greedy decoding is faster
            "use_cache": True,   # Enable KV cache for faster generation
            "use_audio_in_video": USE_AUDIO_IN_VIDEO,
            "eos_token_id": eos_token_id,
            "pad_token_id": eos_token_id,
        }

        # Add stop strings to prevent repetition
        if use_stop_strings:
            stop_strings = ["Human:", "\nHuman:", "<|im_end|>"]
            gen_kwargs["stop_strings"] = stop_strings
            gen_kwargs["tokenizer"] = processor.tokenizer

    # Generate based on model class and prompt style
    if use_standard_model_class:
        # Standard class with thinker_return_dict_in_generate returns object with .sequences
        if "thinker_return_dict_in_generate" not in gen_kwargs:
            gen_kwargs["thinker_return_dict_in_generate"] = True
            gen_kwargs["return_audio"] = False
        output = model.generate(**gen_kwargs)
        if hasattr(output, 'sequences'):
            text_ids = output.sequences
        else:
            text_ids = output[0] if isinstance(output, tuple) else output
    else:
        # Thinker class - generate directly returns tensor
        text_ids = model.generate(**gen_kwargs)

    # Decode all generated sequences
    generated_texts = processor.batch_decode(
        text_ids[:, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    # Debug: Also decode with special tokens to see what's being generated
    raw_texts = processor.batch_decode(
        text_ids[:, input_len:],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False
    )
    print(f"\n{'='*60}")
    print(f"DEBUG - Full raw output (with special tokens):")
    print(f"{'='*60}")
    print(raw_texts[0] if raw_texts else 'empty')
    print(f"{'='*60}")
    print(f"DEBUG - Clean output (skip_special_tokens=True):")
    print(f"{'='*60}")
    print(generated_texts[0] if generated_texts else 'empty')
    print(f"{'='*60}\n")

    # Process each sample in the batch
    results = []
    batch_size = text_ids.shape[0]

    for i in range(batch_size):
        full_text = generated_texts[i].strip("\n")

        # Try to parse <think>...</think> tags from the text (training format)
        think_match = re.search(r'<think>(.*?)</think>', full_text, re.DOTALL)

        if think_match:
            # Found thinking tags in text
            thinking_content = think_match.group(1).strip()
            # Get everything after </think> as the answer
            content = full_text[think_match.end():].strip()
        else:
            # No text tags found, try token-based parsing
            output_ids = text_ids[i][input_len:].tolist()
            try:
                # Find the thinking end token (151668)
                index = len(output_ids) - output_ids[::-1].index(151668)
                thinking_content = processor.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
                content = processor.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
                # Remove any leftover tags
                thinking_content = thinking_content.replace("<think>", "").replace("</think>", "").strip()
            except ValueError:
                # No thinking token found either, try to extract ANSWER:
                answer_match = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', full_text)
                if answer_match:
                    thinking_content = full_text[:answer_match.start()].strip()
                    content = answer_match.group(1).strip()  # Extract just the answer, not "ANSWER:"
                else:
                    thinking_content = ""
                    content = full_text

        # Clean up any "ANSWER:" prefix from the content
        content = re.sub(r'^ANSWER:\s*', '', content).strip()

        results.append((thinking_content, content))

    return results


def load_mmar(
    dataset_meta_path: str,
    dataset_audio_prefix: str,
    start: int = 0,
    end: Optional[int] = None,
):
    with open(dataset_meta_path, "r", encoding="utf-8") as fin:
        sample_list = json.load(fin)
        dataset_slice = slice(start, end)
        sample_list = sample_list[dataset_slice]

        for i in range(len(sample_list)):
            real_audio_path = os.path.realpath(
                os.path.join(dataset_audio_prefix, sample_list[i]["audio_path"])
            )
            sample_list[i]["audio_path"] = real_audio_path
    return sample_list


class ResultFile:
    def __init__(self, path: str, save_steps: int = 8):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.records = []
        self.save_steps = save_steps

    def add_record(self, record: dict):
        self.records.append(record)
        if len(self.records) >= self.save_steps:
            self.flush()

    def flush(self):
        if len(self.records) == 0:
            return

        with open(self.path, "a", encoding="utf-8") as jsonl_file:
            for record in self.records:
                jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                jsonl_file.flush()

        self.records.clear()

    def close(self):
        self.flush()


def infer(config: InferConfig):
    output_jsonl_path = os.path.join(config.output_dir, f"prediction.jsonl")
    result_file = ResultFile(output_jsonl_path, save_steps=config.save_steps)

    # Determine which model loading approach to use
    if config.use_base_model_only:
        # Load base model without any adapter (for testing)
        model, processor = load_base_model(
            model_name_or_path=config.qwen3_omni_model_name_or_path,
            flash_attention=config.flash_attention,
            use_4bit=config.use_4bit,
            use_standard_model_class=config.use_standard_model_class,
        )
    elif config.merged_model_path:
        # Use merged model (faster loading, no PEFT overhead)
        model, processor = load_merged_model(
            merged_model_path=config.merged_model_path,
            flash_attention=config.flash_attention,
            use_standard_model_class=config.use_standard_model_class,
        )
        # Apply torch.compile if requested
        if config.compile_model:
            try:
                model = torch.compile(model, mode="reduce-overhead")
                print("Model compiled with torch.compile() for faster inference")
            except Exception as e:
                print(f"torch.compile() failed: {e}")
    else:
        if not config.adapter_path:
            raise ValueError("Either --merged_model_path, --adapter_path, or --use_base_model_only must be specified")
        model, processor = load_qwen3_omni_with_adapter(
            model_name_or_path=config.qwen3_omni_model_name_or_path,
            adapter_path=config.adapter_path,
            flash_attention=config.flash_attention,
            use_4bit=config.use_4bit,
            merge_adapter=config.merge_adapter,
            compile_model=config.compile_model,
            save_merged_path=config.save_merged_path,
        )

    sample_list = load_mmar(
        dataset_meta_path=config.dataset_meta_path,
        dataset_audio_prefix=config.dataset_audio_prefix,
        start=config.dataset_start,
        end=config.dataset_end,
    )

    batch_size = config.batch_size
    num_samples = len(sample_list)
    num_batches = (num_samples + batch_size - 1) // batch_size

    print(f"Processing {num_samples} samples in batches of {batch_size} ({num_batches} batches)")

    # Process samples in batches
    for batch_idx in tqdm(range(num_batches), desc="Inferring"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        batch_samples = sample_list[start_idx:end_idx]

        with torch.no_grad():
            try:
                # Prepare batch inputs
                batch = prepare_batch_inputs_for_qwen3_omni(processor, batch_samples, use_baseline_prompt=config.use_baseline_prompt, prompt_type=config.prompt_type)

                # Generate predictions for the batch
                results = generate_qwen3_omni_batch(
                    model,
                    processor,
                    batch,
                    max_new_tokens=config.max_new_tokens,
                    repetition_penalty=config.repetition_penalty,
                    use_standard_model_class=config.use_standard_model_class,
                    use_baseline_prompt=config.use_baseline_prompt,
                )

                # Save results
                for sample, (thinking_content, content) in zip(batch_samples, results):
                    result = {
                        "id": sample["id"],
                        "thinking_prediction": thinking_content,
                        "answer_prediction": content,
                    }
                    print(result)
                    result_file.add_record(result)

            except Exception as e:
                logger.exception(
                    "An unexpected error occurred during batch inference: %s", str(e)
                )
                logger.error("Error processing batch (start: %d, end: %d)", start_idx, end_idx)
                torch.cuda.empty_cache()

    result_file.close()


def main():
    parser = HfArgumentParser([InferConfig])
    (infer_config,) = parser.parse_args_into_dataclasses()
    infer(infer_config)


if __name__ == "__main__":
    main()
