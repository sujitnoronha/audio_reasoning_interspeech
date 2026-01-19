from dataclasses import dataclass, field
from typing import Optional, List, Dict
import os
import json
import logging

import torch
from transformers import (
    HfArgumentParser,
    BatchFeature,
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
)
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
    qwen3_omni_model_name_or_path: str = field(
        metadata={
            "help": (
                "Path to a local checkpoint directory OR a model ID on the Hugging-Face Hub, "
                "e.g.  './checkpoints/qwen3_omni_moe'."
            )
        }
    )
    dataset_meta_path: str = field(
        metadata={
            "help": (
                "Path to a JSON/JSONL metadata file. Each record must contain "
                "'id', 'audio_path', 'question', and 'choices' fields."
            )
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

def load_qwen3_omni(
    model_name_or_path: str,
    flash_attention: bool = False,
):
    processor = Qwen3OmniMoeProcessor.from_pretrained(
        model_name_or_path,
    )

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_name_or_path,
        dtype="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if flash_attention else None,
        device_map="auto",
    )
    model.eval()
    print(f"Loaded Qwen3-Omni from {model_name_or_path}")
    return model, processor

def prepare_mmar_inputs_for_qwen3_omni(
    processor: Qwen3OmniMoeProcessor,
    sample: Dict,
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

    # Prepare conversation in the format expected by the model
    conversation = [
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
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
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
):
    """Prepare batch inputs for multiple samples."""
    conversations = []

    for sample in samples:
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

        conversation = [
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
    text = processor.apply_chat_template(
        conversations, add_generation_prompt=True, tokenize=False
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
    model: Qwen3OmniMoeForConditionalGeneration,
    processor: Qwen3OmniMoeProcessor,
    batch: BatchFeature,
    max_new_tokens: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
) -> List[tuple[str, str]]:
    """Generate predictions for a batch of samples."""
    batch_on_device = batch.to(model.device).to(model.dtype)

    # Generate with thinker_return_dict_in_generate for thinking model
    # Batch inference does not support returning audio
    text_ids = model.generate(
        **batch_on_device,
        max_new_tokens=max_new_tokens if max_new_tokens else 512,
        repetition_penalty=repetition_penalty,
        do_sample=False,
        num_beams=1,
        return_audio=False,
        thinker_return_dict_in_generate=True,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
    )

    # Decode all generated sequences
    generated_texts = processor.batch_decode(
        text_ids.sequences[:, batch_on_device["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    # Process each sample in the batch
    results = []
    batch_size = text_ids.sequences.shape[0]

    for i in range(batch_size):
        # Get output IDs for this sample
        output_ids = text_ids.sequences[i][batch_on_device["input_ids"].shape[1]:].tolist()

        try:
            # Find the thinking end token (151668)
            index = len(output_ids) - output_ids[::-1].index(151668)
            thinking_content = processor.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = processor.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        except ValueError:
            # No thinking token found, treat all as answer
            thinking_content = ""
            content = generated_texts[i].strip("\n")

        # Remove <think> and </think> tags from thinking content
        thinking_content = thinking_content.replace("<think>", "").replace("</think>", "").strip()

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

    model, processor = load_qwen3_omni(
        model_name_or_path=config.qwen3_omni_model_name_or_path,
        flash_attention=config.flash_attention,
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
                batch = prepare_batch_inputs_for_qwen3_omni(processor, batch_samples)

                # Generate predictions for the batch
                results = generate_qwen3_omni_batch(
                    model,
                    processor,
                    batch,
                    max_new_tokens=config.max_new_tokens,
                    repetition_penalty=config.repetition_penalty,
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