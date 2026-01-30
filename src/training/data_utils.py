"""
Data loading utilities for audio reasoning training.

Provides dataset classes and data processing functions for
ReST and GRPO training.
"""

import json
import os
from typing import Optional, List, Dict, Any, Union

import torch
from torch.utils.data import Dataset, DataLoader
from qwen_omni_utils import process_mm_info

USE_AUDIO_IN_VIDEO = True


class AudioReasoningDataset(Dataset):
    """Base dataset for audio reasoning tasks.

    Loads samples with audio, question, and answer fields.
    Supports both MCQ format and free-form Q&A.
    """

    def __init__(
        self,
        data_path: str,
        processor,
        audio_dir: str = "",
        max_length: int = 2048,
        include_audio_features: bool = True,
    ):
        """Initialize dataset.

        Args:
            data_path: Path to JSON file containing samples
            processor: Model processor for tokenization
            audio_dir: Directory containing audio files
            max_length: Maximum sequence length
            include_audio_features: Whether to process audio features
        """
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.processor = processor
        self.audio_dir = audio_dir
        self.max_length = max_length
        self.include_audio_features = include_audio_features

    def __len__(self):
        return len(self.data)

    def _get_question_answer(self, sample: Dict) -> tuple[str, str]:
        """Extract question and answer from sample."""
        question = sample.get("question", "")
        answer = sample.get("answer", "")

        # If not found, extract from conversations
        if not question:
            for conv in sample.get("conversations", []):
                if conv["from"] == "human":
                    question = conv["value"]
                elif conv["from"] == "gpt":
                    answer = conv["value"]

        return question, answer

    def __getitem__(self, idx) -> Dict[str, Any]:
        sample = self.data[idx]

        # Get audio path
        audio_path = os.path.join(self.audio_dir, sample["sound"])

        # Get question and answer
        question, answer = self._get_question_answer(sample)

        # Clean question (remove <sound> tag)
        clean_question = question.replace("<sound>", "").strip()

        # Create conversation format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": clean_question},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer},
                ],
            },
        ]

        if self.include_audio_features:
            # Process multimedia
            audios, images, videos = process_mm_info(
                conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO
            )

            # Apply chat template
            text = self.processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False
            )

            # Tokenize with audio
            inputs = self.processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                use_audio_in_video=USE_AUDIO_IN_VIDEO,
            )

            # Squeeze batch dimension
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}

            # Create labels
            inputs["labels"] = inputs["input_ids"].clone()
        else:
            # Return raw data without processing
            inputs = {
                "question": clean_question,
                "answer": answer,
                "audio_path": audio_path,
            }

        inputs["id"] = sample.get("id", str(idx))
        return inputs


class PromptOnlyDataset(Dataset):
    """Dataset that returns only prompts (for generation/RL training).

    Used for GRPO and other RL methods where we need to generate
    completions from prompts.
    """

    def __init__(
        self,
        data_path: str,
        processor,
        audio_dir: str = "",
        max_prompt_length: int = 1024,
    ):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.processor = processor
        self.audio_dir = audio_dir
        self.max_prompt_length = max_prompt_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, Any]:
        sample = self.data[idx]

        # Get audio path
        audio_path = os.path.join(self.audio_dir, sample["sound"])

        # Get question and answer
        question = sample.get("question", "")
        answer = sample.get("answer", "")

        if not question:
            for conv in sample.get("conversations", []):
                if conv["from"] == "human":
                    question = conv["value"]
                elif conv["from"] == "gpt":
                    answer = conv["value"]

        # Clean question
        clean_question = question.replace("<sound>", "").strip()

        return {
            "prompt": clean_question,
            "audio_path": audio_path,
            "answer": answer,
            "id": sample.get("id", str(idx)),
        }


class AudioDataCollator:
    """Data collator for audio reasoning datasets.

    Handles batching of processed inputs including audio features.
    """

    def __init__(self, processor, max_length: int = 2048, pad_to_multiple_of: int = 8):
        self.processor = processor
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {}

        for key in features[0].keys():
            values = [f[key] for f in features]

            if isinstance(values[0], torch.Tensor):
                # Stack tensors
                batch[key] = torch.stack(values)
            elif isinstance(values[0], (int, float)):
                batch[key] = torch.tensor(values)
            else:
                # Keep as list (strings, etc.)
                batch[key] = values

        return batch


def load_dataset_splits(
    train_path: str,
    processor,
    audio_dir: str = "",
    eval_path: Optional[str] = None,
    max_length: int = 2048,
    dataset_class: type = AudioReasoningDataset,
    **kwargs,
) -> tuple[Dataset, Optional[Dataset]]:
    """Load training and optional evaluation dataset splits.

    Args:
        train_path: Path to training data JSON
        processor: Model processor
        audio_dir: Directory containing audio files
        eval_path: Optional path to evaluation data JSON
        max_length: Maximum sequence length
        dataset_class: Dataset class to use
        **kwargs: Additional arguments for dataset class

    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    train_dataset = dataset_class(
        data_path=train_path,
        processor=processor,
        audio_dir=audio_dir,
        max_length=max_length,
        **kwargs,
    )

    eval_dataset = None
    if eval_path:
        eval_dataset = dataset_class(
            data_path=eval_path,
            processor=processor,
            audio_dir=audio_dir,
            max_length=max_length,
            **kwargs,
        )

    return train_dataset, eval_dataset


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    collate_fn=None,
    pin_memory: bool = False,
) -> DataLoader:
    """Create a DataLoader for the dataset.

    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        collate_fn: Custom collate function
        pin_memory: Whether to pin memory

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )


def filter_samples_by_audio(
    data: List[Dict],
    audio_dir: str,
) -> List[Dict]:
    """Filter samples to only include those with existing audio files.

    Args:
        data: List of sample dictionaries
        audio_dir: Directory containing audio files

    Returns:
        Filtered list of samples
    """
    filtered = []
    for sample in data:
        audio_path = os.path.join(audio_dir, sample["sound"])
        if os.path.exists(audio_path):
            filtered.append(sample)
    return filtered
