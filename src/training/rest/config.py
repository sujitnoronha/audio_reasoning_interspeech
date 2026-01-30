"""
Configuration for ReST (Reinforced Self-Training) pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class GenerationConfig:
    """Configuration for Phase 1: Candidate Generation."""

    # vLLM server settings
    vllm_base_url: str = "http://127.0.0.1:8901/v1"
    model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Thinking"

    # Generation parameters
    num_samples_per_problem: int = 16  # K samples per problem
    temperature: float = 0.9
    top_p: float = 0.95
    top_k: int = 50
    max_tokens: int = 2048

    # Data settings
    data_path: str = ""
    audio_dir: str = ""
    output_dir: str = "./outputs/rest/generations"

    # Processing settings
    batch_size: int = 8  # Concurrent API requests
    start_idx: int = 0
    end_idx: Optional[int] = None
    save_every: int = 100

    # Resume
    resume: bool = False


@dataclass
class FilterConfig:
    """Configuration for Phase 2: Evaluation and Filtering."""

    # Input/Output paths
    generations_path: str = ""  # Path to generations JSONL
    output_path: str = ""  # Path to filtered output

    # Filter strategy: "all", "best", "top_k"
    filter_strategy: str = "top_k"
    top_k: int = 3  # For top_k strategy

    # Answer extraction
    answer_pattern: str = r"<answer>(.*?)</answer>"  # Regex to extract answer
    fallback_patterns: List[str] = field(default_factory=lambda: [
        r"\(([A-D])\)",  # Match (A), (B), etc.
        r"^([A-D])\.",   # Match A., B., etc.
        r"answer is[:\s]*(.+?)(?:\.|$)",  # Match "answer is X"
    ])

    # Matching settings
    case_sensitive: bool = False
    strip_whitespace: bool = True


@dataclass
class TrainConfig:
    """Configuration for Phase 3: SFT Training."""

    # Model settings
    model_name_or_path: str = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
    use_4bit: bool = True
    use_flash_attention: bool = True

    # Data settings
    train_data_path: str = ""  # Filtered data from Phase 2
    eval_data_path: Optional[str] = None
    audio_dir: str = ""

    # LoRA settings
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05

    # Training settings
    output_dir: str = "./outputs/rest/training"
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    bf16: bool = True
    gradient_checkpointing: bool = True

    # Saving
    save_steps: int = 500
    save_total_limit: int = 3
    models_dir: str = "../../models"


@dataclass
class ReSTrainingConfig:
    """Configuration for full ReST training loop."""

    # Number of ReST iterations
    num_iterations: int = 3

    # Learning rate schedule (decreasing per iteration)
    learning_rates: List[float] = field(default_factory=lambda: [2e-5, 1e-5, 5e-6])

    # Base paths
    data_path: str = ""
    audio_dir: str = ""
    output_base_dir: str = "./outputs/rest"
    models_dir: str = "../../models"

    # Model
    base_model: str = "Qwen/Qwen3-Omni-30B-A3B-Thinking"

    # Generation settings
    num_samples_per_problem: int = 16
    temperature: float = 0.9

    # Filter settings
    filter_strategy: str = "top_k"
    top_k: int = 3

    # vLLM settings
    vllm_base_url: str = "http://127.0.0.1:8901/v1"
