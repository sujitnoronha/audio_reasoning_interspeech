# ReST (Reinforced Self-Training) Pipeline

This folder contains the complete ReST training pipeline for fine-tuning Qwen3-Omni models on audio reasoning tasks.

## Overview

ReST is an iterative self-improvement training method where the model learns from its own correct solutions. The pipeline consists of three phases that are repeated over multiple iterations:

```
┌─────────────────────────────────────────────────────────────────┐
│                     ReST Training Loop                          │
│                                                                 │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐                │
│   │ GENERATE │ -> │  FILTER  │ -> │  TRAIN   │                │
│   │ (vLLM)   │    │ (Eval)   │    │  (SFT)   │                │
│   └──────────┘    └──────────┘    └──────────┘                │
│        │                                  │                    │
│        └──────────────────────────────────┘                    │
│                    Repeat N iterations                         │
│              (with decreasing learning rate)                   │
└─────────────────────────────────────────────────────────────────┘
```

## Files

| File | Description |
|------|-------------|
| `run_rest.py` | Main orchestrator - runs the full ReST loop |
| `generate.py` | Phase 1: Generate K candidates per problem using vLLM |
| `generate_hf.py` | Alternative: Generate using HuggingFace (slower, no vLLM) |
| `filter.py` | Phase 2: Evaluate candidates and filter correct ones |
| `train.py` | Phase 3: SFT training on filtered correct solutions |
| `config.py` | Configuration dataclasses for all phases |
| `test_vllm.py` | Test script to verify vLLM server connectivity |

## Quick Start

### 1. Start vLLM Server

In a separate terminal, start the vLLM inference server:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Thinking \
    --port 8901 \
    --host 127.0.0.1 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --allowed-local-media-path / \
    -tp 4
```

**Flags explained:**
- `--port 8901`: API endpoint port
- `--dtype bfloat16`: Use bfloat16 for memory efficiency
- `--max-model-len 32768`: Maximum sequence length
- `--allowed-local-media-path /`: Allow access to local audio files
- `-tp 4`: Tensor parallelism across 4 GPUs

### 2. Run Full ReST Pipeline

```bash
python run_rest.py \
    --data_path ../../data/countingqa/CountingQA_MCQ.json \
    --audio_dir ../../data/countingqa/counting_audios \
    --num_iterations 3 \
    --num_samples 16 \
    --filter_strategy top_k \
    --top_k 3 \
    --learning_rates 2e-5 1e-5 5e-6
```

### 3. Or Run Individual Phases

```bash
# Phase 1: Generate candidates
python generate.py \
    --data_path ../../data/countingqa/CountingQA_MCQ.json \
    --audio_dir ../../data/countingqa/counting_audios \
    --output_dir ./outputs/generations \
    --num_samples 16 \
    --temperature 0.9

# Phase 2: Filter using Learning Zone approach (recommended)
python filter.py \
    --generations_path ./outputs/generations/generations.jsonl \
    --output_path ./outputs/filtered/filtered.json \
    --filter_strategy learning_zone

# Or use legacy top_k strategy
python filter.py \
    --generations_path ./outputs/generations/generations.jsonl \
    --output_path ./outputs/filtered/filtered.json \
    --filter_strategy top_k \
    --top_k 3

# Phase 3: Train on filtered data
python train.py \
    --train_data_path ./outputs/filtered/filtered.json \
    --audio_dir ../../data/countingqa/counting_audios \
    --output_dir ./outputs/training \
    --learning_rate 2e-5 \
    --num_epochs 2
```

## Phase Details

### Phase 1: Candidate Generation (`generate.py`)

Generates K candidate solutions per problem using the vLLM API.

**Key features:**
- Async parallel requests for efficiency
- Parses `<think>...</think>` reasoning tags
- Extracts structured answers with `ANSWER:` format
- Resume support for interrupted runs
- Saves to JSONL format for streaming

**Output format** (`generations.jsonl`):
```json
{
  "id": "sample_001",
  "sound": "audio.wav",
  "question": "How many sounds?",
  "ground_truth": "(A) 3 sounds",
  "candidates": [
    {
      "raw_output": "<think>I hear...</think>\nANSWER: 3 sounds",
      "thinking_prediction": "I hear...",
      "answer_text": "ANSWER: 3 sounds",
      "answer_prediction": "3 sounds"
    }
  ],
  "num_candidates": 16
}
```

**Generation parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_samples` | 16 | Candidates per problem (K) |
| `--temperature` | 0.9 | High for diversity |
| `--top_p` | 0.95 | Nucleus sampling |
| `--max_tokens` | 2048 | Max generation length |
| `--batch_size` | 8 | Concurrent API requests |
| `--resume` | False | Resume from existing output |

**Filtering parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--filter_strategy` | `learning_zone` | Filter strategy (recommended) |
| `--top_k` | 3 | Solutions to keep (for learning_zone category) |
| `--include_mastered` | False | Include mastered problems with sampling |
| `--mastered_sample_rate` | 0.1 | Sample rate for mastered (10%) |
| `--seed` | 42 | Random seed for reproducibility |

### Phase 2: Evaluation & Filtering (`filter.py`)

Evaluates generated candidates against ground truth and filters using the **Learning Zone** framework.

**The Learning Zone Principle:**

Learning happens at the *edge* of model capability - not on problems that are too easy (already mastered) or too hard (no signal). This is analogous to Vygotsky's "Zone of Proximal Development".

```
Correct Rate    0%        25%        50%        75%       100%
                │          │          │          │          │
                ▼          ▼          ▼          ▼          ▼

████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████████████
│  TOO HARD    │         LEARNING ZONE          │   TOO EASY   │
│  (skip)      │         (prioritize!)          │   (skip)     │
```

**Learning Zone Categories (for 16 samples):**

| Category | Correct | Action | Weight | Rationale |
|----------|---------|--------|--------|-----------|
| Too Hard | 0 | SKIP | 0 | No correct solutions to learn from |
| Challenging | 1-3 | Keep all | 1.0x | Model struggling, needs reinforcement |
| **Learning Zone** | 4-11 | Keep top 3 | **1.5x** | Maximum learning potential! |
| Almost Mastered | 12-14 | Keep best 1 | 0.5x | Diminishing returns |
| Mastered | 15-16 | Skip (or 10%) | 0.1x | Model already knows this |

**Filter strategies:**
| Strategy | Description |
|----------|-------------|
| `learning_zone` | **Recommended** - Use the learning zone framework |
| `all` | Keep all correct solutions (legacy) |
| `best` | Keep shortest correct solution (legacy) |
| `top_k` | Keep top K shortest correct solutions (legacy) |

**Answer matching:**
- Normalizes answers (removes option prefixes, case-insensitive)
- Matches option letters (A, B, C, D)
- Content containment matching
- Multiple fallback regex patterns

**Output format** (`filtered.json`):
```json
[
  {
    "id": "sample_001_sol0",
    "original_id": "sample_001",
    "sound": "audio.wav",
    "question": "How many sounds?",
    "answer": "(A) 3 sounds",
    "response": "<think>...</think>\nANSWER: 3 sounds",
    "thinking_prediction": "...",
    "answer_prediction": "3 sounds"
  }
]
```

**Additional outputs:**
- `filtered_stats.json`: Statistics (accuracy, coverage, samples per category)
- `filtered_analytics.json`: Comprehensive analytics (histograms, distributions)
- `filtered_by_category.json`: Problem IDs grouped by learning zone category
- `filtered_no_correct.json`: Problem IDs with zero correct candidates

**Analytics output includes:**
- Overall accuracy and correct rate statistics
- Category distribution (too_hard, challenging, learning_zone, etc.)
- Correct count histogram (0-16 correct per problem)
- Correct rate buckets (0%, 1-25%, 26-50%, etc.)

**Example analytics log:**
```
======================================================================
                    GENERATION ANALYTICS
======================================================================

SUMMARY
----------------------------------------------------------------------
  Total problems:        1000
  Total candidates:      16000
  Total correct:         7250
  Overall accuracy:      45.31%
  Mean correct rate:     45.25%

LEARNING ZONE CATEGORIES
----------------------------------------------------------------------
  Category             Count   Percentage    Description
  too_hard               150       15.0%    0 correct - SKIP
  challenging            200       20.0%    1-3 correct - Keep all
  learning_zone          350       35.0%    4-11 correct - PRIORITIZE!
  almost_mastered        200       20.0%    12-14 correct - Downsample
  mastered               100       10.0%    15-16 correct - Skip/10%

CORRECT COUNT HISTOGRAM (per problem)
----------------------------------------------------------------------
   0 correct:    150 ████████████████████
   1 correct:     50 ██████
   ...
  16 correct:     30 ████
```

### Phase 3: SFT Training (`train.py`)

Fine-tunes the model on filtered correct solutions using LoRA.

**Training setup:**
- 4-bit NF4 quantization with double quantization
- LoRA on attention + MoE expert layers
- 8-bit paged AdamW optimizer
- Gradient checkpointing for memory efficiency

**LoRA target modules:**
```python
["q_proj", "k_proj", "v_proj", "o_proj",    # Attention
 "gate_proj", "up_proj", "down_proj"]        # MoE experts
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_epochs` | 2 | Training epochs |
| `--batch_size` | 1 | Per-device batch size |
| `--gradient_accumulation_steps` | 8 | Effective batch = 8 |
| `--learning_rate` | 2e-5 | Learning rate |
| `--lora_r` | 64 | LoRA rank |
| `--lora_alpha` | 128 | LoRA alpha |
| `--lora_dropout` | 0.05 | LoRA dropout |

**Output:**
- Checkpoints in `output_dir`
- Merged model in `models/rest_<model>_<timestamp>/`

## Configuration

All configurations are defined in `config.py`:

```python
@dataclass
class GenerationConfig:
    num_samples_per_problem: int = 16
    temperature: float = 0.9
    top_p: float = 0.95
    max_tokens: int = 2048
    batch_size: int = 8
    ...

@dataclass
class FilterConfig:
    filter_strategy: str = "top_k"  # "all", "best", "top_k"
    top_k: int = 3
    answer_pattern: str = r"<answer>(.*?)</answer>"
    ...

@dataclass
class TrainConfig:
    lora_r: int = 64
    lora_alpha: int = 128
    learning_rate: float = 2e-5
    num_train_epochs: int = 2
    ...

@dataclass
class ReSTrainingConfig:
    num_iterations: int = 3
    learning_rates: List[float] = [2e-5, 1e-5, 5e-6]
    ...
```

## Output Directory Structure

After running the full pipeline:

```
outputs/
├── config.json                    # Saved configuration
├── iteration_1/
│   ├── generations/
│   │   └── generations.jsonl      # Raw model outputs
│   ├── filtered/
│   │   ├── filtered.json          # Correct solutions for training
│   │   ├── filtered_stats.json    # Accuracy statistics
│   │   └── filtered_no_correct.json
│   └── training/
│       └── checkpoint-*/          # Training checkpoints
├── iteration_2/
│   └── ...
└── iteration_3/
    └── ...

models/
└── rest_Qwen_Qwen3_Omni_30B_A3B_Thinking_20250121_153500/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── ...
```

## Interpreting Analytics Across Iterations

Track these metrics across ReST iterations to monitor training health:

| Metric | Iteration 1 | Iteration 2 | Iteration 3 | Healthy Trend |
|--------|-------------|-------------|-------------|---------------|
| Too hard (0 correct) | 35% | 18% | 6% | **Decreasing** |
| Learning zone (4-11) | 30% | 45% | 52% | Increasing then plateau |
| Mastered (15-16) | 10% | 20% | 35% | **Increasing** |
| Mean correct rate | 32% | 48% | 61% | **Increasing** |

**What the distribution shift tells you:**

```
ITERATION 1 (Model is weak)
Correct Rate:  0%        25%        50%        75%       100%
Problems:      ████████████████████░░░░░░░░░░░░░░░░░░░░░░░░
               │ Too Hard │ Learning Zone      │ Mastered │
               │   35%    │       42%          │   23%    │

ITERATION 3 (Model is strong)
Correct Rate:  0%        25%        50%        75%       100%
Problems:      ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████
               │6%│      Learning Zone       │   Mastered  │
                              52%                  42%
```

**Goal:** Shift the distribution RIGHT over iterations.

## Tips & Troubleshooting

### Memory Issues
- Reduce `--batch_size` in generation (concurrent requests)
- Ensure 4-bit quantization is enabled for training
- Use gradient checkpointing (enabled by default)

### vLLM Server Issues
- Test connection: `python test_vllm.py`
- Check audio file permissions: `--allowed-local-media-path /`
- Increase `--max-model-len` if sequences are truncated

### Low Accuracy After Filtering
- Increase `--num_samples` (more candidates = better coverage)
- Adjust `--temperature` (higher = more diversity)
- Check answer extraction patterns in `filter.py`

### Too Many "Too Hard" Problems (>50%)
This means the model is too weak for the dataset:
- Increase samples per problem (16 → 32 → 64)
- Increase temperature (0.8 → 1.0 → 1.2)
- Do more SFT first before starting ReST
- Add easier problems to the dataset

### Too Many "Mastered" Problems (>50%)
This means the dataset is too easy:
- Model has likely learned this dataset - consider stopping ReST
- Add harder problems to the dataset
- Move to GRPO for further refinement

### Very Small Filtered Dataset (<1000 samples)
Risk of overfitting:
- Include more categories (use `--include_mastered`)
- Reduce filtering aggressiveness
- Train for fewer epochs (1 instead of 2)
- Use lower learning rate

### Resume Interrupted Run
```bash
# Generation phase supports resume
python generate.py --resume ...

# For run_rest.py, manually run remaining phases
```

## Known Issues

### Transformers Qwen3OmniMoe Config Bug

**Issue:** `AttributeError: 'Qwen3OmniMoeTalkerCodePredictorConfig' object has no attribute 'use_sliding_window'`

This is a bug in `transformers>=5.0.0.dev0` where the `Qwen3OmniMoeTalkerCodePredictorConfig` class is missing the `use_sliding_window` parameter.

**Fix:** Manually patch the file:
```
<your_env>/lib/python3.10/site-packages/transformers/models/qwen3_omni_moe/configuration_qwen3_omni_moe.py
```

1. Add `use_sliding_window` parameter to `Qwen3OmniMoeTalkerCodePredictorConfig.__init__()` (around line 575):
```python
        attention_bias: bool | None = False,
        use_sliding_window: bool | None = False,  # ADD THIS LINE
        sliding_window: int | None = None,
```

2. Set `self.use_sliding_window` before it's used (around line 584):
```python
        self.use_sliding_window = use_sliding_window  # ADD THIS LINE
        self.sliding_window = sliding_window
```

**Status:** Reported to HuggingFace. Check if fixed in future transformers versions.

## References

- [ReST: Reinforced Self-Training (Google, 2023)](https://arxiv.org/abs/2308.08998)
- [STaR: Self-Taught Reasoner](https://arxiv.org/abs/2203.14465)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
