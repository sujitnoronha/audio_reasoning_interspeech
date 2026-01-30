# Audio Reasoning Interspeech

Fine-tuning Qwen3-Omni models for audio reasoning tasks using ReST (Reinforced Self-Training) and GRPO (Group Relative Policy Optimization).

## Project Structure

```
audio_reasoning_interspeech/
├── src/
│   ├── data/                     # Data download and preprocessing scripts
│   │   ├── download_MMAR.py
│   │   ├── download_youtube_audio.py
│   │   ├── countingqa_audioskills/
│   │   ├── musicbench_audioskills/
│   │   └── tacos_audioskills/
│   ├── inference/
│   │   └── infer_single_model_baseline.py
│   ├── training/
│   │   ├── train_rest.py         # Standalone ReST training script
│   │   ├── train_grpo.py         # GRPO training script
│   │   ├── utils.py              # Model loading, LoRA setup, merging
│   │   ├── data_utils.py         # Dataset utilities
│   │   └── rest/                 # Full ReST pipeline
│   │       ├── run_rest.py       # Orchestration script
│   │       ├── config.py         # Configuration dataclasses
│   │       ├── train.py          # Phase 3: SFT training
│   │       ├── generate.py       # Phase 1: Candidate generation
│   │       └── filter.py         # Phase 2: Evaluation & filtering
│   ├── analyze_errors_by_category.py
│   └── validate_results_zip.py
├── models/                       # Saved fine-tuned models
├── notebooks/                    # Jupyter notebooks
└── requirements.txt
```

## Training Methods

### ReST (Reinforced Self-Training)

ReST is an iterative self-improvement method that trains the model on its own correct solutions. It consists of three phases repeated over multiple iterations:

#### Phase 1: GENERATE
- Generate K candidate solutions per problem using the current model
- Uses vLLM for efficient inference with high temperature sampling (0.9)
- Default: 16 samples per problem

#### Phase 2: FILTER
- Evaluate all generated candidates against ground truth
- Extract answers using regex patterns
- Keep only correct solutions (configurable: all correct, best, or top-K)

#### Phase 3: TRAIN
- Supervised Fine-Tuning (SFT) on the filtered correct solutions
- Uses 4-bit quantization with LoRA PEFT for memory efficiency
- Decreasing learning rate schedule across iterations: [2e-5, 1e-5, 5e-6]

### GRPO (Group Relative Policy Optimization)

GRPO is a reinforcement learning method that optimizes the model using relative rewards within a group of generations:

- Generates multiple completions per prompt
- Computes rewards based on answer correctness
- Updates model using policy gradient with KL penalty
- Uses lower learning rate (1e-6) typical for RL fine-tuning

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Required packages:
# - torch
# - transformers
# - trl
# - peft
# - bitsandbytes
# - qwen-omni-utils
# - vllm (for ReST generation phase)
```

### Running ReST Training

1. **Start vLLM server** (in a separate terminal):
```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Thinking \
    --port 8901 \
    --host 127.0.0.1 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --allowed-local-media-path / \
    -tp 4
```

2. **Run the full ReST pipeline**:
```bash
cd src/training/rest

python run_rest.py \
    --data_path ../data/countingqa/CountingQA_MCQ.json \
    --audio_dir ../data/countingqa/counting_audios \
    --num_iterations 3 \
    --num_samples 16 \
    --filter_strategy top_k \
    --top_k 3
```

3. **Or run individual phases**:
```bash
# Phase 3 only (SFT on pre-filtered data)
python train.py \
    --train_data_path ./outputs/rest/filtered/filtered.json \
    --audio_dir ../data/countingqa/counting_audios \
    --output_dir ./outputs/rest/training \
    --learning_rate 2e-5 \
    --num_epochs 2
```

### Running GRPO Training

```bash
cd src/training

python train_grpo.py \
    --train_data_path ../data/countingqa/CountingQA_MCQ.json \
    --audio_dir ../data/countingqa/counting_audios \
    --output_dir ./outputs/grpo_training \
    --num_train_epochs 3 \
    --learning_rate 1e-6 \
    --num_generations 4
```

## Configuration Options

### Model Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name_or_path` | `Qwen/Qwen3-Omni-30B-A3B-Thinking` | Base model |
| `use_4bit` | `True` | Enable 4-bit quantization |
| `use_flash_attention` | `True` | Use Flash Attention 2 |

### LoRA Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora_r` | 64 | LoRA rank |
| `lora_alpha` | 128 | LoRA alpha scaling |
| `lora_dropout` | 0.05 | Dropout probability |

Target modules for Qwen3-Omni-MoE:
- Attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- MoE experts: `gate_proj`, `up_proj`, `down_proj`
- Shared expert: `shared_expert.gate_proj`, etc.

### Training Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_train_epochs` | 2-3 | Epochs per iteration |
| `per_device_train_batch_size` | 1 | Batch size |
| `gradient_accumulation_steps` | 8 | Effective batch = 8 |
| `learning_rate` | 2e-5 (SFT) / 1e-6 (RL) | Learning rate |
| `warmup_ratio` | 0.1 | Warmup proportion |
| `bf16` | True | Use bfloat16 precision |
| `gradient_checkpointing` | True | Memory optimization |
| `optim` | `paged_adamw_8bit` | 8-bit optimizer |

### ReST-Specific Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_iterations` | 3 | ReST iterations |
| `num_samples_per_problem` | 16 | Candidates per problem |
| `temperature` | 0.9 | Sampling temperature |
| `filter_strategy` | `top_k` | Filter method |
| `top_k` | 3 | Keep top K correct |
| `learning_rates` | [2e-5, 1e-5, 5e-6] | LR per iteration |

### GRPO-Specific Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_generations` | 4 | Generations per prompt |
| `beta` | 0.1 | KL penalty coefficient |
| `temperature` | 0.7 | Sampling temperature |

## Output Structure

After training, models are saved with timestamps:
```
models/
├── rest_Qwen_Qwen3_Omni_30B_A3B_Thinking_20250121_153500/
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer files...
└── grpo_Qwen_Qwen3_Omni_30B_A3B_Thinking_20250121_160000/
    └── ...
```

## Data Format

Training data should be JSON with the following structure:
```json
[
  {
    "id": "sample_001",
    "sound": "audio_file.wav",
    "question": "<sound> How many distinct sounds do you hear?",
    "answer": "(A) 3 sounds"
  }
]
```

Or conversation format:
```json
[
  {
    "sound": "audio.wav",
    "conversations": [
      {"from": "human", "value": "<sound> What instrument is playing?"},
      {"from": "gpt", "value": "(B) Piano"}
    ]
  }
]
```

## Hardware Requirements

- **GPU**: 4x A100 80GB recommended for full model
- **With 4-bit quantization + LoRA**: Single A100 40GB feasible
- **vLLM serving**: Tensor parallelism across 4 GPUs

## References

- [ReST: Reinforced Self-Training](https://arxiv.org/abs/2308.08998)
- [GRPO: Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
- [Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)





# Check if the wait loop or inference is running
ps aux | grep wait_and_run

# Check if the python inference is running
ps aux | grep infer_single_model_finetuned_v8

# Tail the log file (if you launched with nohup)
tail -f /home/ikulkar1/qwen_omni_finetune/audio_reasoning_interspeech/src/inference/wait_and_run.log
If you haven't launched it yet, run:


cd /home/ikulkar1/qwen_omni_finetune/audio_reasoning_interspeech/src/inference
nohup bash wait_and_run.sh > wait_and_run.log 2>&1 &
Then monitor with tail -f wait_and_run.log.