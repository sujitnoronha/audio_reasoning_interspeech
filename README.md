# Audio Reasoning - Interspeech 2025

Fine-tuning Qwen3-Omni for the MMAR (Multi-Modal Audio Reasoning) benchmark using ReST (Reinforced Self-Training) with LoRA adapters.

## Approach

We use **ReST (Reinforced Self-Training)** to iteratively improve a Qwen3-Omni-30B-A3B-Thinking model on audio reasoning tasks. The pipeline has three phases per iteration:

1. **Generate** — Sample K candidate answers per question using the current model
2. **Filter** — Evaluate candidates against ground truth, keep only correct solutions
3. **Train** — Supervised fine-tuning (SFT) on filtered correct solutions using LoRA

After multiple iterations, the model learns from its own correct reasoning traces, progressively improving on audio comprehension, counting, temporal analysis, and semantic understanding tasks.

### Key Design Choices

- **Model**: Qwen3-Omni-30B-A3B-Thinking (MoE architecture, 3B active parameters)
- **LoRA targets**: Attention layers (q/k/v/o_proj) + MoE expert layers (gate/up/down_proj)
- **LoRA config**: rank=64, alpha=128, dropout=0.05
- **Quantization**: 4-bit (NF4) during training, BF16 for inference
- **Inference**: Greedy decoding with merged adapter (`merge_and_unload`), Flash Attention 2
- **Prompt engineering**: Structured system prompts with counting guidance, anti-Yes bias, strict choice constraints (v8/v9 variants)

## Project Structure

```
audio_reasoning_interspeech/
├── src/
│   ├── data/                          # Data download and preprocessing
│   │   ├── MMAR-meta.json             # MMAR benchmark (1000 audio MCQ samples)
│   │   ├── download_MMAR.py           # Download MMAR dataset
│   │   ├── download_youtube_audio.py  # Download audio from YouTube
│   │   ├── combine_datasets.py        # Combine multiple training datasets
│   │   ├── countingqa_audioskills/     # CountingQA dataset scripts
│   │   ├── musicbench_audioskills/     # MusicBench dataset scripts
│   │   └── tacos_audioskills/          # TaCos dataset scripts
│   ├── inference/
│   │   ├── infer_single_model_baseline.py      # Baseline inference
│   │   ├── infer_single_model_finetuned.py     # Finetuned model inference
│   │   ├── infer_single_model_finetuned_v7.py  # V7 prompt variant
│   │   ├── infer_single_model_finetuned_v8.py  # V8: async prefetch + optimizations
│   │   ├── debug_inference.py                  # Debug/test inference with v9 prompt
│   │   └── wait_and_run.sh                     # GPU wait + launch script
│   ├── training/
│   │   ├── train_rest.py              # Standalone ReST training
│   │   ├── train_grpo.py              # GRPO training
│   │   ├── utils.py                   # Model loading, LoRA setup, merging
│   │   ├── data_utils.py              # Dataset utilities
│   │   └── rest/                      # Full ReST pipeline
│   │       ├── run_rest.py            # Orchestration script
│   │       ├── config.py              # Configuration
│   │       ├── generate.py            # Phase 1: Candidate generation
│   │       ├── filter.py              # Phase 2: Evaluation & filtering
│   │       ├── train.py               # Phase 3: SFT training
│   │       └── evaluate.py            # Evaluation script
│   ├── analyze_errors_by_category.py
│   └── validate_results_zip.py
├── docs/                              # Bug reports and analysis
├── notebooks/                         # Debug notebooks
└── requirements.txt
```

## Steps to Reproduce

### 1. Setup

```bash
pip install -r requirements.txt
```

### 2. Download Data

```bash
cd src/data

# Download MMAR benchmark dataset
python download_MMAR.py

# Download audio files
python download_youtube_audio.py

# (Optional) Download additional training datasets
python countingqa_audioskills/augment_countingqa_mcq.py
python musicbench_audioskills/download_musicbench.py
python tacos_audioskills/download_tacos.py

# Combine all training datasets
python combine_datasets.py
```

### 3. Start vLLM Server (for ReST generation phase)

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Thinking \
    --port 8901 \
    --host 127.0.0.1 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --allowed-local-media-path / \
    -tp 4
```

### 4. Run ReST Training Pipeline

```bash
cd src/training/rest

python run_rest.py \
    --data_path ../../data/combined_train.json \
    --audio_dir ../../data \
    --num_iterations 3 \
    --num_samples 16 \
    --filter_strategy top_k \
    --top_k 3
```

This runs 3 iterations of Generate → Filter → Train with decreasing learning rates [2e-5, 1e-5, 5e-6].

### 5. Run Inference on MMAR Benchmark

```bash
cd src/inference

python infer_single_model_finetuned_v8.py \
    --dataset_meta_path ../data/MMAR-meta.json \
    --dataset_audio_prefix ../data \
    --qwen3_omni_model_name_or_path Qwen/Qwen3-Omni-30B-A3B-Thinking \
    --adapter_path <path_to_lora_adapter> \
    --output_dir ../../outputs/v8_finetuned \
    --batch_size 1 \
    --do_sample False \
    --resume
```

### 6. Validate Submission

```bash
python validate_results_zip.py --zip_path ../../outputs/submission.zip
```

## Training Configuration

| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen3-Omni-30B-A3B-Thinking |
| LoRA rank | 64 |
| LoRA alpha | 128 |
| Quantization (training) | NF4 4-bit |
| Precision (inference) | BF16 |
| Batch size | 1 (gradient accumulation: 8) |
| ReST iterations | 3 |
| Samples per problem | 16 |
| Filter strategy | top_k (k=3) |
| Learning rates | [2e-5, 1e-5, 5e-6] |
| Optimizer | paged_adamw_8bit |
| Max new tokens | 6000 |
| Repetition penalty | 1.2 |

## Hardware Requirements

- **Training**: Single A100 80GB with 4-bit quantization + LoRA
- **Inference**: Single A100 80GB with BF16 merged model
- **vLLM serving**: 4x A100 80GB with tensor parallelism

## References

- [ReST: Reinforced Self-Training](https://arxiv.org/abs/2308.08998)
- [Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
