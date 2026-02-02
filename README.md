# Audio Reasoning - Interspeech 2026

Fine-tuning and prompt engineering for the MMAR (Multi-Modal Audio Reasoning) benchmark using Qwen3-Omni-30B-A3B-Thinking.

## Overview

This project explores multiple approaches to improve audio reasoning on the MMAR benchmark (1000 audio MCQ samples across 4 categories):

1. **Prompt Engineering** — Iterative prompt design (8 variants) targeting specific failure modes
2. **DSPy MIPROv2 Optimization** — Automated prompt optimization using DSPy's MIPROv2 optimizer
3. **ReST (Reinforced Self-Training)** — Iterative Generate → Filter → Train pipeline with LoRA adapters
4. **Sampling Experiments** — Temperature, max_tokens, and decoding strategy variations

## Prompt Engineering Experiments

All prompts are defined in `src/inference/infer_vllm_baseline.py` and served via vLLM (OpenAI-compatible API).

### Prompt Variant Summary

| Variant | Strategy | Key Idea |
|---------|----------|----------|
| baseline | Minimal | "Think step-by-step, answer must be correct" — no structured guidance |
| v8 | Expert analyst | Counting rules, emotion/sarcasm detection, comparison guidelines, strict output format |
| v9 | v8 + anti-overthinking | Added "Trust your first impression, do NOT second-guess" to reduce reasoning loops |
| v10 | Category-aware | Question-type detection with specialized strategies (counting, music, timing, spatial, emotions, anomaly, cultural, counterfactual) |
| v11 | v8 + targeted hints | v8 core + added guidance for weak subcategories (music/instruments, correlation/cause-effect, spatial) |
| v12 | Structured reasoning | HEARD → ANALYSIS → ANSWER format with anti-looping rules ("Do NOT loop back with Wait or Actually") |
| dspy_v1 | DSPy-optimized (simple) | "Listen to the audio and answer the multiple-choice question" |
| dspy_v2 | DSPy-optimized (cross-layer) | "Integrate Signal, Semantic, Perception, and Cultural layers. Use cross-layer reasoning." |

### Prompt Design Rationale

**v8** was the first major structured prompt. Error analysis revealed:
- Counting questions: model merged overlapping events → added "number each distinct occurrence (1, 2, 3...)"
- Emotion questions: model defaulted to positive → added "Sarcasm, frustration, and nervousness are common"
- Comparison questions: model used subjective preference → added "focus on clarity, smoothness, and technical quality"

**v9** addressed overthinking (reasoning loops where the model says "Wait, actually..." and reverses correct answers). Added explicit anti-revision instructions. However, this also suppressed legitimate self-correction, so accuracy did not improve.

**v10** tried a category-aware approach: detect the question type first, then apply a specialized analysis strategy. This covered 11 question types (counting, music, timing, environment, spatial, emotions, speaker identity, anomaly, audio quality, cultural, counterfactual). The prompt became long and the model sometimes misclassified the question type.

**v11** was a targeted fix — kept v8's proven core and added specific hints only for the weakest subcategories identified in error analysis: Music & Instruments, Correlation & Cause-Effect, and Spatial reasoning.

**v12** restructured the reasoning trace format itself. Instead of free-form thinking, it enforced:
1. **HEARD**: Describe exactly what you hear (sounds, speech, music, tone, background noise)
2. **ANALYSIS**: Connect observations to the question, evaluate each choice
3. **ANSWER**: State which choice best fits and why

This improved reasoning trace quality and reduced overthinking loops.

## DSPy MIPROv2 Optimization

Automated prompt optimization using DSPy's MIPROv2 optimizer (`src/inference/optimize_prompts_dspy.py`).

### Setup

- **Signature**: `AudioMCQ` with inputs: question, choices, category, sub_category, modality, language → output: answer
- **Module**: `AudioMCQModule` bridges DSPy optimization with vLLM audio inference calls
- **Data split**: Stratified train/eval split by category from MMAR-meta.json
- **Optimizer**: MIPROv2 with automatic instruction tuning

### DSPy Prompt Results

Two optimized prompts were generated:

- **dspy_v1**: Simple, direct instruction — "Listen to the audio and answer the multiple-choice question." Minimal system prompt with standard MCQ framing.
- **dspy_v2**: Cross-layer reasoning — "Integrate Signal, Semantic, Perception, and Cultural layers." Encourages the model to reason across audio analysis dimensions.

Optimized prompts are saved in `outputs/dspy_optimized/`:
- `optimized_prompts.json` / `optimized_prompts_module.json` (v1)
- `optimized_prompts_v2.json` / `optimized_prompts_v2_module.json` (v2)

## ReST (Reinforced Self-Training)

Iterative self-training pipeline to improve the model on its own correct reasoning traces.

### Pipeline

```
For each iteration:
  1. Generate — Sample 16 candidates per problem (temp=0.9) using vLLM async API
  2. Filter  — Evaluate against ground truth, categorize by difficulty
  3. Train   — SFT on filtered correct solutions using LoRA
```

### Learning Zone Framework

Problems are categorized by model accuracy across 16 samples:

| Category | Correct Rate | Training Action |
|----------|-------------|-----------------|
| too_hard | 0% (0/16) | Skip — no correct samples to learn from |
| challenging | 1–25% | Skip — too few correct samples |
| learning_zone | 26–75% | **Train** — optimal difficulty for learning |
| almost_mastered | 76–99% | Skip — diminishing returns |
| mastered | 100% (16/16) | Skip — already solved |

### ReST Generation Statistics

From the candidate generation phase (4,153 problems, 16 samples each):

| Metric | Value |
|--------|-------|
| Total candidates | 66,448 |
| Total correct | 43,182 (64.99%) |
| Problems with ≥1 correct | 3,669 / 4,153 |
| Learning zone problems | 889 (21.4%) |
| Training samples selected | 4,361 |

### ReST Configuration

| Parameter | Value |
|-----------|-------|
| Base model | Qwen3-Omni-30B-A3B-Thinking |
| Iterations | 3 |
| Learning rates | [2e-5, 1e-5, 5e-6] |
| Samples per problem | 16 |
| Generation temperature | 0.9 |
| Filter strategy | top_k (k=3) |
| LoRA rank / alpha | 64 / 128 |
| LoRA dropout | 0.05 |
| LoRA targets | q/k/v/o_proj + gate/up/down_proj (MoE experts) |
| Quantization (training) | NF4 4-bit |
| Optimizer | paged_adamw_8bit |
| Batch size | 1 (gradient accumulation: 8) |

### ReST Results

The finetuned model achieved ~65.3% local accuracy on MMAR, which was lower than the base model with optimized prompts (~67–72% depending on prompt variant). This suggests the training data (from auxiliary datasets) did not transfer well to the MMAR test distribution, or the LoRA capacity was insufficient.

Known issue: Transformers Qwen3OmniMoe Config has a `use_sliding_window` bug that requires a patch during training.

## Sampling Experiments

### Temperature

- **temp=0.1** (near-greedy): Used for all final submissions. Low temperature reduces variability and produces more deterministic answers.
- **temp=0.9**: Used during ReST candidate generation to maximize diversity.

### Max Tokens

- **Default (no limit)**: Full reasoning traces, sometimes exceeding 10,000 character limit for competition submission.
- **max_tokens=4096**: Constrained traces improved leaderboard reasoning quality scores (54.15 vs 52.55).
- **max_tokens=2000**: Used to rerun samples with naturally long traces (>9,000 chars) to keep them under the 10,000 char competition limit without post-hoc editing.

### Inference Details

- **vLLM server**: Qwen3-Omni-30B-A3B-Thinking served via vLLM on port 8901, OpenAI-compatible API, Single A100 80GB
- **Thinking traces**: vLLM returns thinking inside `<think>` tags within the `content` field (not in `reasoning_content`)
- **Answer extraction**: Regex-based extraction of `ANSWER: [choice]` from model output, with fuzzy matching to exact choices (stripped/case-insensitive)

## MMAR Benchmark Categories

| Category | Samples | Description |
|----------|---------|-------------|
| Semantic Layer | 412 | Language understanding, speaker intent, dialogue comprehension |
| Perception Layer | 404 | Counting, temporal analysis, spatial reasoning, audio quality |
| Cultural Layer | 141 | Music theory, regional accents, cultural context |
| Signal Layer | 43 | Audio signal properties, frequency, noise analysis |

### Per-Category Accuracy (v12 prompt, local GT)

Weakest subcategories identified through error analysis:
- Audio Difference Analysis: 25%
- Imagination: 50%
- Temporal Analysis: 53.6%
- Music Theory: 54%

Note: Local ground truth (MMAR-meta.json) is noisy and does not perfectly match the hidden test set. Some answers in the local GT are not even valid choices. Local accuracy is estimated to be ~1–2% pessimistic compared to the hidden test.

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
│   │   ├── infer_vllm_baseline.py            # vLLM inference with all prompt variants
│   │   ├── optimize_prompts_dspy.py          # DSPy MIPROv2 prompt optimization
│   │   ├── infer_single_model_baseline.py    # Baseline inference (local model)
│   │   ├── infer_single_model_finetuned.py   # Finetuned model inference
│   │   ├── infer_single_model_finetuned_v7.py
│   │   ├── infer_single_model_finetuned_v8.py
│   │   ├── debug_inference.py
│   │   ├── wait_and_run.sh
│   │   └── outputs/                   # Inference outputs per experiment
│   │       ├── vllm_v8_temp0.1/
│   │       ├── vllm_v8_temp0.1_max4k/
│   │       ├── vllm_v12_temp0.1/
│   │       └── v8_finetuned/
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
├── outputs/
│   └── dspy_optimized/                # DSPy optimized prompts
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

### 3. Start vLLM Server

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Thinking \
    --port 8901 \
    --host 127.0.0.1 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --allowed-local-media-path / \
    -tp 4
```

### 4. Run Inference (Prompt Engineering)

```bash
cd src/inference

# Run with any prompt variant: baseline, v8, v9, v10, v11, v12, dspy_v1, dspy_v2
python infer_vllm_baseline.py \
    --prompt v12 \
    --temperature 0.1 \
    --max_tokens 10000 \
    --output_dir outputs/vllm_v12_temp0.1
```

### 5. Run DSPy Prompt Optimization

```bash
cd src/inference

python optimize_prompts_dspy.py \
    --meta_path ../data/MMAR-meta.json \
    --audio_dir ../data \
    --output_dir ../../outputs/dspy_optimized
```

### 6. Run ReST Training Pipeline

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

### 7. Validate and Create Submission

```bash
python validate_results_zip.py \
    --prediction_path src/inference/outputs/vllm_v12_temp0.1/result.jsonl \
    --create_zip
```

Submission format: zip containing `result.jsonl` with fields `id`, `thinking_prediction` (< 10,000 chars), `answer_prediction`.

## Hardware Requirements

- **Training**: Single A100 80GB with 4-bit quantization + LoRA
- **Inference / vLLM serving**: Single A100 80GB

## References

- [ReST: Reinforced Self-Training](https://arxiv.org/abs/2308.08998)
- [Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [DSPy: Programming with Foundation Models](https://github.com/stanfordnlp/dspy)
- [MMAR Benchmark](https://mmar-benchmark.github.io/)
