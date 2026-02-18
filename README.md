# Structured Prompting vs. Self-Training for Audio Reasoning Under Limited Data and Compute

**Interspeech 2026**

When labeled data is scarce and compute budgets are tight, how should practitioners improve multimodal language models on audio reasoning tasks? We systematically compare three approaches on the [MMAR benchmark](https://mmar-benchmark.github.io/) (1,000 audio MCQs across four reasoning layers) using [Qwen3-Omni-30B-A3B-Thinking](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking):

1. **Structured Prompt Engineering** — Zero training cost
2. **Automated Prompt Optimization** (DSPy MIPROv2)
3. **Reinforced Self-Training** (ReST) with LoRA

## Key Results

| Approach | Accuracy (%) | &Delta; Baseline |
|----------|:---:|:---:|
| Baseline | 67.1 | --- |
| Baseline 4-bit | 65.7 | -1.4 |
| **Structured Prompting** | | |
| &nbsp;&nbsp;Expert Analyst | 68.3 | +1.2 |
| &nbsp;&nbsp;Category-Aware | 68.1 | +1.0 |
| &nbsp;&nbsp;Targeted Hints | 67.4 | +0.3 |
| &nbsp;&nbsp;Structured Reasoning | **72.0** | **+4.9** |
| **Automated Optimization** | | |
| &nbsp;&nbsp;MIPROv2 Optimized | 63.3 | -3.8 |
| **Self-Training (4-bit)** | | |
| &nbsp;&nbsp;Full LoRA + basic | 64.7 | -2.4 |
| &nbsp;&nbsp;Full LoRA + Structured Reasoning | 65.5 | -1.6 |

**Structured Reasoning achieves the best accuracy (+4.9%) at zero training cost**, while ReST self-training degrades performance by -2.4%. Applying the Structured Reasoning prompt to the ReST-trained model yields only 65.5%, confirming that fine-tuning damage cannot be reversed through prompting.

## Key Findings

- **Structured prompting improves 13 of 16 subcategories**, with the largest gains on Correlation Analysis (+12.0%) and Counting (+13.1%)
- **ReST self-training regresses on 10 of 16 subcategories** — catastrophic forgetting outweighs learning zone gains
- **Restructuring reasoning format outperforms adding domain rules** — the HEARD &rarr; ANALYSIS &rarr; ANSWER pipeline (+4.9%) beats explicit rules like counting guidelines (+1.2%) or targeted hints (+0.3%)
- **Automated optimization (MIPROv2) underperforms baseline** — generic prompts lack the error-correction strategies discovered through manual analysis

## Methodology

### Structured Prompt Engineering

Prompts were developed through iterative error analysis at zero training cost:

| Prompt | Strategy | Key Idea |
|--------|----------|----------|
| Expert Analyst | Domain rules | Counting rules, emotion/sarcasm detection, comparison guidelines |
| Category-Aware | Question-type detection | Classify question type first, then apply specialized strategy |
| Targeted Hints | Subcategory guidance | Targeted hints only for weakest subcategories (music, correlation, spatial) |
| Structured Reasoning | Reasoning format | HEARD &rarr; ANALYSIS &rarr; ANSWER with anti-looping constraint |

### Reinforced Self-Training (ReST)

Iterative Generate &rarr; Filter &rarr; Train pipeline:

1. **Generate**: 16 diverse reasoning traces per problem (temp=0.9)
2. **Filter**: Learning zone framework — train only on problems with 26-75% success rate
3. **Train**: LoRA fine-tuning (rank=64, alpha=128, 4-bit NF4) targeting attention + MoE expert layers

From 4,153 training problems, only 889 (21.4%) fell in the learning zone, yielding 4,361 training samples — insufficient for a 30B model.

### DSPy MIPROv2

Automated prompt optimization using meta-learning over prompt variations with stratified train/eval split from MMAR.

## MMAR Benchmark

| Category | Samples | Description |
|----------|:---:|-------------|
| Semantic Layer | 412 | Language understanding, speaker intent, dialogue comprehension |
| Perception Layer | 404 | Counting, temporal analysis, spatial reasoning |
| Cultural Layer | 141 | Music theory, regional accents, cultural context |
| Signal Layer | 43 | Audio signal properties, frequency, noise analysis |

## Project Structure

```
audio_reasoning_interspeech/
├── docs/                              # Paper (main.tex, mybib.bib)
├── src/
│   ├── data/                          # Data download and preprocessing
│   │   ├── MMAR-meta.json             # MMAR benchmark (1000 audio MCQ samples)
│   │   ├── download_MMAR.py           # Download MMAR dataset
│   │   ├── download_youtube_audio.py  # Download audio from YouTube
│   │   ├── combine_datasets.py        # Combine training datasets
│   │   ├── countingqa_audioskills/     # CountingQA dataset
│   │   ├── musicbench_audioskills/     # MusicBench dataset
│   │   └── tacos_audioskills/          # TaCos dataset
│   ├── inference/
│   │   ├── infer_vllm_baseline.py            # vLLM inference (all prompt variants)
│   │   ├── infer_rest_with_prompts.py        # ReST model inference with prompt switching
│   │   ├── infer_single_model_finetuned.py   # Finetuned model inference
│   │   ├── infer_single_model_baseline.py    # Baseline inference
│   │   └── optimize_prompts_dspy.py          # DSPy MIPROv2 optimization
│   ├── training/
│   │   ├── train_rest.py              # Standalone ReST training
│   │   ├── utils.py                   # Model loading, LoRA setup, merging
│   │   ├── data_utils.py              # Dataset utilities
│   │   └── rest/                      # Full ReST pipeline
│   │       ├── run_rest.py            # Orchestration script
│   │       ├── config.py              # Configuration
│   │       ├── generate.py            # Phase 1: Candidate generation
│   │       ├── filter.py              # Phase 2: Evaluation & filtering
│   │       ├── train.py               # Phase 3: SFT training
│   │       └── evaluate.py            # Evaluation script
│   └── analyze_errors_by_category.py
└── requirements.txt
```

## Reproduction

### 1. Setup

```bash
pip install -r requirements.txt
```

### 2. Download Data

```bash
cd src/data
python download_MMAR.py
python download_youtube_audio.py

# (Optional) Auxiliary training datasets for ReST
python countingqa_audioskills/augment_countingqa_mcq.py
python musicbench_audioskills/download_musicbench.py
python tacos_audioskills/download_tacos.py
python combine_datasets.py
```

### 3. Prompt Engineering Experiments (vLLM)

```bash
# Start vLLM server
vllm serve Qwen/Qwen3-Omni-30B-A3B-Thinking \
    --port 8901 --host 127.0.0.1 --dtype bfloat16 \
    --max-model-len 32768 --allowed-local-media-path / -tp 4

# Run with any prompt variant: baseline, v8, v9, v10, v11, v12, dspy_v1, dspy_v2
cd src/inference
python infer_vllm_baseline.py \
    --prompt v12 --temperature 0.1 --max_tokens 10000 \
    --output_dir outputs/vllm_v12_temp0.1
```

### 4. ReST Training

```bash
cd src/training/rest
python run_rest.py \
    --data_path ../../data/combined_train.json \
    --audio_dir ../../data \
    --num_iterations 3 --num_samples 16 \
    --filter_strategy top_k --top_k 3
```

### 5. ReST Inference with Prompt Switching

```bash
cd src/inference
python infer_rest_with_prompts.py \
    --adapter_path <path_to_adapter> \
    --qwen3_omni_model_name_or_path Qwen/Qwen3-Omni-30B-A3B-Thinking \
    --dataset_meta_path ../data/MMAR-meta.json \
    --dataset_audio_prefix ../data \
    --output_dir ../../outputs_rest \
    --prompt_type v12 --use_4bit --merge_adapter --batch_size 1
```

## Hardware

- **Training**: Single NVIDIA A100 80GB (4-bit quantization + LoRA)
- **Inference (vLLM)**: Single A100 80GB with tensor parallelism=4

---

## ReST Fine-Tuning Details

### Pipeline

```
For each iteration:
  1. Generate — Sample 16 candidates per problem (temp=0.9) using vLLM async API
  2. Filter  — Evaluate against ground truth, categorize by difficulty
  3. Train   — SFT on filtered correct solutions using LoRA
```

### Learning Zone Framework

Problems are categorized by model accuracy across 16 samples to select training data at the optimal difficulty level:

| Category | Correct Rate | Training Action |
|----------|-------------|-----------------|
| too_hard | 0% (0/16) | Skip — no correct samples to learn from |
| challenging | 1-25% | Skip — too few correct samples |
| learning_zone | 26-75% | **Train** — optimal difficulty for learning |
| almost_mastered | 76-99% | Skip — diminishing returns |
| mastered | 100% (16/16) | Skip — already solved |

### Generation Statistics

From the candidate generation phase (4,153 problems, 16 samples each):

| Metric | Value |
|--------|-------|
| Total candidates | 66,448 |
| Total correct | 43,182 (65.0%) |
| Problems with >=1 correct | 3,669 / 4,153 |
| Learning zone problems | 889 (21.4%) |
| Training samples selected | 4,361 |

### Training Configuration

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

### ReST Results and Insights

The ReST pipeline achieved 64.7% on MMAR with a basic prompt, below the 67.1% baseline. Key observations:

- **Limited learning zone coverage**: Only 21.4% of problems fell in the learning zone, yielding 4,361 training samples — insufficient for a 30B parameter model across 16 diverse subcategories.
- **Distribution mismatch**: Auxiliary datasets (CountingQA, MusicBench, TaCos) showed limited transfer to MMAR's question distribution. The largest regression (Music Theory: -15.4%) occurred in a domain covered by MusicBench, suggesting distribution mismatch actively harms performance.
- **Catastrophic forgetting**: ReST improved only 6 of 16 subcategories while degrading 10. Fine-tuning biased the model toward specific reasoning patterns at the expense of general capabilities.
- **Prompting cannot rescue fine-tuning damage**: Applying the Structured Reasoning prompt to the ReST model yields 65.5% — a marginal gain over basic (64.7%) but far below the base model with the same prompt (72.0%). The 6.5% gap suggests fine-tuning degraded the model's ability to follow structured reasoning instructions.

### Error-Driven Prompt Refinement

The prompt engineering effort was integrated with the ReST pipeline. Each prompt iteration was informed by error analysis from ReST evaluation phases:

| Variant | Strategy | Key Idea |
|---------|----------|----------|
| baseline | Minimal | "Think step-by-step, answer must be correct" — no structured guidance |
| v8 (Expert Analyst) | Expert analyst | Counting rules, emotion/sarcasm detection, comparison guidelines, strict output format |
| v9 | v8 + anti-overthinking | Added "Trust your first impression, do NOT second-guess" to reduce reasoning loops |
| v10 (Category-Aware) | Category-aware | Question-type detection with specialized strategies per type |
| v11 (Targeted Hints) | v8 + targeted hints | v8 core + guidance for weak subcategories (music, correlation, spatial) |
| v12 (Structured Reasoning) | Structured reasoning | HEARD &rarr; ANALYSIS &rarr; ANSWER format with anti-looping rules |

**v8** addressed counting errors (model merged overlapping events), emotion misidentification (defaulted to positive), and subjective comparisons.

**v10** introduced category-aware reasoning but the long prompt sometimes caused question-type misclassification.

**v12** restructured the reasoning format itself rather than adding more rules — imposing a strict three-step structure with anti-looping constraints ("Do NOT loop back with Wait or Actually"). This produced the best results (+4.9%).

### Inference Details

- **vLLM server**: Qwen3-Omni served via vLLM on port 8901, OpenAI-compatible API, single A100 80GB
- **Thinking traces**: vLLM returns thinking inside `<think>` tags within the `content` field
- **Answer extraction**: Regex-based extraction of `ANSWER: [choice]` from model output, with fuzzy matching
- **Temperature**: 0.1 (near-greedy) for all final evaluations; 0.9 for ReST candidate generation

Known issue: Transformers Qwen3OmniMoe Config has a `use_sliding_window` bug that requires a patch during training.

## References

- [ReST: Reinforced Self-Training](https://arxiv.org/abs/2308.08998)
- [Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [DSPy: Programming with Foundation Models](https://github.com/stanfordnlp/dspy)
- [MMAR Benchmark](https://mmar-benchmark.github.io/)
