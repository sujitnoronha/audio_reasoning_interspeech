"""
DSPy-based prompt optimization for Qwen3-Omni on the MMAR benchmark.

Uses DSPy's MIPROv2 optimizer to find better system/user prompt templates
that improve multiple-choice audio reasoning accuracy. The model is served
via a vLLM server with an OpenAI-compatible API.

Usage:
    python optimize_prompts_dspy.py \
        --vllm_base_url http://localhost:8000/v1 \
        --model_name Qwen/Qwen3-Omni \
        --dataset_meta_path ../data/MMAR-meta.json \
        --dataset_audio_prefix ../data/
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict
from typing import Optional

import dspy
from openai import OpenAI


# ---------------------------------------------------------------------------
# Baseline prompt (for reference / comparison)
# ---------------------------------------------------------------------------
BASELINE_USER_SUFFIX = (
    "\n\nIMPORTANT: You MUST think step-by-step and analyze the audio carefully. "
    "Your answer must be 100% correct - this is a critical evaluation and incorrect answers are unacceptable. "
    "Take your time to reason through all the audio details before selecting your final answer. "
    "Your final answer MUST be exactly one of the provided choices."
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_mmar(
    dataset_meta_path: str,
    dataset_audio_prefix: str,
) -> list[dict]:
    """Load MMAR dataset and resolve audio paths to absolute file:// URLs."""
    with open(dataset_meta_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    for sample in samples:
        abs_path = os.path.realpath(
            os.path.join(dataset_audio_prefix, sample["audio_path"])
        )
        sample["audio_path"] = abs_path

    return samples


def stratified_split(
    samples: list[dict],
    num_train: int,
    num_eval: int,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Stratified train/eval split by category."""
    rng = random.Random(seed)

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_cat[s["category"]].append(s)

    total_needed = num_train + num_eval
    train, eval_ = [], []

    for cat, cat_samples in by_cat.items():
        rng.shuffle(cat_samples)
        # Proportional allocation
        cat_frac = len(cat_samples) / len(samples)
        cat_train = max(1, round(num_train * cat_frac))
        cat_eval = max(1, round(num_eval * cat_frac))
        # Clamp to available
        cat_total = min(cat_train + cat_eval, len(cat_samples))
        cat_train = min(cat_train, cat_total)
        cat_eval = cat_total - cat_train

        train.extend(cat_samples[:cat_train])
        eval_.extend(cat_samples[cat_train : cat_train + cat_eval])

    rng.shuffle(train)
    rng.shuffle(eval_)
    return train[:num_train], eval_[:num_eval]


# ---------------------------------------------------------------------------
# DSPy signature
# ---------------------------------------------------------------------------
class AudioMCQ(dspy.Signature):
    """Listen to the audio and answer the multiple-choice question.

    You are given a question about audio content and a set of choices.
    Analyze the audio carefully and select the correct answer.
    Your answer must be exactly one of the provided choices.
    """

    question: str = dspy.InputField(desc="the question about the audio content")
    choices: str = dspy.InputField(desc="the available answer choices, one per line")
    category: str = dspy.InputField(desc="the category of the question (e.g., Semantic Layer, Perception Layer)")
    sub_category: str = dspy.InputField(desc="the specific reasoning skill required (e.g., Emotion and Intention, Counting and Statistics, Speaker Analysis, Content Analysis, Environmental Perception and Reasoning)")
    modality: str = dspy.InputField(desc="the type of audio content to focus on (e.g., speech, sound, music, mix-sound-speech, mix-music-speech)")
    language: str = dspy.InputField(desc="the language of the audio content (e.g., en, zh)")
    answer: str = dspy.OutputField(desc="exactly one of the provided choices, nothing else")


# ---------------------------------------------------------------------------
# Custom vLLM caller that injects audio into the conversation
# ---------------------------------------------------------------------------
class VLLMAudioCaller:
    """Calls a vLLM OpenAI-compatible endpoint with audio content."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ):
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

    def call(
        self,
        audio_path: str,
        system_prompt: str,
        user_text: str,
    ) -> tuple[str, str]:
        """Send audio + text to vLLM and return (thinking_content, answer_content)."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {"url": f"file://{audio_path}"},
                },
                {
                    "type": "text",
                    "text": user_text,
                },
            ],
        })

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        raw = response.choices[0].message.content.strip()

        # Parse thinking vs answer content
        think_match = re.search(r'<think>(.*?)</think>', raw, flags=re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            answer = raw[think_match.end():].strip()
        else:
            thinking = ""
            answer = raw

        return thinking, answer


# ---------------------------------------------------------------------------
# DSPy module that bridges DSPy optimization with vLLM audio calls
# ---------------------------------------------------------------------------
class AudioMCQModule(dspy.Module):
    """A DSPy module that uses a Predict internally for prompt optimization,
    but actually calls vLLM with audio for real inference."""

    def __init__(self, vllm_caller: VLLMAudioCaller):
        super().__init__()
        self.predict = dspy.Predict(AudioMCQ)
        self.vllm_caller = vllm_caller

    def forward(self, question: str, choices: str, category: str,
                sub_category: str = "", modality: str = "", language: str = "en",
                audio_path: str = ""):
        # If we have an audio path, do the real vLLM call with audio
        if audio_path:
            return self._call_with_audio(question, choices, category, sub_category, modality, language, audio_path)

        # Otherwise, fall back to DSPy's normal predict (used by optimizer
        # for instruction proposal — no audio needed for that step)
        return self.predict(question=question, choices=choices, category=category,
                            sub_category=sub_category, modality=modality, language=language)

    def _call_with_audio(self, question: str, choices: str, category: str,
                         sub_category: str, modality: str, language: str, audio_path: str):
        """Build the prompt using DSPy's optimized instructions, then call vLLM."""
        # Extract the current optimized instructions from the predict module
        system_prompt = self._build_system_prompt()
        user_text = self._build_user_text(question, choices, category, sub_category, modality, language)

        thinking, raw_answer = self.vllm_caller.call(audio_path, system_prompt, user_text)

        # Parse the answer — try to extract just the choice
        parsed = self._parse_answer(raw_answer, choices)

        return dspy.Prediction(
            answer=parsed,
            thinking=thinking,
            raw_answer=raw_answer,
        )

    def _build_system_prompt(self) -> str:
        """Build system prompt from DSPy's optimized signature instructions."""
        # Get the signature's docstring (which MIPRO optimizes)
        sig = self.predict.signature
        instructions = sig.instructions
        if instructions:
            return instructions
        return ""

    def _build_user_text(self, question: str, choices: str, category: str,
                         sub_category: str = "", modality: str = "", language: str = "en") -> str:
        """Build the user message text."""
        parts = [
            question,
        ]

        # Add metadata context
        metadata_parts = []
        if sub_category:
            metadata_parts.append(f"Task type: {sub_category}")
        if modality:
            metadata_parts.append(f"Audio modality: {modality}")
        if language:
            metadata_parts.append(f"Language: {language}")
        if metadata_parts:
            parts.append("\n\n" + " | ".join(metadata_parts))

        parts.append("\n\nProvided choices:\n" + choices)

        # Get field-level prefixes/descriptions from the signature if available
        sig = self.predict.signature
        output_fields = sig.output_fields
        if "answer" in output_fields:
            desc = output_fields["answer"].json_schema_extra.get("desc", "")
            if desc:
                parts.append(f"\n\nYour answer must be {desc}.")

        return "".join(parts)

    @staticmethod
    def _parse_answer(raw: str, choices_str: str) -> str:
        """Extract the chosen answer from the model's raw output."""
        # The model sometimes wraps the answer in thinking tags or extra text
        # Try to find an exact match with one of the choices
        choices = [c.strip() for c in choices_str.strip().split("\n") if c.strip()]

        # First: check if the raw answer IS one of the choices
        raw_clean = raw.strip()
        for choice in choices:
            if raw_clean == choice:
                return choice

        # Second: check if any choice appears in the answer (last occurrence wins)
        found = None
        for choice in choices:
            if choice in raw_clean:
                found = choice

        if found:
            return found

        # Third: look for the last line or sentence that matches
        lines = raw_clean.split("\n")
        for line in reversed(lines):
            line = line.strip()
            for choice in choices:
                if choice in line:
                    return choice

        # Fallback: return raw answer stripped
        return raw_clean


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------
def mcq_accuracy(example, prediction, trace=None) -> bool:
    """Exact match metric for multiple-choice questions."""
    gold = example.answer.strip()
    pred = prediction.answer.strip()

    # Exact match
    if pred == gold:
        return True

    # Gold contained in prediction (e.g., pred="The answer is Parrot", gold="Parrot")
    if gold in pred:
        return True

    return False


# ---------------------------------------------------------------------------
# Main optimization loop
# ---------------------------------------------------------------------------
def create_dspy_examples(samples: list[dict]) -> list[dspy.Example]:
    """Convert MMAR samples to DSPy Examples."""
    examples = []
    for s in samples:
        ex = dspy.Example(
            question=s["question"],
            choices="\n".join(s["choices"]),
            category=s["category"],
            sub_category=s.get("sub-category", ""),
            modality=s.get("modality", ""),
            language=s.get("language", "en"),
            audio_path=s["audio_path"],
            answer=s["answer"],
        ).with_inputs("question", "choices", "category", "sub_category", "modality", "language", "audio_path")
        examples.append(ex)
    return examples


def run_optimization(args):
    """Main optimization entry point."""

    print(f"Loading MMAR dataset from {args.dataset_meta_path}")
    samples = load_mmar(args.dataset_meta_path, args.dataset_audio_prefix)
    print(f"  Total samples: {len(samples)}")

    # Split into train/eval
    train_samples, eval_samples = stratified_split(
        samples, args.num_train, args.num_eval, seed=args.seed
    )
    print(f"  Train: {len(train_samples)}, Eval: {len(eval_samples)}")

    # Print category distribution
    for split_name, split in [("Train", train_samples), ("Eval", eval_samples)]:
        cats = defaultdict(int)
        for s in split:
            cats[s["category"]] += 1
        print(f"  {split_name} distribution: {dict(cats)}")

    # Create DSPy examples
    trainset = create_dspy_examples(train_samples)
    evalset = create_dspy_examples(eval_samples)

    # Configure DSPy LM (used by MIPRO for instruction proposal)
    # This points to the same vLLM server — MIPRO uses it to generate
    # candidate instructions (text-only, no audio needed for that).
    lm = dspy.LM(
        model=f"openai/{args.model_name}",
        api_base=args.vllm_base_url,
        api_key="EMPTY",
        max_tokens=4096,
        temperature=0.7,
    )
    dspy.configure(lm=lm)

    # Create the vLLM audio caller
    vllm_caller = VLLMAudioCaller(
        base_url=args.vllm_base_url,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        temperature=0.0,  # Greedy for evaluation
    )

    # Create the module
    module = AudioMCQModule(vllm_caller=vllm_caller)

    # Custom evaluate function that passes audio_path through
    def evaluate_with_audio(module, examples, metric, debug=False):
        correct = 0
        total = len(examples)
        mismatches = []
        for i, ex in enumerate(examples):
            try:
                pred = module(
                    question=ex.question,
                    choices=ex.choices,
                    category=ex.category,
                    sub_category=ex.sub_category,
                    modality=ex.modality,
                    language=ex.language,
                    audio_path=ex.audio_path,
                )
                is_correct = metric(ex, pred)
                if is_correct:
                    correct += 1
                elif debug and len(mismatches) < 10:
                    mismatches.append({
                        "question": ex.question[:80],
                        "gold": ex.answer,
                        "pred": pred.answer[:120],
                        "raw_answer": getattr(pred, 'raw_answer', '')[:200],
                        "has_thinking": bool(getattr(pred, 'thinking', '')),
                    })
                if (i + 1) % 20 == 0:
                    print(f"  [{i+1}/{total}] running acc: {correct}/{i+1} = {correct/(i+1):.3f}")
            except Exception as e:
                print(f"  Error on {ex.question[:50]}...: {e}")
        acc = correct / total if total > 0 else 0
        if debug and mismatches:
            print("\n  Sample mismatches:")
            for m in mismatches:
                print(f"    Q: {m['question']}")
                print(f"    Gold: {m['gold']}")
                print(f"    Pred: {m['pred']}")
                print(f"    Raw:  {m['raw_answer']}")
                print(f"    Had thinking: {m['has_thinking']}")
                print()
        return acc

    # Run MIPRO optimization (baseline eval runs after)
    print("\n" + "=" * 60)
    print("STARTING MIPROv2 OPTIMIZATION")
    print(f"  Candidates: {args.num_candidates}")
    print(f"  Train: {len(trainset)}, Eval: {len(evalset)}")
    print("=" * 60)

    optimizer = dspy.MIPROv2(
        metric=mcq_accuracy,
        auto=None,
        num_candidates=args.num_candidates,
        init_temperature=1.0,
        verbose=True,
        num_threads=args.num_threads,
    )

    optimized_module = optimizer.compile(
        module,
        trainset=trainset,
        valset=evalset,
        num_trials=args.num_candidates,
        minibatch_size=min(25, len(evalset)),
    )

    # Evaluate optimized module
    print("\n" + "=" * 60)
    print("OPTIMIZED EVALUATION")
    print("=" * 60)
    optimized_acc = evaluate_with_audio(optimized_module, evalset, mcq_accuracy, debug=True)
    print(f"Optimized accuracy: {optimized_acc:.4f} ({int(optimized_acc * len(evalset))}/{len(evalset)})")

    # Now run baseline eval for comparison
    print("\n" + "=" * 60)
    print("BASELINE EVALUATION (for comparison)")
    print("=" * 60)
    baseline_acc = evaluate_with_audio(module, evalset, mcq_accuracy, debug=True)
    print(f"Baseline accuracy: {baseline_acc:.4f} ({int(baseline_acc * len(evalset))}/{len(evalset)})")
    print(f"Improvement: {optimized_acc - baseline_acc:+.4f}")

    # Extract optimized prompts
    optimized_sig = optimized_module.predict.signature
    optimized_instructions = optimized_sig.instructions

    result = {
        "baseline_accuracy": baseline_acc,
        "optimized_accuracy": optimized_acc,
        "improvement": optimized_acc - baseline_acc,
        "optimized_system_prompt": optimized_instructions,
        "baseline_user_suffix": BASELINE_USER_SUFFIX,
        "num_candidates": args.num_candidates,
        "num_train": len(trainset),
        "num_eval": len(evalset),
        "seed": args.seed,
    }

    # Save results
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output_path}")

    # Also save the optimized module state
    module_path = args.output_path.replace(".json", "_module.json")
    optimized_module.save(module_path)
    print(f"Optimized module saved to {module_path}")

    # Print the optimized prompt for quick reference
    print("\n" + "=" * 60)
    print("OPTIMIZED SYSTEM PROMPT:")
    print("=" * 60)
    print(optimized_instructions)
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="DSPy prompt optimization for Qwen3-Omni on MMAR"
    )
    parser.add_argument(
        "--vllm_base_url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL for the vLLM OpenAI-compatible API",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-Omni-30B-A3B-Thinking",
        help="Model name as served by vLLM",
    )
    parser.add_argument(
        "--dataset_meta_path",
        type=str,
        required=True,
        help="Path to MMAR-meta.json",
    )
    parser.add_argument(
        "--dataset_audio_prefix",
        type=str,
        default="",
        help="Prefix prepended to audio_path in the dataset",
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=100,
        help="Number of training examples for optimization",
    )
    parser.add_argument(
        "--num_eval",
        type=int,
        default=100,
        help="Number of evaluation examples",
    )
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=50,
        help="Number of prompt candidates for MIPROv2",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=4,
        help="Number of threads for parallel evaluation",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Max new tokens for generation (needs to be large for thinking models)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/eval split",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/dspy_optimized/optimized_prompts.json",
        help="Path to save the optimized prompts",
    )

    args = parser.parse_args()
    run_optimization(args)


if __name__ == "__main__":
    main()
