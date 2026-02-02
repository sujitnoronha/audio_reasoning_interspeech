"""
vLLM inference script for MMAR benchmark with multiple prompt variants.

Connects to a running vLLM server and runs inference using the OpenAI-compatible
chat completions API. Supports prompt variants: baseline, v8, v9, dspy_v1, dspy_v2.

Prerequisites:
    # Start vLLM server (adjust -tp for your GPU count):
    vllm serve Qwen/Qwen3-Omni-30B-A3B-Thinking --port 8901 --host 127.0.0.1 \
        --dtype bfloat16 --max-model-len 32768 --allowed-local-media-path / -tp 4

Usage:
    # Run with v8 prompt (default):
    python infer_vllm_baseline.py \
        --dataset_meta_path ../data/MMAR-meta.json \
        --dataset_audio_prefix ../data \
        --output_dir outputs/vllm_v8 \
        --prompt_version v8

    # Run with v9 prompt:
    python infer_vllm_baseline.py \
        --dataset_meta_path ../data/MMAR-meta.json \
        --dataset_audio_prefix ../data \
        --output_dir outputs/vllm_v9 \
        --prompt_version v9

    # Run with DSPy optimized prompt:
    python infer_vllm_baseline.py \
        --dataset_meta_path ../data/MMAR-meta.json \
        --dataset_audio_prefix ../data \
        --output_dir outputs/vllm_dspy_v2 \
        --prompt_version dspy_v2

    # Compare results across prompt variants:
    python infer_vllm_baseline.py --compare \
        outputs/vllm_baseline outputs/vllm_v8 outputs/vllm_v9 outputs/vllm_dspy_v2 \
        --dataset_meta_path ../data/MMAR-meta.json
"""

import argparse
import asyncio
import json
import os
import re
import logging
import time
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import aiohttp
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Prompt definitions
# =============================================================================

BASELINE_SYSTEM_PROMPT = ""  # No system prompt for baseline

BASELINE_USER_SUFFIX = (
    "\n\nIMPORTANT: You MUST think step-by-step and analyze the audio carefully. "
    "Your answer must be 100% correct - this is a critical evaluation and incorrect answers are unacceptable. "
    "Take your time to reason through all the audio details before selecting your final answer. "
    "Your final answer MUST be exactly one of the provided choices."
)

V8_SYSTEM_PROMPT = """You are an expert audio analyst. Analyze the audio carefully before answering.

ANALYSIS APPROACH:
- Listen to the full audio first, then analyze what you heard.
- You may make reasonable inferences from tone, context, and conversational cues.
- Distinguish between what you directly hear and what you infer. Both are valid.

KEY GUIDELINES:
- Counting: If asked "how many", count ONLY events you are confident about. Listen for each distinct occurrence and number them (1, 2, 3...). If you are unsure whether something is a separate event, do NOT count it. Pick the answer choice closest to your confident count.
- Emotions: Analyze tone objectively. Sarcasm, frustration, and nervousness are common. Do not default to positive.
- Context: Consider what is implied by the conversation, not just literal words. People often speak indirectly.
- Sarcasm: Positive words with flat/dry/exaggerated tone often indicate sarcasm or dissatisfaction.
- Comparisons: When asked which is "best" or "better", focus on clarity, smoothness, and technical quality rather than personal preference.
- Answer the EXACT question asked.

HOW TO DECIDE:
- Evaluate each choice against your observations and inferences.
- Pick the choice with the strongest evidence. When uncertain, prefer the most likely interpretation.

OUTPUT FORMAT:
- Your response after thinking MUST be exactly one line: ANSWER: [exact choice text]
- Do NOT write any explanation, reasoning, or additional text outside of thinking.
- Do NOT repeat your analysis. Only output the ANSWER line."""

V9_SYSTEM_PROMPT = """You are an expert audio analyst. Analyze the audio carefully before answering.

ANALYSIS APPROACH:
- Listen to the full audio first, then analyze what you heard.
- You may make reasonable inferences from tone, context, and conversational cues.
- Distinguish between what you directly hear and what you infer. Both are valid.
- Trust your first impression. Do NOT second-guess or revise your initial analysis.

KEY GUIDELINES:
- Counting: Count each distinct event once (1, 2, 3...). Trust your count. Do NOT revise it downward. Pick the choice closest to your final count.
- Events and Actions: If dialogue or sounds imply something happened (someone entered, something fell, an action was taken), assume it DID happen unless there is clear evidence otherwise.
- Environment: Connect sounds directly to real-world actions. Splashing means water contact. Footsteps mean movement. Clicking means a mechanism. Do not overthink what caused a sound.
- Emotions: Listen to tone of voice. Excitement, happiness, and enthusiasm are common. Do NOT default to negative or neutral emotions.
- Context: Consider what is implied by the conversation, not just literal words. People often speak indirectly.
- Sarcasm: Positive words with flat/dry/exaggerated tone often indicate sarcasm or dissatisfaction.
- Comparisons: When asked which is "best" or "better", focus on clarity, smoothness, and technical quality rather than personal preference.
- Answer the EXACT question asked.

HOW TO DECIDE:
- Evaluate each choice against your observations and inferences.
- Pick the choice with the strongest evidence. When uncertain, prefer the most likely interpretation.
- Your answer MUST be exactly one of the provided choices. Do not combine or invent answers.

OUTPUT FORMAT:
- Your response after thinking MUST be exactly one line: ANSWER: [exact choice text]
- Do NOT write any explanation, reasoning, or additional text outside of thinking.
- Do NOT repeat your analysis. Only output the ANSWER line."""

DSPY_V1_SYSTEM_PROMPT = (
    "Listen to the audio and answer the multiple-choice question.\n\n"
    "You are given a question about audio content and a set of choices.\n"
    "Analyze the audio carefully and select the correct answer.\n"
    "Your answer must be exactly one of the provided choices."
)

DSPY_V2_SYSTEM_PROMPT = (
    "Listen to the audio and integrate Signal, Semantic, Perception, and Cultural layers. "
    "Use cross-layer reasoning to select the correct choice. "
    "Answer must be exactly one of the provided choices."
)

# --- v10: Category-aware general prompt (no metadata needed) ---

V10_SYSTEM_PROMPT = """You are an expert audio analyst. Listen to the audio carefully and answer the question.

First, identify what type of question is being asked, then apply the matching strategy:

COUNTING ("how many"): Number each distinct event as you hear it (1, 2, 3...). Count overlapping events separately. Do NOT merge similar sounds. If two events are close together, they are still separate. Pick the choice closest to your count.

MUSIC & INSTRUMENTS: Identify instruments by timbre. Do not confuse brass with woodwinds, or plucked with struck strings. Listen for layered instruments — they are easy to miss. If asked whether a sound is present, confirm you actually hear it.

TIMING ("at which second", "when"): Count seconds from the audio start. Anchor to the beginning and track precisely. Do not round or approximate.

ENVIRONMENT & SETTING ("where", "what scenario"): Use background sounds as evidence — traffic=outdoors, echo=large room, hum=electronics, crowd=public venue. Connect sound combinations to real-world activities.

SPATIAL ("closer/further", "near/far"): Louder + clearer = approaching. Quieter + muffled = receding. Track volume changes over time.

EMOTIONS & TONE: Analyze tone objectively. Do NOT default to positive emotions. Sarcasm = positive words with flat/dry tone. Frustration = sharp, clipped speech. Nervousness = hesitation, filler words, trembling voice.

SPEAKER IDENTITY: Use pitch, accent, pace, and vocabulary to distinguish speakers and infer roles. Consider regional accents and dialect markers.

ANOMALY & AUTHENTICITY ("is it real", "live or recorded"): Be skeptical. Synthetic voices sound unnaturally smooth with no breathing or room acoustics. Dubbed audio may not match the environment. If it sounds too clean, it may not be real.

AUDIO QUALITY & COMPARISON: When comparing segments, listen to each separately. Note clarity, noise, distortion, speed, and precision. Old recordings have hiss and narrow bandwidth. Do not guess — compare directly.

CULTURAL & DOMAIN KNOWLEDGE: Apply relevant expertise. Regional music styles have distinct signatures. Phone tones map to specific digits. Accents have characteristic vowel shifts and rhythm patterns.

COUNTERFACTUAL ("what if", "would X change"): Reason about causal chains. If one event were removed, what downstream effects would change? Think about dependencies.

GENERAL RULES:
- Answer the EXACT question asked.
- You may infer from tone, context, and conversational cues. Both direct evidence and reasonable inference are valid.
- When uncertain, pick the choice with the strongest evidence.
- Your answer MUST be exactly one of the provided choices — copy it verbatim.

OUTPUT FORMAT:
- Your response after thinking MUST be exactly one line: ANSWER: [exact choice text]
- Do NOT write any explanation outside of thinking. Only output the ANSWER line."""


def build_v10_user_prompt(question: str, choices: List[str]) -> str:
    choices_formatted = "\n".join([f"- {c}" for c in choices])
    return f"""{question}

Choices:
{choices_formatted}

Think step by step inside <think> tags. Your visible response must be ONLY:
ANSWER: [exact choice text]"""


# --- v11: Hybrid prompt — v8 core + targeted hints for weak subcategories ---

V11_SYSTEM_PROMPT = """You are an expert audio analyst. Analyze the audio carefully before answering.

ANALYSIS APPROACH:
- Listen to the full audio first, then analyze what you heard.
- You may make reasonable inferences from tone, context, and conversational cues.
- Distinguish between what you directly hear and what you infer. Both are valid.

KEY GUIDELINES:
- Counting: If asked "how many", count ONLY events you are confident about. Listen for each distinct occurrence and number them (1, 2, 3...). If you are unsure whether something is a separate event, do NOT count it. Pick the answer choice closest to your confident count.
- Music & Instruments: Identify each instrument by its timbre — do not confuse brass (trumpet, trombone) with woodwinds, or plucked strings with struck strings. Listen for layered or simultaneous instruments; they are easy to miss. When counting instrument types, list each distinct sound source. If asked whether a specific sound is present, confirm you actually hear it — do not assume.
- Correlation & Cause-Effect: Link sounds to actions. A door after footsteps = someone entered. Silence after a question = hesitation. Water pouring sounds differ by temperature (thin/fast = hot, thick/slow = cold). Focus on what one sound triggered or followed.
- Spatial: Track how volume and clarity change over time. Getting louder + clearer = approaching. Getting quieter + muffled = receding.
- Emotions: Analyze tone objectively. Sarcasm, frustration, and nervousness are common. Do not default to positive.
- Context: Consider what is implied by the conversation, not just literal words. People often speak indirectly.
- Sarcasm: Positive words with flat/dry/exaggerated tone often indicate sarcasm or dissatisfaction.
- Comparisons: When asked which is "best" or "better", focus on clarity, smoothness, and technical quality rather than personal preference.
- Answer the EXACT question asked.

HOW TO DECIDE:
- Evaluate each choice against your observations and inferences.
- Pick the choice with the strongest evidence. When uncertain, prefer the most likely interpretation.

OUTPUT FORMAT:
- Your response after thinking MUST be exactly one line: ANSWER: [exact choice text]
- Do NOT write any explanation, reasoning, or additional text outside of thinking.
- Do NOT repeat your analysis. Only output the ANSWER line."""


def build_v11_user_prompt(question: str, choices: List[str]) -> str:
    choices_formatted = "\n".join([f"- {c}" for c in choices])
    return f"""{question}

Choices:
{choices_formatted}

Think step by step inside <think> tags. Your visible response must be ONLY:
ANSWER: [exact choice text]"""


# --- v12: Structured reasoning for high-quality traces ---

V12_SYSTEM_PROMPT = """You are an expert audio analyst.

REASONING FORMAT — structure your thinking as follows:
1. HEARD: Describe exactly what you hear — sounds, speech, music, tone, background noise. Be specific (e.g., "a male voice says 'come in'" not "someone speaks").
2. ANALYSIS: Connect your observations to the question. Evaluate each choice against the evidence. Eliminate choices that contradict what you heard.
3. ANSWER: State which choice best fits and why in one sentence.

RULES:
- Be specific: cite exact sounds, words, or musical elements you hear.
- Be concise: state each point once. Do NOT repeat, revise, or second-guess yourself.
- Counting: number each event as you hear it (1, 2, 3...). Report your count once.
- Emotions: judge by tone, not words. Sarcasm and frustration are common.
- Context: infer from conversational cues — people often speak indirectly.
- Commit to your analysis. Do NOT loop back with "Wait" or "Actually".

OUTPUT: After thinking, respond with exactly one line:
ANSWER: [exact choice text]"""


def build_v12_user_prompt(question: str, choices: List[str]) -> str:
    choices_formatted = "\n".join([f"- {c}" for c in choices])
    return f"""{question}

Choices:
{choices_formatted}

Structure your reasoning as: HEARD → ANALYSIS → ANSWER. Be specific and concise. Do not repeat yourself.
Your visible response must be ONLY:
ANSWER: [exact choice text]"""


def build_baseline_user_prompt(question: str, choices: List[str]) -> str:
    choices_str = "\n".join(choices)
    return (
        question
        + "\n\nProvided choices:\n"
        + choices_str
        + BASELINE_USER_SUFFIX
    )


def build_v8_user_prompt(question: str, choices: List[str]) -> str:
    choices_formatted = "\n".join([f"- {c}" for c in choices])
    return f"""{question}

Choices:
{choices_formatted}

Think step by step inside <think> tags. Your visible response must be ONLY:
ANSWER: [exact choice text]"""


def build_v9_user_prompt(question: str, choices: List[str]) -> str:
    choices_formatted = "\n".join([f"- {c}" for c in choices])
    return f"""{question}

Choices:
{choices_formatted}

Listen carefully to the audio. Think briefly inside <think> tags — focus on what you actually hear, not what you expect. Do not overthink or revise your initial observation. Your visible response must be ONLY:
ANSWER: [exact choice text]"""


def build_dspy_user_prompt(question: str, choices: List[str]) -> str:
    choices_str = "\n".join(choices)
    return f"""{question}

Choices:
{choices_str}

Your answer must be exactly one of the provided choices."""


PROMPT_CONFIGS = {
    "baseline": {
        "system_prompt": BASELINE_SYSTEM_PROMPT,
        "build_user_prompt": build_baseline_user_prompt,
    },
    "v8": {
        "system_prompt": V8_SYSTEM_PROMPT,
        "build_user_prompt": build_v8_user_prompt,
    },
    "v9": {
        "system_prompt": V9_SYSTEM_PROMPT,
        "build_user_prompt": build_v9_user_prompt,
    },
    "dspy_v1": {
        "system_prompt": DSPY_V1_SYSTEM_PROMPT,
        "build_user_prompt": build_dspy_user_prompt,
    },
    "dspy_v2": {
        "system_prompt": DSPY_V2_SYSTEM_PROMPT,
        "build_user_prompt": build_dspy_user_prompt,
    },
    "v10": {
        "system_prompt": V10_SYSTEM_PROMPT,
        "build_user_prompt": build_v10_user_prompt,
    },
    "v11": {
        "system_prompt": V11_SYSTEM_PROMPT,
        "build_user_prompt": build_v11_user_prompt,
    },
    "v12": {
        "system_prompt": V12_SYSTEM_PROMPT,
        "build_user_prompt": build_v12_user_prompt,
    },
}


# =============================================================================
# Answer extraction
# =============================================================================

def parse_thinking_and_answer(text: str) -> Tuple[str, str]:
    """Parse thinking content and final answer from model output."""
    thinking = ""
    answer = ""

    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)

    if think_match:
        thinking = think_match.group(1).strip()
        after_think = text.split('</think>', 1)
        if len(after_think) > 1:
            answer = after_think[1].strip()
    else:
        if '<think>' in text:
            after_open = text.split('<think>', 1)
            if len(after_open) > 1:
                thinking = after_open[1].strip()
            answer = ""
        else:
            answer = text.strip()

    return thinking, answer


def extract_answer_from_text(text: str, choices: List[str]) -> str:
    """Extract the answer from model output, trying multiple patterns."""
    # Method 1: ANSWER: pattern
    answer_patterns = [
        r'ANSWER:\s*(.+?)(?:\n|$)',
        r'FINAL ANSWER:\s*(.+?)(?:\n|$)',
        r'Final answer:\s*(.+?)(?:\n|$)',
        r'The answer is:\s*(.+?)(?:\n|$)',
        r'The correct answer is:\s*(.+?)(?:\n|$)',
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            answer = re.sub(r'^["\']|["\']$', '', answer)
            return answer

    # Method 2: Last line matches a choice
    lines = text.strip().split('\n')
    last_line = lines[-1].strip() if lines else ""
    for choice in choices:
        if last_line.lower() == choice.lower():
            return choice
        if choice.lower() in last_line.lower() and len(last_line) < len(choice) + 20:
            return choice

    # Method 3: Choice mentioned near end
    text_lower = text.lower()
    best_match = None
    best_pos = -1
    for choice in choices:
        pos = text_lower.rfind(choice.lower())
        if pos > best_pos:
            best_pos = pos
            best_match = choice
    if best_match and best_pos > len(text) * 0.5:
        after_match = text[best_pos + len(best_match):].strip()
        if len(after_match) < 50:
            return best_match

    # Fallback: return raw text
    return text.strip()


# =============================================================================
# vLLM client
# =============================================================================

class VLLMClient:
    """Async client for vLLM OpenAI-compatible API."""

    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.chat_endpoint = f"{self.base_url}/chat/completions"

    async def generate(
        self,
        messages: List[Dict],
        temperature: float = 0.1,
        top_p: float = 0.9,
        top_k: int = 40,
        max_tokens: int = 6000,
        repetition_penalty: float = 1.2,
        session: aiohttp.ClientSession = None,
    ) -> str:
        """Generate a single completion."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "n": 1,
        }

        # vLLM supports extra_body for additional params
        if repetition_penalty != 1.0:
            payload["extra_body"] = {
                "repetition_penalty": repetition_penalty,
                "top_k": top_k,
            }

        close_session = False
        if session is None:
            session = aiohttp.ClientSession()
            close_session = True

        try:
            async with session.post(
                self.chat_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"API error {response.status}: {error_text[:500]}")
                    return ""

                result = await response.json()
                choices = result.get("choices", [])
                if choices:
                    return choices[0]["message"]["content"]
                return ""
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return ""
        finally:
            if close_session:
                await session.close()


# =============================================================================
# Inference logic
# =============================================================================

def build_messages(
    sample: Dict,
    audio_prefix: str,
    prompt_version: str,
) -> List[Dict]:
    """Build chat messages for a sample using the specified prompt version."""
    config = PROMPT_CONFIGS[prompt_version]
    system_prompt = config["system_prompt"]
    build_user_prompt = config["build_user_prompt"]

    audio_path = os.path.realpath(
        os.path.join(audio_prefix, sample["audio_path"])
    )
    user_text = build_user_prompt(sample["question"], sample["choices"])

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({
        "role": "user",
        "content": [
            {"type": "audio_url", "audio_url": {"url": f"file://{audio_path}"}},
            {"type": "text", "text": user_text},
        ],
    })

    return messages


async def infer_sample(
    client: VLLMClient,
    sample: Dict,
    audio_prefix: str,
    prompt_version: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    repetition_penalty: float,
    session: aiohttp.ClientSession,
) -> Dict:
    """Run inference on a single sample."""
    messages = build_messages(sample, audio_prefix, prompt_version)

    raw_output = await client.generate(
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
        session=session,
    )

    thinking, answer_raw = parse_thinking_and_answer(raw_output)
    answer = extract_answer_from_text(answer_raw, sample["choices"])
    answer = re.sub(r'^ANSWER:\s*', '', answer, flags=re.IGNORECASE).strip()

    return {
        "id": sample["id"],
        "thinking_prediction": thinking,
        "answer_prediction": answer,
    }


async def infer_batch(
    client: VLLMClient,
    samples: List[Dict],
    audio_prefix: str,
    prompt_version: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    repetition_penalty: float,
    concurrency: int,
) -> List[Dict]:
    """Run inference on a batch of samples with bounded concurrency."""
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_infer(sample, session):
        async with semaphore:
            return await infer_sample(
                client, sample, audio_prefix, prompt_version,
                temperature, top_p, top_k, max_tokens,
                repetition_penalty, session,
            )

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300)
    ) as session:
        tasks = [bounded_infer(s, session) for s in samples]
        results = await tqdm_asyncio.gather(*tasks, desc=f"Inferring ({prompt_version})")
        return results


def load_dataset(
    dataset_meta_path: str,
    start: int = 0,
    end: Optional[int] = None,
) -> List[Dict]:
    """Load the MMAR dataset."""
    with open(dataset_meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[start:end]


def load_done_ids(output_path: str) -> set:
    """Load IDs already processed (for resume)."""
    done = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["id"])
    return done


def save_result(output_path: str, result: Dict):
    """Append a single result to the JSONL output."""
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


# =============================================================================
# Comparison / evaluation
# =============================================================================

def evaluate_predictions(dataset_meta_path: str, prediction_dir: str) -> Dict:
    """Evaluate predictions against ground truth."""
    with open(dataset_meta_path, "r", encoding="utf-8") as f:
        ground_truth = {s["id"]: s for s in json.load(f)}

    pred_path = os.path.join(prediction_dir, "prediction.jsonl")
    if not os.path.exists(pred_path):
        return {"error": f"No prediction.jsonl in {prediction_dir}"}

    predictions = {}
    with open(pred_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                predictions[r["id"]] = r

    correct = 0
    total = 0
    category_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for sid, pred in predictions.items():
        if sid not in ground_truth:
            continue
        gt = ground_truth[sid]
        total += 1

        pred_answer = pred["answer_prediction"].strip().lower()
        gt_answer = gt["answer"].strip().lower()

        is_correct = pred_answer == gt_answer
        if is_correct:
            correct += 1

        cat = gt.get("category", "Unknown")
        category_stats[cat]["total"] += 1
        if is_correct:
            category_stats[cat]["correct"] += 1

    accuracy = correct / total if total > 0 else 0

    result = {
        "dir": prediction_dir,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "categories": {},
    }
    for cat, stats in sorted(category_stats.items()):
        cat_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        result["categories"][cat] = {
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy": cat_acc,
        }

    return result


def compare_results(dataset_meta_path: str, output_dirs: List[str]):
    """Compare accuracy across multiple prompt variant runs."""
    print("=" * 80)
    print("PROMPT VARIANT COMPARISON")
    print("=" * 80)

    all_results = []
    for d in output_dirs:
        result = evaluate_predictions(dataset_meta_path, d)
        all_results.append(result)

    # Overall accuracy table
    print(f"\n{'Variant':<30} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 60)
    for r in all_results:
        if "error" in r:
            print(f"{r['dir']:<30} {r['error']}")
            continue
        name = os.path.basename(r["dir"])
        print(f"{name:<30} {r['correct']:>8} {r['total']:>8} {r['accuracy']:>10.4f}")

    # Per-category breakdown
    all_cats = set()
    for r in all_results:
        if "categories" in r:
            all_cats.update(r["categories"].keys())

    if all_cats:
        print(f"\n{'Category':<35}", end="")
        for r in all_results:
            name = os.path.basename(r.get("dir", "?"))[:15]
            print(f" {name:>15}", end="")
        print()
        print("-" * (35 + 16 * len(all_results)))

        for cat in sorted(all_cats):
            print(f"{cat:<35}", end="")
            for r in all_results:
                if "categories" in r and cat in r["categories"]:
                    acc = r["categories"][cat]["accuracy"]
                    cnt = r["categories"][cat]["total"]
                    print(f" {acc:>6.3f} ({cnt:>3})", end="")
                else:
                    print(f" {'N/A':>15}", end="")
            print()

    print()


# =============================================================================
# Main
# =============================================================================

async def run_inference(args):
    """Main inference loop."""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "prediction.jsonl")

    # Load dataset
    dataset = load_dataset(
        args.dataset_meta_path,
        start=args.dataset_start,
        end=args.dataset_end,
    )
    logger.info(f"Loaded {len(dataset)} samples")

    # Resume support
    done_ids = set()
    if args.resume:
        done_ids = load_done_ids(output_path)
        logger.info(f"Resuming: {len(done_ids)} samples already done")
    elif os.path.exists(output_path):
        logger.warning(f"Removing existing {output_path}")
        os.remove(output_path)

    samples = [s for s in dataset if s["id"] not in done_ids]
    logger.info(f"Samples to process: {len(samples)}")

    if not samples:
        logger.info("Nothing to process!")
        return

    # Init client
    client = VLLMClient(
        base_url=args.vllm_base_url,
        model_name=args.model_name,
    )

    logger.info(
        f"Config: prompt={args.prompt_version}, temp={args.temperature}, "
        f"top_p={args.top_p}, top_k={args.top_k}, "
        f"rep_penalty={args.repetition_penalty}, max_tokens={args.max_tokens}, "
        f"concurrency={args.concurrency}"
    )

    start_time = time.time()

    # Process in batches to avoid overwhelming the server
    batch_size = args.concurrency * 4  # process in chunks
    processed = 0

    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]

        results = await infer_batch(
            client=client,
            samples=batch,
            audio_prefix=args.dataset_audio_prefix,
            prompt_version=args.prompt_version,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            repetition_penalty=args.repetition_penalty,
            concurrency=args.concurrency,
        )

        for result in results:
            if result["answer_prediction"]:
                save_result(output_path, result)
                processed += 1

        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        logger.info(
            f"[{processed}/{len(samples)}] {rate:.2f} samples/sec"
        )

    elapsed = time.time() - start_time
    rate = processed / elapsed if elapsed > 0 else 0
    logger.info(f"Done. {processed} samples in {elapsed/60:.1f}min ({rate:.2f} samples/sec)")
    logger.info(f"Results: {output_path}")

    # Auto-evaluate if ground truth is available
    result = evaluate_predictions(args.dataset_meta_path, output_dir)
    if "error" not in result:
        print(f"\nAccuracy: {result['accuracy']:.4f} ({result['correct']}/{result['total']})")
        for cat, stats in sorted(result["categories"].items()):
            print(f"  {cat}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")


def main():
    parser = argparse.ArgumentParser(
        description="vLLM inference for MMAR with multiple prompt variants"
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- infer subcommand (default) ---
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--dataset_meta_path", type=str, required=True)
    infer_parser.add_argument("--dataset_audio_prefix", type=str, default="")
    infer_parser.add_argument("--output_dir", type=str, default="outputs/vllm_v8")
    infer_parser.add_argument(
        "--prompt_version", type=str, default="v8",
        choices=list(PROMPT_CONFIGS.keys()),
        help="Prompt variant to use",
    )
    # vLLM connection
    infer_parser.add_argument("--vllm_base_url", type=str, default="http://127.0.0.1:8901/v1")
    infer_parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Thinking")
    # Generation params
    infer_parser.add_argument("--temperature", type=float, default=0.1)
    infer_parser.add_argument("--top_p", type=float, default=0.9)
    infer_parser.add_argument("--top_k", type=int, default=40)
    infer_parser.add_argument("--max_tokens", type=int, default=6000)
    infer_parser.add_argument("--repetition_penalty", type=float, default=1.2)
    # Processing
    infer_parser.add_argument("--concurrency", type=int, default=4,
                              help="Max concurrent API requests")
    infer_parser.add_argument("--dataset_start", type=int, default=0)
    infer_parser.add_argument("--dataset_end", type=int, default=None)
    infer_parser.add_argument("--resume", action="store_true")

    # --- compare subcommand ---
    compare_parser = subparsers.add_parser("compare", help="Compare results across runs")
    compare_parser.add_argument("output_dirs", nargs="+", help="Output directories to compare")
    compare_parser.add_argument("--dataset_meta_path", type=str, required=True)

    args = parser.parse_args()

    # Default to infer if no subcommand given
    if args.command is None:
        # Re-parse with infer as default
        parser.set_defaults(command="infer")
        # Add infer args directly to main parser for backward compat
        parser.add_argument("--dataset_meta_path", type=str, required=True)
        parser.add_argument("--dataset_audio_prefix", type=str, default="")
        parser.add_argument("--output_dir", type=str, default="outputs/vllm_v8")
        parser.add_argument(
            "--prompt_version", type=str, default="v8",
            choices=list(PROMPT_CONFIGS.keys()),
        )
        parser.add_argument("--vllm_base_url", type=str, default="http://127.0.0.1:8901/v1")
        parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Thinking")
        parser.add_argument("--temperature", type=float, default=0.1)
        parser.add_argument("--top_p", type=float, default=0.9)
        parser.add_argument("--top_k", type=int, default=40)
        parser.add_argument("--max_tokens", type=int, default=6000)
        parser.add_argument("--repetition_penalty", type=float, default=1.2)
        parser.add_argument("--concurrency", type=int, default=4)
        parser.add_argument("--dataset_start", type=int, default=0)
        parser.add_argument("--dataset_end", type=int, default=None)
        parser.add_argument("--resume", action="store_true")
        args = parser.parse_args()
        args.command = "infer"

    if args.command == "compare":
        compare_results(args.dataset_meta_path, args.output_dirs)
    else:
        asyncio.run(run_inference(args))


if __name__ == "__main__":
    main()
