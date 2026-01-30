# Qwen3-Omni Chat Template Bug: Thinking Tags Stripped for Multimodal Content

## Summary

When finetuning Qwen3-Omni with thinking/reasoning content (`<think>...</think>` tags), the chat template silently strips these tags when using array-format content (required for multimodal inputs like audio). This causes the finetuned model to lose its thinking capability.

## The Problem

### Symptoms
- Base model generates thinking content: `<think>I hear a bird...</think>\n\nANSWER: Bird`
- Finetuned model only outputs: `ANSWER: Bird` (no thinking)
- Training data contained `<think>` tags, but model didn't learn to generate them

### Root Cause

The Qwen3-Omni `chat_template.json` has logic that determines whether to include thinking content based on the `last_query_index`:

```jinja
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and ... %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
```

**The key condition is `message.content is string`.**

For multimodal content, you must use array format:
```python
{
    "role": "user",
    "content": [
        {"type": "audio", "audio": "/path/to/audio.wav"},
        {"type": "text", "text": "What sound is this?"},
    ],
}
```

When content is an array (not a string), the condition `message.content is string` is FALSE. This means:
- `ns.last_query_index` stays at its initial value: `messages|length - 1` (e.g., 2 for 3 messages)
- For the assistant message (also at index 2), the condition `loop.index0 > ns.last_query_index` becomes `2 > 2` = FALSE
- Therefore, thinking tags are NOT included in the output

## Demonstration

```python
from transformers import Qwen3OmniMoeProcessor

processor = Qwen3OmniMoeProcessor.from_pretrained('Qwen/Qwen3-Omni-30B-A3B-Thinking')

# Using ARRAY content (multimodal) - THINKING GETS STRIPPED
conversation_array = [
    {"role": "system", "content": [{"type": "text", "text": "You are helpful."}]},
    {"role": "user", "content": [
        {"type": "audio", "audio": "/path/to/audio.wav"},
        {"type": "text", "text": "What sound?"},
    ]},
    {"role": "assistant", "content": [
        {"type": "text", "text": "<think>\nI hear a bird.\n</think>\n\nANSWER: Bird"}
    ]},
]

text = processor.apply_chat_template(conversation_array, tokenize=False, add_generation_prompt=False)
print(text)
# Output: ...assistant\nANSWER: Bird<|im_end|>
# NOTE: <think> tags are GONE!

# Using STRING content (text-only) - THINKING IS PRESERVED
conversation_string = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What sound?"},
    {"role": "assistant", "reasoning_content": "I hear a bird.", "content": "ANSWER: Bird"},
]

text = processor.apply_chat_template(conversation_string, tokenize=False, add_generation_prompt=False)
print(text)
# Output: ...assistant\n<think>\nI hear a bird.\n</think>\n\nANSWER: Bird<|im_end|>
# NOTE: <think> tags are PRESERVED!
```

## Impact on Training

### What Happened

During training, we used `apply_chat_template()` on the full conversation:

```python
text = processor.apply_chat_template(
    conversation,  # includes system, user (with audio), and assistant
    tokenize=False,
    add_generation_prompt=False
)
```

The template stripped the `<think>` tags, so the model learned:

```
Input:  <|im_start|>assistant\n
Output: ANSWER: Bird<|im_end|>
```

Instead of:

```
Input:  <|im_start|>assistant\n
Output: <think>\nI hear a bird.\n</think>\n\nANSWER: Bird<|im_end|>
```

### Why Base Model Still Works

During inference, the chat template only processes the prompt (system + user messages), not the assistant response. The model generates freely from `<|im_start|>assistant\n`, including `<think>` tags because it was pretrained to do so.

The finetuning "overwrote" this pretrained behavior by teaching the model to skip thinking.

## The Fix

Bypass the chat template for the assistant response:

```python
# Build conversation WITHOUT assistant message
conversation_without_assistant = [
    {"role": "system", "content": [{"type": "text", "text": "..."}]},
    {"role": "user", "content": [
        {"type": "audio", "audio": audio_path},
        {"type": "text", "text": question},
    ]},
]

# Apply template with add_generation_prompt=True to get:
# <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
text = processor.apply_chat_template(
    conversation_without_assistant,
    tokenize=False,
    add_generation_prompt=True
)

# Manually append the assistant response WITH thinking tags preserved
text = text + response + "<|im_end|>\n"
```

## Lessons Learned

1. **Always verify tokenized output**: Print the actual text/tokens before training to ensure your data looks correct.

2. **Chat templates can have hidden behavior**: They're designed for inference, not training. When using them for training data, verify they preserve all content.

3. **Test with a single sample first**: Before full training, verify one sample's tokenized output matches your expectations.

4. **Multimodal models have edge cases**: The template logic was written primarily for text. Multimodal inputs (arrays) can trigger unexpected behavior.

5. **Debug by comparison**: When finetuning breaks expected behavior, compare:
   - What the base model sees during inference
   - What your training data actually contains after processing

## Files Changed

- `src/training/rest/train.py`: Fixed to bypass chat template for assistant response

## Verification

After fixing, verify the training text includes thinking:

```python
# After the fix, text should look like:
# <|im_start|>system
# You are a helpful assistant...
# <|im_end|>
# <|im_start|>user
# <|audio_start|><|audio_pad|><|audio_end|>What sound is this?
# <|im_end|>
# <|im_start|>assistant
# <think>
# I hear a bird chirping...
# </think>
#
# ANSWER: Bird<|im_end|>
```
