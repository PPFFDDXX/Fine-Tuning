"""One Qwen chat-template path for training and evaluation."""

from __future__ import annotations

from collections.abc import Mapping


def token_ids(encoded) -> list[int]:
    """Normalize tokenizer outputs to one plain, unbatched ID list."""
    if isinstance(encoded, Mapping):
        return token_ids(encoded["input_ids"])
    if hasattr(encoded, "ids"):
        return token_ids(encoded.ids)
    if hasattr(encoded, "tolist"):
        return token_ids(encoded.tolist())
    if isinstance(encoded, (list, tuple)) and len(encoded) == 1 and not isinstance(encoded[0], int):
        return token_ids(encoded[0])
    if isinstance(encoded, tuple):
        encoded = list(encoded)
    if not isinstance(encoded, list) or not all(isinstance(value, int) for value in encoded):
        raise TypeError(f"Expected one token ID list, got {type(encoded).__name__}")
    return encoded


def prompt_messages(record: dict, examples: list[dict] | None = None) -> list[dict]:
    messages = record["messages"]
    roles = [item["role"] for item in messages]
    if (len(roles) < 3 or roles[0] != "system" or roles[-1] != "assistant"
            or any(role != ("user" if i % 2 else "assistant")
                   for i, role in enumerate(roles[1:], 1))):
        raise ValueError("Expected system followed by alternating user/assistant messages")
    result = [messages[0]]
    for example in examples or []:
        turns = example["messages"]
        prompt_messages(example)
        if turns[0]["content"] != messages[0]["content"]:
            raise ValueError("Few-shot system prompt differs from evaluation prompt")
        result.extend(turns[1:])
    result.extend(messages[1:-1])
    return result


def generation_ids(tokenizer, record: dict, examples: list[dict] | None = None) -> list[int]:
    return token_ids(tokenizer.apply_chat_template(
        prompt_messages(record, examples), tokenize=True, add_generation_prompt=True
    ))


def sft_features(tokenizer, record: dict) -> dict[str, list[int]]:
    """Mask system, user and assistant header; train on answer and turn terminator."""
    prefix = generation_ids(tokenizer, record)
    full = token_ids(tokenizer.apply_chat_template(record["messages"], tokenize=True))
    if full[:len(prefix)] != prefix:
        raise ValueError("Chat template prefix mismatch; cannot safely mask loss")
    if len(full) == len(prefix):
        raise ValueError("Assistant answer has no trainable tokens")
    return {"input_ids": full, "attention_mask": [1] * len(full),
            "labels": [-100] * len(prefix) + full[len(prefix):]}
