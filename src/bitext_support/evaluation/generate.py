"""Shared JSONL utilities and deterministic evaluation generation."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from bitext_support.data.formatting import generation_ids


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()






def generate_base(loaded, selected: list[dict], examples: list[dict], generation: dict):
    import torch
    tokenizer, model = loaded.tokenizer, loaded.model
    for row in selected:
        ids = generation_ids(tokenizer, row, examples)
        if len(ids) > generation["max_input_tokens"]:
            raise ValueError(f"Input too long for {row['sample_id']}: {len(ids)} tokens")
        context = getattr(model.config, "max_position_embeddings", None)
        if context and len(ids) + generation["max_new_tokens"] > context:
            raise ValueError(f"Context limit exceeded for {row['sample_id']}")
        inputs = torch.tensor([ids], dtype=torch.long, device=loaded.device)
        options = {"max_new_tokens": generation["max_new_tokens"],
                   "do_sample": generation.get("do_sample", False),
                   "repetition_penalty": generation.get("repetition_penalty", 1.0),
                   "pad_token_id": tokenizer.eos_token_id}
        if options["do_sample"]:
            options.update(temperature=generation.get("temperature", 0.7),
                           top_p=generation.get("top_p", 0.9))
        torch.cuda.synchronize(loaded.device)
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(input_ids=inputs, **options)
        torch.cuda.synchronize(loaded.device)
        new_ids = output[0, len(ids):]
        new_token_ids = new_ids.tolist()
        eos = model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        ended_with_eos = bool(new_token_ids and new_token_ids[-1] in eos_ids)
        yield {"sample_id": row["sample_id"], "group_id": row["group_id"],
               "intent": row["intent"], "question": row["messages"][-2]["content"],
               "reference": row["messages"][-1]["content"],
               "answer": tokenizer.decode(new_ids, skip_special_tokens=True).strip(),
               "input_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
               "hit_token_limit": len(new_token_ids) >= generation["max_new_tokens"],
               "ended_with_eos": ended_with_eos,
               "input_tokens": len(ids), "output_tokens": int(new_ids.numel()),
               "generation_seconds": round(time.perf_counter() - start, 3)}


