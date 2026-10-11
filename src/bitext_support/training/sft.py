"""Step 4.1: bounded QLoRA training and independent Adapter reload verification."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import random
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from bitext_support.data.formatting import generation_ids, sft_features
from bitext_support.evaluation.generate import digest, read_jsonl
from bitext_support.training.model_loader import PROJECT_ROOT, load_base_model


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_config(config: dict) -> None:
    for key in ("sample_count", "max_seq_length", "max_steps", "batch_size",
                "gradient_accumulation_steps"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("learning_rate", "max_grad_norm"):
        if not math.isfinite(float(config[key])) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not config["probe"]["prompt"].strip() or config["probe"]["max_new_tokens"] < 1:
        raise ValueError("Probe needs a nonempty prompt and positive output limit")


def select_features(tokenizer, rows: list[dict], config: dict) -> tuple[list[dict], list[dict], int]:
    """Select whole answers from distinct train groups; never silently truncate."""
    ranked = sorted(rows, key=lambda row: hashlib.sha256(
        f"{config['seed']}:{row['sample_id']}".encode()).hexdigest())
    features, selected, groups = [], [], set()
    skipped = 0
    for row in ranked:
        if row["group_id"] in groups:
            continue
        if not row["messages"][-1]["content"].strip():
            raise ValueError(f"Empty answer: {row['sample_id']}")
        feature = sft_features(tokenizer, row)
        if len(feature["input_ids"]) > config["max_seq_length"]:
            skipped += 1
            continue
        features.append(feature)
        selected.append(row)
        groups.add(row["group_id"])
        if len(features) == config["sample_count"]:
            break
    if len(features) != config["sample_count"]:
        raise ValueError("Not enough distinct, eligible training examples")
    return features, selected, skipped


def padded_batch(features: list[dict], pad_id: int) -> dict[str, list[list[int]]]:
    """Right-pad inputs; exclude padding from attention and supervised labels."""
    width = max(len(row["input_ids"]) for row in features)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for row in features:
        ids, labels = row["input_ids"], row["labels"]
        if not (len(ids) == len(labels) == len(row["attention_mask"])):
            raise ValueError("Feature lengths differ")
        active = [i for i, label in enumerate(labels) if label != -100]
        if not active or active[0] == 0 or active != list(range(active[0], len(ids))):
            raise ValueError("Expected a masked prompt followed by a supervised answer")
        if any(labels[i] != ids[i] for i in active):
            raise ValueError("Supervised labels must match answer token IDs")
        padding = width - len(ids)
        batch["input_ids"].append(ids + [pad_id] * padding)
        batch["attention_mask"].append(row["attention_mask"] + [0] * padding)
        batch["labels"].append(labels + [-100] * padding)
    return batch


def adapter_digest(model) -> str:
    """Fingerprint actual Adapter tensor contents, independent of file metadata."""
    from peft import get_peft_model_state_dict
    import torch
    checksum = hashlib.sha256()
    for name, value in sorted(get_peft_model_state_dict(model).items()):
        value = value.detach().cpu().contiguous()
        checksum.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        checksum.update(value.view(torch.uint8).numpy().tobytes())
    return checksum.hexdigest()


def probe_answer(model, tokenizer, device, record: dict, max_new_tokens: int) -> dict:
    import torch
    ids = generation_ids(tokenizer, record)
    context = getattr(model.config, "max_position_embeddings", None)
    if context and len(ids) + max_new_tokens > context:
        raise ValueError("Probe exceeds model context")
    inputs = torch.tensor([ids], dtype=torch.long, device=device)
    model.eval()
    with torch.inference_mode():
        output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                max_new_tokens=max_new_tokens, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id, use_cache=True)
    tokens = output[0, len(ids):].tolist()
    answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()
    if not answer:
        raise RuntimeError("Adapter probe produced an empty answer")
    return {"token_ids": tokens, "answer": answer}


def train_smoke(config: dict) -> Path:
    """Run a fixed number of optimizer steps; save evidence before reload check."""
    import torch
    import torch.nn.functional as F
    import yaml
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    validate_config(config)
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    inference = yaml.safe_load(project_path(config["model_config"]).read_text(encoding="utf-8"))
    model_config = inference["model"]
    loaded = load_base_model(model_config)
    tokenizer = loaded.tokenizer
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id")
    train_path = project_path(config["train_file"])
    features, selected, skipped = select_features(tokenizer, read_jsonl(train_path), config)
    context = getattr(loaded.model.config, "max_position_embeddings", None)
    if context and max(len(row["input_ids"]) for row in features) > context:
        raise ValueError("Training example exceeds model context")

    model = prepare_model_for_kbit_training(
        loaded.model, use_gradient_checkpointing=config["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", bias="none", **config["lora"]))
    model.config.use_cache = False
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("Only LoRA parameters may be trainable")
    model.print_trainable_parameters()
    initial_digest = adapter_digest(model)
    stamp = datetime.now(timezone.utc).strftime("smoke-%Y%m%dT%H%M%S%fZ")
    run_dir = project_path(config["output_dir"]) / stamp
    run_dir.mkdir(parents=True)
    write_json(run_dir / "config.json", config)
    audit = []
    for row, feature in zip(selected, features):
        boundary = next(i for i, label in enumerate(feature["labels"]) if label != -100)
        audit.append({"sample_id": row["sample_id"], "group_id": row["group_id"],
                      "tokens": len(feature["input_ids"]), "masked_tokens": boundary,
                      "supervised_tokens": len(feature["labels"]) - boundary,
                      "supervised_text": tokenizer.decode(feature["labels"][boundary:],
                                                           skip_special_tokens=False)})
    write_json(run_dir / "mask_audit.json", {"samples": audit, "skipped_overlength": skipped})

    def tensor_batch(indices):
        batch = padded_batch([features[i] for i in indices], tokenizer.eos_token_id)
        return {key: torch.tensor(value, dtype=torch.long, device=loaded.device)
                for key, value in batch.items()}

    dtype = torch.bfloat16 if loaded.compute_dtype == "torch.bfloat16" else torch.float16
    # Independently recompute shifted cross-entropy to check the model honors -100.
    model.eval()
    batch = tensor_batch(range(min(config["batch_size"], len(features))))
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        check = model(**batch)
        expected = F.cross_entropy(check.logits[:, :-1].float().reshape(-1, check.logits.shape[-1]),
                                   batch["labels"][:, 1:].reshape(-1), ignore_index=-100)
    if not torch.isfinite(expected) or not torch.allclose(check.loss.float(), expected, rtol=1e-4, atol=1e-4):
        raise RuntimeError("Model loss does not match masked causal cross-entropy")
    loss_check = {"model_loss": check.loss.item(), "masked_cross_entropy": expected.item()}
    del check, expected, batch
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(loaded.device)
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=config["learning_rate"])
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    rng = random.Random(config["seed"])
    order, cursor = [], 0
    optimizer.zero_grad(set_to_none=True)
    model.train()
    start = time.perf_counter()
    history = []
    with (run_dir / "train_log.jsonl").open("w", encoding="utf-8") as log:
        for step in range(1, config["max_steps"] + 1):
            step_loss = 0.0
            for _ in range(config["gradient_accumulation_steps"]):
                indices = []
                for _ in range(config["batch_size"]):
                    if cursor == len(order):
                        order = list(range(len(features)))
                        rng.shuffle(order)
                        cursor = 0
                    indices.append(order[cursor])
                    cursor += 1
                with torch.autocast("cuda", dtype=dtype):
                    loss = model(**tensor_batch(indices)).loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {step}")
                step_loss += loss.detach().item() / config["gradient_accumulation_steps"]
                scaler.scale(loss / config["gradient_accumulation_steps"]).backward()
            scaler.unscale_(optimizer)
            if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                raise RuntimeError("Frozen parameters unexpectedly received gradients")
            norm = torch.nn.utils.clip_grad_norm_(
                [p for _, p in trainable], config["max_grad_norm"], error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            entry = {"step": step, "loss": step_loss, "grad_norm": norm.item()}
            history.append(entry)
            log.write(json.dumps(entry) + "\n")
            log.flush()
            print(f"step {step}/{config['max_steps']} loss={step_loss:.4f}", flush=True)
    torch.cuda.synchronize(loaded.device)
    train_seconds = time.perf_counter() - start
    peak_mib = torch.cuda.max_memory_allocated(loaded.device) / 1048576
    final_digest = adapter_digest(model)
    if final_digest == initial_digest:
        raise RuntimeError("LoRA weights did not change")
    adapter_dir = run_dir / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_disable()
    system = selected[0]["messages"][0]["content"]
    probe = {"messages": [{"role": "system", "content": system},
                           {"role": "user", "content": config["probe"]["prompt"]},
                           {"role": "assistant", "content": "unused"}]}
    before_reload = probe_answer(model, tokenizer, loaded.device, probe, config["probe"]["max_new_tokens"])
    report = {"status": "trained_pending_reload", "config": config, "model": model_config,
              "resolved_commit": getattr(loaded.model.config, "_commit_hash", None),
              "train_sha256": digest(train_path), "loss_check": loss_check,
              "trainable_parameters": sum(p.numel() for _, p in trainable),
              "trainable_names": [name for name, _ in trainable],
              "initial_adapter_sha256": initial_digest, "adapter_sha256": final_digest,
              "train_seconds": train_seconds, "peak_allocated_mib": peak_mib,
              "loss_history": history, "probe": probe, "before_reload": before_reload,
              "versions": {name: version(name) for name in ("torch", "transformers", "peft", "bitsandbytes")}}
    write_json(run_dir / "train_report.json", report)
    return run_dir


def verify_reload(run_dir: Path) -> dict:
    """Verify the exported smoke/full Adapter against its common train_report."""
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoTokenizer

    report = json.loads((run_dir / "train_report.json").read_text(encoding="utf-8"))
    model_config = dict(report["model"])
    if report["resolved_commit"]:
        model_config["revision"] = report["resolved_commit"]
    loaded = load_base_model(model_config)
    # Match training-time casting of nonquantized layers for a fair reload probe.
    base = prepare_model_for_kbit_training(loaded.model, use_gradient_checkpointing=False)
    model = PeftModel.from_pretrained(base, run_dir / "adapter", is_trainable=False)
    tokenizer = AutoTokenizer.from_pretrained(run_dir / "adapter", local_files_only=True)
    if adapter_digest(model) != report["adapter_sha256"]:
        raise RuntimeError("Reloaded Adapter tensors differ from saved training state")
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("Reload verification must use frozen inference parameters")
    after = probe_answer(model, tokenizer, loaded.device, report["probe"],
                         report["config"]["probe"]["max_new_tokens"])
    tokens_match = after["token_ids"] == report["before_reload"]["token_ids"]
    result = {"status": "passed" if tokens_match else "weights_verified_generation_diff",
              "adapter_weights_match": True, "generation_tokens_match": tokens_match,
              "before_reload": report["before_reload"], "after_reload": after}
    write_json(run_dir / "reload_report.json", result)
    return result
