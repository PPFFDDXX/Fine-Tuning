"""Full QLoRA SFT with token-weighted validation and resumable checkpoints."""

from __future__ import annotations

import json
import math
import random
import shutil
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from bitext_support.data.formatting import sft_features
from bitext_support.evaluation.generate import digest, read_jsonl
from bitext_support.training.model_loader import load_base_model
from bitext_support.training.sft import (
    adapter_digest, padded_batch, probe_answer, project_path, write_json,
)


def validate_full_config(config: dict) -> None:
    for key in ("epochs", "max_seq_length", "batch_size", "eval_batch_size",
                "gradient_accumulation_steps", "eval_steps", "save_steps", "logging_steps"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("learning_rate", "max_grad_norm"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not 0 <= config["warmup_ratio"] < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if not math.isfinite(config["weight_decay"]) or config["weight_decay"] < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if not config["probe"]["prompt"].strip() or config["probe"]["max_new_tokens"] < 1:
        raise ValueError("Invalid probe configuration")


def encode_split(tokenizer, rows: list[dict], limit: int, split: str) -> tuple[list[dict], dict]:
    features, kept, skipped = [], [], []
    for index, row in enumerate(rows):
        if not row["messages"][-1]["content"].strip():
            raise ValueError(f"Empty answer in {split}: {row['sample_id']}")
        feature = sft_features(tokenizer, row)
        if len(feature["input_ids"]) > limit:
            skipped.append(row["sample_id"])
        else:
            padded_batch([feature], tokenizer.eos_token_id)  # Validate every mask.
            features.append(feature)
            kept.append(row["sample_id"])
        if (index + 1) % 1000 == 0:
            print(f"encode {split}: {index + 1}/{len(rows)}", flush=True)
    if not features:
        raise ValueError(f"No eligible examples in {split}")
    return features, {"source_count": len(rows), "kept_count": len(kept),
                      "kept_ids": kept, "skipped_overlength_ids": skipped,
                      "supervised_tokens": sum(supervised_count(f) for f in features)}


def supervised_count(feature: dict) -> int:
    # Causal LM shifts labels by one position before computing loss.
    return sum(label != -100 for label in feature["labels"][1:])


def batch_tensors(features, pad_id, device):
    import torch
    return {key: torch.tensor(value, dtype=torch.long, device=device)
            for key, value in padded_batch(features, pad_id).items()}


def evaluate_loss(model, features, batch_size, pad_id, device, dtype) -> dict:
    import torch
    was_training = model.training
    model.eval()
    total_loss, total_tokens = 0.0, 0
    start = time.perf_counter()
    try:
        with torch.no_grad():
            for offset in range(0, len(features), batch_size):
                records = features[offset:offset + batch_size]
                count = sum(supervised_count(row) for row in records)
                with torch.autocast("cuda", dtype=dtype):
                    loss = model(**batch_tensors(records, pad_id, device)).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite validation loss")
                total_loss += loss.item() * count
                total_tokens += count
                if (offset // batch_size + 1) % 100 == 0:
                    print(f"validation: {min(offset + batch_size, len(features))}/{len(features)}", flush=True)
    finally:
        model.train(was_training)
    return {"loss": total_loss / total_tokens, "supervised_tokens": total_tokens,
            "samples": len(features), "seconds": time.perf_counter() - start}


def train_full(config: dict, resume: Path | None = None) -> Path:
    import torch
    import torch.nn.functional as F
    import yaml
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    validate_full_config(config)
    previous = None
    if resume:
        previous = json.loads((resume / "checkpoint.json").read_text(encoding="utf-8"))
        if previous["config"] != config:
            raise ValueError("Resume requires the original full-training configuration")
    versions = {name: version(name) for name in ("torch", "transformers", "peft", "bitsandbytes")}
    if previous and previous["versions"] != versions:
        raise ValueError("Restore the checkpoint's package versions before resuming")
    model_config = yaml.safe_load(project_path(config["model_config"]).read_text(encoding="utf-8"))["model"]
    if previous and model_config != previous["model"]:
        raise ValueError("Model configuration changed since checkpoint")
    hashes = {split: digest(project_path(config[f"{split}_file"])) for split in ("train", "validation")}
    if previous and hashes != previous["data_sha256"]:
        raise ValueError("Train or validation file changed since checkpoint")
    rows = {split: read_jsonl(project_path(config[f"{split}_file"])) for split in hashes}
    for split, records in rows.items():
        if not records or len({row["sample_id"] for row in records}) != len(records):
            raise ValueError(f"Empty split or duplicate sample IDs: {split}")
    for key in ("sample_id", "group_id"):
        if {r[key] for r in rows["train"]} & {r[key] for r in rows["validation"]}:
            raise ValueError(f"Train/validation overlap: {key}")
    systems = {r["messages"][0]["content"] for records in rows.values() for r in records}
    if len(systems) != 1:
        raise ValueError("Expected one fixed system prompt across train and validation")
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    source = dict(model_config)
    if previous and previous["resolved_commit"]:
        source["revision"] = previous["resolved_commit"]
    loaded = load_base_model(source)
    tokenizer = (AutoTokenizer.from_pretrained(resume / "adapter", local_files_only=True)
                 if resume else loaded.tokenizer)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define EOS")
    features, data_audit = {}, {}
    context = getattr(loaded.model.config, "max_position_embeddings", None)
    limit = min(config["max_seq_length"], context) if context else config["max_seq_length"]
    for split in rows:
        features[split], data_audit[split] = encode_split(tokenizer, rows[split], limit, split)
    probe = {"messages": [{"role": "system", "content": next(iter(systems))},
                           {"role": "user", "content": config["probe"]["prompt"]},
                           {"role": "assistant", "content": "unused"}]}
    del rows
    base = prepare_model_for_kbit_training(
        loaded.model, use_gradient_checkpointing=config["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model = (PeftModel.from_pretrained(base, resume / "adapter", is_trainable=True) if resume
             else get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", bias="none", **config["lora"])))
    model.config.use_cache = False
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("Only LoRA parameters may be trainable")
    model.print_trainable_parameters()
    loaded_digest = adapter_digest(model)
    if previous and loaded_digest != previous["adapter_sha256"]:
        raise RuntimeError("Checkpoint Adapter tensors differ")
    initial_digest = previous["initial_adapter_sha256"] if previous else loaded_digest
    dtype = torch.bfloat16 if loaded.compute_dtype == "torch.bfloat16" else torch.float16
    if previous and previous["compute_dtype"] != loaded.compute_dtype:
        raise ValueError("Compute dtype changed since checkpoint")

    # Check the actual model's causal mask semantics before any updates.
    model.eval()
    batch = batch_tensors(features["train"][:1], tokenizer.eos_token_id, loaded.device)
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        output = model(**batch)
        expected = F.cross_entropy(output.logits[:, :-1].float().reshape(-1, output.logits.shape[-1]),
                                   batch["labels"][:, 1:].reshape(-1), ignore_index=-100)
    if not torch.isfinite(expected) or not torch.allclose(output.loss.float(), expected, rtol=1e-4, atol=1e-4):
        raise RuntimeError("Model loss differs from masked causal cross-entropy")
    loss_check = {"model_loss": output.loss.item(), "masked_cross_entropy": expected.item()}
    del output, expected, batch
    torch.cuda.empty_cache()
    micro_batches = math.ceil(len(features["train"]) / config["batch_size"])
    updates_per_epoch = math.ceil(micro_batches / config["gradient_accumulation_steps"])
    total_steps = config["epochs"] * updates_per_epoch
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=min(math.ceil(total_steps * config["warmup_ratio"]), total_steps - 1),
        num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    stamp = datetime.now(timezone.utc).strftime("full-%Y%m%dT%H%M%S%fZ")
    run_dir = project_path(config["output_dir"]) / stamp
    run_dir.mkdir(parents=True)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "data_audit.json", data_audit)
    best = previous["best"] if previous else None
    best_dir = None
    step = previous["step"] if previous else 0
    start_epoch = previous["next_epoch"] if previous else 0
    start_batch = previous["next_batch"] if previous else 0
    if previous and best:
        best_dir = run_dir / "best_from_resume"
        shutil.copytree(resume / "best_adapter", best_dir)
    if previous:
        if previous["total_steps"] != total_steps or previous["trainable_names"] != [n for n, _ in trainable]:
            raise ValueError("Training plan or trainable parameter ordering changed")
        state = torch.load(resume / "training_state.pt", map_location="cpu", weights_only=True)
        scheduler.load_state_dict(state["scheduler"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        del state
    optimizer.zero_grad(set_to_none=True)
    model.train()
    torch.cuda.reset_peak_memory_stats(loaded.device)
    start_time = time.perf_counter()
    prior_seconds = previous["elapsed_seconds"] if previous else 0.0
    prior_peak = previous["peak_allocated_mib"] if previous else 0.0
    metadata = {"format_version": 1, "config": config, "model": model_config,
                "resolved_commit": getattr(loaded.model.config, "_commit_hash", None),
                "compute_dtype": loaded.compute_dtype, "versions": versions, "data_sha256": hashes,
                "initial_adapter_sha256": initial_digest, "total_steps": total_steps,
                "trainable_names": [n for n, _ in trainable],
                "trainable_parameters": sum(p.numel() for _, p in trainable),
                "resumed_from": str(resume) if resume else None}
    write_json(run_dir / "run_metadata.json", metadata)

    def save_checkpoint(next_epoch, next_batch):
        destination = run_dir / "checkpoints" / f"step-{step:08d}"
        temporary = destination.with_name(destination.name + ".incomplete")
        temporary.mkdir(parents=True)
        model.save_pretrained(temporary / "adapter", safe_serialization=True)
        tokenizer.save_pretrained(temporary / "adapter")
        if best_dir:
            shutil.copytree(best_dir, temporary / "best_adapter")
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(), "python_rng": random.getstate(),
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()},
                   temporary / "training_state.pt")
        info = {**metadata, "step": step, "next_epoch": next_epoch, "next_batch": next_batch,
                "best": best, "adapter_sha256": adapter_digest(model),
                "elapsed_seconds": prior_seconds + time.perf_counter() - start_time,
                "peak_allocated_mib": max(prior_peak, torch.cuda.max_memory_allocated(loaded.device) / 1048576)}
        write_json(temporary / "checkpoint.json", info)
        temporary.rename(destination)
        write_json(run_dir / "latest_checkpoint.json", {"path": str(destination.resolve())})
        print(f"checkpoint: {destination}", flush=True)

    with (run_dir / "train_log.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(start_epoch, config["epochs"]):
            order = list(range(len(features["train"])))
            random.Random(config["seed"] + epoch).shuffle(order)
            batches = [order[i:i + config["batch_size"]] for i in range(0, len(order), config["batch_size"])]
            offset = start_batch if epoch == start_epoch else 0
            while offset < len(batches):
                window = batches[offset:offset + config["gradient_accumulation_steps"]]
                token_total = sum(supervised_count(features["train"][i]) for indices in window for i in indices)
                window_loss = 0.0
                lr_used = optimizer.param_groups[0]["lr"]
                for indices in window:
                    records = [features["train"][i] for i in indices]
                    count = sum(supervised_count(row) for row in records)
                    with torch.autocast("cuda", dtype=dtype):
                        loss = model(**batch_tensors(records, tokenizer.eos_token_id, loaded.device)).loss
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite training loss before step {step + 1}")
                    window_loss += loss.detach().item() * count / token_total
                    scaler.scale(loss * (count / token_total)).backward()
                scaler.unscale_(optimizer)
                if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                    raise RuntimeError("Frozen base unexpectedly received gradients")
                norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], config["max_grad_norm"],
                                                     error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                offset += len(window)
                at_epoch_end = offset == len(batches)
                entry = {"step": step, "epoch": epoch + offset / len(batches), "train_loss": window_loss,
                         "supervised_tokens": token_total, "learning_rate": lr_used, "grad_norm": norm.item()}
                evaluate = step % config["eval_steps"] == 0 or at_epoch_end
                if evaluate:
                    entry["validation"] = evaluate_loss(model, features["validation"], config["eval_batch_size"],
                                                         tokenizer.eos_token_id, loaded.device, dtype)
                    val_loss = entry["validation"]["loss"]
                    print(f"step {step}: validation_loss={val_loss:.6f}", flush=True)
                    if best is None or val_loss < best["validation_loss"]:
                        best_dir = run_dir / "selection" / f"step-{step:08d}"
                        model.save_pretrained(best_dir, safe_serialization=True)
                        tokenizer.save_pretrained(best_dir)
                        best = {"step": step, "validation_loss": val_loss, "adapter_sha256": adapter_digest(model)}
                entry["elapsed_seconds"] = prior_seconds + time.perf_counter() - start_time
                entry["peak_allocated_mib"] = max(prior_peak, torch.cuda.max_memory_allocated(loaded.device) / 1048576)
                log.write(json.dumps(entry) + "\n")
                log.flush()
                if step % config["logging_steps"] == 0 or at_epoch_end:
                    print(f"step {step}/{total_steps} epoch={entry['epoch']:.3f} loss={window_loss:.4f} lr={lr_used:.3g}", flush=True)
                if evaluate or step % config["save_steps"] == 0:
                    save_checkpoint(epoch + 1 if at_epoch_end else epoch, 0 if at_epoch_end else offset)
    if best_dir is None:
        raise RuntimeError("No validation checkpoint available")
    if best["adapter_sha256"] == initial_digest:
        raise RuntimeError("Selected Adapter has not changed from initialization")
    # Export the best validation checkpoint, not necessarily the final training step.
    shutil.copytree(best_dir, run_dir / "adapter")
    set_peft_model_state_dict(model, load_file(str(run_dir / "adapter" / "adapter_model.safetensors")))
    if adapter_digest(model) != best["adapter_sha256"]:
        raise RuntimeError("Best Adapter failed to load for the export probe")
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_disable()
    before_reload = probe_answer(model, tokenizer, loaded.device, probe, config["probe"]["max_new_tokens"])
    report = {**metadata, "status": "trained_pending_reload", "best": best, "completed_steps": step,
              "adapter_sha256": best["adapter_sha256"], "loss_check": loss_check, "probe": probe,
              "before_reload": before_reload, "data_audit_file": "data_audit.json",
              "elapsed_seconds": prior_seconds + time.perf_counter() - start_time,
              "peak_allocated_mib": max(prior_peak, torch.cuda.max_memory_allocated(loaded.device) / 1048576)}
    write_json(run_dir / "train_report.json", report)
    return run_dir
