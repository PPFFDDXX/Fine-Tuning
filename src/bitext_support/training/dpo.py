"""DPO from an exported SFT Adapter with TRL's frozen copy of that Adapter."""
import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import version

from bitext_support.data.formatting import sft_features
from bitext_support.evaluation.generate import digest, read_jsonl
from bitext_support.training.sft import adapter_digest, project_path


def named_digest(model, name):
    import torch
    from peft import get_peft_model_state_dict
    checksum = hashlib.sha256()
    for key, value in sorted(get_peft_model_state_dict(model, adapter_name=name).items()):
        value = value.detach().cpu().contiguous()
        checksum.update(f"{key}:{value.dtype}:{tuple(value.shape)}".encode())
        checksum.update(value.view(torch.uint8).numpy().tobytes())
    return checksum.hexdigest()


def align_reference_adapter(model):
    """Match TRL's frozen reference to the policy after its QLoRA dtype cast."""
    import torch

    policy = {name: param for name, param in model.named_parameters() if ".default." in name}
    if not policy:
        raise RuntimeError("No trainable policy Adapter parameters found")
    with torch.no_grad():
        for name, param in policy.items():
            reference_name = name.replace(".default.", ".ref.")
            reference = model.get_parameter(reference_name)
            if reference.requires_grad:
                raise RuntimeError(f"Reference parameter is trainable: {reference_name}")
            reference.data = param.detach().clone()


def train_dpo(config, training_dir, max_steps=None):
    import torch
    from datasets import Dataset
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoTokenizer
    from trl import DPOConfig, DPOTrainer
    from bitext_support.training.model_loader import load_base_model

    if version("trl") != "1.13.0":
        raise RuntimeError("This workflow is verified against trl==1.13.0; see requirements-ultrafeedback.txt")
    gradient_checkpointing = config.get("gradient_checkpointing", True)
    if not isinstance(gradient_checkpointing, bool):
        raise ValueError("gradient_checkpointing must be true or false")
    for key in ("train_batch_size", "eval_batch_size", "precompute_ref_batch_size"):
        if not isinstance(config.get(key, 1), int) or config.get(key, 1) < 1:
            raise ValueError(f"{key} must be a positive integer")
    report = json.loads((training_dir / "train_report.json").read_text(encoding="utf-8"))
    manifest_path = project_path(config["data_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for split in ("train", "validation"):
        if report["data_sha256"][split] != manifest["files"][f"sft/{split}.jsonl"]:
            raise ValueError("SFT and preference data must derive from the same prepared dataset")
    if report["adapter_sha256"] != report["best"]["adapter_sha256"]:
        raise ValueError("SFT export is not its selected best Adapter")
    tokenizer = AutoTokenizer.from_pretrained(training_dir / "adapter", local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    datasets = {}
    groups = {}
    for split in ("train", "validation"):
        path = manifest_path.parent / "preference" / f"{split}.jsonl"
        if digest(path) != manifest["files"][f"preference/{split}.jsonl"]:
            raise ValueError("Preference file hash mismatch")
        rows = read_jsonl(path)
        groups[split] = {row["group_id"] for row in rows}
        for row in rows:
            for side in ("chosen", "rejected"):
                feature = sft_features(tokenizer, {"messages": row["prompt"] + row[side]})
                if len(feature["input_ids"]) > config["max_length"]:
                    raise ValueError("Prepared data exceed DPO max_length; prepare shorter pairs first")
        datasets[split] = Dataset.from_list([{k: row[k] for k in ("prompt", "chosen", "rejected")}
                                           for row in rows])
    if groups["train"] & groups["validation"]:
        raise ValueError("Cross-split prompt leakage")
    source = dict(report["model"])
    if report.get("resolved_commit"):
        source["revision"] = report["resolved_commit"]
    loaded = load_base_model(source)
    base = prepare_model_for_kbit_training(loaded.model, use_gradient_checkpointing=gradient_checkpointing,
                                           gradient_checkpointing_kwargs={"use_reentrant": False})
    model = PeftModel.from_pretrained(base, training_dir / "adapter", is_trainable=True)
    if adapter_digest(model) != report["adapter_sha256"]:
        raise ValueError("SFT Adapter fingerprint mismatch")
    initial = adapter_digest(model)
    run = project_path(config["output_dir"]) / datetime.now(timezone.utc).strftime("dpo-%Y%m%dT%H%M%S%fZ")
    args = DPOConfig(output_dir=str(run), beta=config["beta"], max_length=config["max_length"],
        learning_rate=config["learning_rate"], num_train_epochs=config["epochs"],
        max_steps=max_steps if max_steps is not None else -1,
        per_device_train_batch_size=config.get("train_batch_size", 1),
        per_device_eval_batch_size=config.get("eval_batch_size", 1),
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=loaded.compute_dtype == "torch.bfloat16", fp16=loaded.compute_dtype == "torch.float16",
        eval_strategy="steps", eval_steps=config["eval_steps"], save_strategy="steps",
        save_steps=config["eval_steps"], save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        precompute_ref_log_probs=True,
        precompute_ref_batch_size=config.get("precompute_ref_batch_size", 1),
        logging_steps=10, report_to="none", seed=config["seed"], dataloader_num_workers=0)
    trainer = DPOTrainer(model=model, args=args, processing_class=tokenizer,
                         train_dataset=datasets["train"], eval_dataset=datasets["validation"])
    if "ref" not in model.peft_config:
        raise RuntimeError("TRL did not create the frozen SFT reference Adapter")
    align_reference_adapter(model)
    reference = named_digest(model, "ref")
    if reference != named_digest(model, "default"):
        raise RuntimeError("Initial policy and reference differ")
    if any(p.requires_grad for n, p in model.named_parameters() if ".ref." in n):
        raise RuntimeError("Reference Adapter must be frozen")
    result = trainer.train()
    last_eval = next((entry for entry in reversed(trainer.state.log_history)
                      if "eval_loss" in entry), None)
    metrics = ({key: value for key, value in last_eval.items() if key.startswith("eval_")}
               if last_eval and last_eval.get("step") == trainer.state.global_step
               else trainer.evaluate())
    if named_digest(model, "ref") != reference:
        raise RuntimeError("Reference Adapter changed during training")
    model.set_adapter("default")
    model.save_pretrained(run / "adapter", selected_adapters=["default"])
    tokenizer.save_pretrained(run / "adapter")
    final = adapter_digest(model)
    output = {"kind": "dpo", "config": config, "model": source,
              "resolved_commit": report.get("resolved_commit"),
              "data_sha256": report["data_sha256"], "adapter_sha256": final,
              "initial_adapter_sha256": initial, "reference_adapter_sha256": reference,
              "best": {"adapter_sha256": final, "checkpoint": trainer.state.best_model_checkpoint},
              "training_metrics": result.metrics, "validation_metrics": metrics,
              "sft_run": str(training_dir), "sft_report_sha256": digest(training_dir / "train_report.json"),
              "data_manifest_sha256": digest(manifest_path),
              "versions": {k: version(k) for k in ("torch", "transformers", "peft", "trl")}}
    (run / "train_report.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return run
