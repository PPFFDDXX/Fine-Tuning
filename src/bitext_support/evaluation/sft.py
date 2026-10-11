"""Load and fingerprint exported SFT or DPO Adapters for evaluation."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from bitext_support.data.formatting import generation_ids
from bitext_support.evaluation.generate import digest
from bitext_support.training.sft import adapter_digest


def runtime_metadata(loaded) -> dict:
    import torch
    return {"resolved_commit": getattr(loaded.model.config, "_commit_hash", None),
            "chat_template_sha256": hashlib.sha256(json.dumps(
                loaded.tokenizer.chat_template, sort_keys=True).encode()).hexdigest(),
            "compute_dtype": loaded.compute_dtype,
            "gpu": torch.cuda.get_device_name(loaded.device),
            "versions": {name: version(name) for name in ("torch", "transformers", "bitsandbytes")},
            "quantization": "nf4_double_quant",
            "timing_protocol": "serial_no_warmup_v2"}




def attach_adapter(loaded, training_dir: Path, report: dict, selected: list[dict], examples: list[dict]):
    from peft import PeftModel
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(training_dir / "adapter", local_files_only=True)
    for row in selected:
        if generation_ids(tokenizer, row, examples) != generation_ids(loaded.tokenizer, row, examples):
            raise ValueError(f"Saved SFT tokenizer changes the evaluation prompt: {row['sample_id']}")
    # Use the same inference-time base precision as Base evaluation.
    loaded.model = PeftModel.from_pretrained(loaded.model, training_dir / "adapter", is_trainable=False)
    loaded.model.requires_grad_(False)
    loaded.model.eval()
    loaded.tokenizer = tokenizer
    if adapter_digest(loaded.model) != report["adapter_sha256"]:
        raise ValueError("Loaded Adapter differs from the selected training export")
    return {"training_dir": str(training_dir.resolve()),
            "train_report_sha256": digest(training_dir / "train_report.json"),
            "adapter_sha256": report["adapter_sha256"], "best": report["best"],
            "peft_version": version("peft"), "data_revision": report.get("data_revision")}
