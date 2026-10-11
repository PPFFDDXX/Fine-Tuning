"""Reusable single-turn generation; one model load per Predictor instance."""
import hashlib
import json
import math
import time
from copy import deepcopy

from bitext_support.training.model_loader import load_base_model
from bitext_support.training.sft import project_path


def validate_generation(options):
    for key in ("max_input_tokens", "max_new_tokens"):
        if type(options.get(key)) is not int or options[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(options.get("do_sample", False)) is not bool:
        raise ValueError("do_sample must be true or false")
    for key, default in (("temperature", 0.7), ("top_p", 0.9), ("repetition_penalty", 1.0)):
        value = options.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be a finite positive number")
    if options.get("top_p", 0.9) > 1:
        raise ValueError("top_p must be at most 1")


def resolve_model(config, kind="base", training_run=None):
    """Validate completed exports and pin the base model to its training revision."""
    if kind not in ("base", "sft", "dpo"):
        raise ValueError("kind must be base, sft or dpo")
    source = deepcopy(config["model"])
    if kind == "base":
        if training_run:
            raise ValueError("Base inference cannot use --training-run")
        return source, None, None
    if not training_run:
        raise ValueError("SFT/DPO inference requires --training-run")
    run = project_path(str(training_run)).resolve()
    for name in ("train_report.json", "adapter/adapter_config.json", "adapter/adapter_model.safetensors",
                 "adapter/tokenizer_config.json"):
        if not (run / name).is_file():
            raise ValueError(f"Incomplete training export: missing {run / name}")
    report = json.loads((run / "train_report.json").read_text(encoding="utf-8"))
    if report.get("kind", "sft") != kind:
        raise ValueError("Run kind does not match requested inference kind")
    if not report.get("adapter_sha256"):
        raise ValueError("Training report has no Adapter fingerprint")
    if report.get("best") and report["best"]["adapter_sha256"] != report["adapter_sha256"]:
        raise ValueError("Export does not match the selected best Adapter")
    source["name_or_path"] = report["model"]["name_or_path"]
    source["revision"] = report.get("resolved_commit") or report["model"].get("revision")
    adapter_config = json.loads((run / "adapter/adapter_config.json").read_text(encoding="utf-8"))
    if adapter_config.get("base_model_name_or_path") != source["name_or_path"]:
        raise ValueError("Adapter base model differs from its training report")
    return source, run, report


def verify_export(run, report):
    """Check saved tensor bytes before PEFT changes inference parameter dtypes."""
    import torch
    from safetensors.torch import load_file
    checksum = hashlib.sha256()
    for name, value in sorted(load_file(str(run / "adapter/adapter_model.safetensors")).items()):
        value = value.detach().cpu().contiguous()
        checksum.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        checksum.update(value.view(torch.uint8).numpy().tobytes())
    if checksum.hexdigest() != report["adapter_sha256"]:
        raise ValueError("Saved Adapter differs from the training export fingerprint")


class Predictor:
    def __init__(self, config, kind="base", training_run=None, system_prompt=""):
        validate_generation(config["generation"])
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("System prompt must contain non-whitespace text")
        self.generation = deepcopy(config["generation"])
        self.system_prompt = system_prompt.strip()
        self.kind = kind
        self.model_config, self.run, report = resolve_model(config, kind, training_run)
        import torch
        if self.run:
            verify_export(self.run, report)
        start = time.perf_counter()
        self.loaded = load_base_model(self.model_config)
        if self.run:
            from peft import PeftModel
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.run / "adapter", local_files_only=True)
            if tokenizer.chat_template != self.loaded.tokenizer.chat_template:
                raise ValueError("Exported tokenizer chat template differs from the original base")
            if tokenizer.get_vocab() != self.loaded.tokenizer.get_vocab():
                raise ValueError("Exported tokenizer vocabulary differs from the original base")
            self.loaded.model = PeftModel.from_pretrained(
                self.loaded.model, self.run / "adapter", is_trainable=False, local_files_only=True)
            self.loaded.tokenizer = tokenizer
        self.loaded.model.requires_grad_(False)
        self.loaded.model.eval()
        torch.cuda.synchronize(self.loaded.device)
        self.load_seconds = time.perf_counter() - start
        self.adapter_sha256 = report["adapter_sha256"] if report else None

    def generate(self, prompt, **overrides):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must contain non-whitespace text")
        options = dict(self.generation)
        unknown = set(overrides) - {"max_input_tokens", "max_new_tokens", "do_sample", "temperature", "top_p", "repetition_penalty"}
        if unknown:
            raise ValueError(f"Unknown generation options: {sorted(unknown)}")
        options.update(overrides)
        validate_generation(options)
        import torch
        from bitext_support.data.formatting import token_ids
        model, tokenizer, device = self.loaded.model, self.loaded.tokenizer, self.loaded.device
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}]
        ids = token_ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
        if len(ids) > options["max_input_tokens"]:
            raise ValueError(f"Prompt has {len(ids)} tokens, exceeding max_input_tokens={options['max_input_tokens']}")
        context = getattr(model.config, "max_position_embeddings", None)
        if context and len(ids) + options["max_new_tokens"] > context:
            raise ValueError("Input plus max_new_tokens exceeds model context length")
        inputs = torch.tensor([ids], dtype=torch.long, device=device)
        kwargs = {"max_new_tokens": options["max_new_tokens"], "do_sample": options.get("do_sample", False),
                  "repetition_penalty": options.get("repetition_penalty", 1.0), "pad_token_id": tokenizer.eos_token_id}
        if kwargs["do_sample"]:
            kwargs.update(temperature=options.get("temperature", 0.7), top_p=options.get("top_p", 0.9))
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), **kwargs)
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - start
        new_ids = output[0, len(ids):].tolist()
        answer = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        if not answer:
            raise RuntimeError("Generation returned an empty answer")
        eos = model.generation_config.eos_token_id
        ended = bool(new_ids and new_ids[-1] in (eos if isinstance(eos, list) else [eos]))
        return {"kind": self.kind, "model": self.model_config,
                "resolved_commit": getattr(model.config, "_commit_hash", None),
                "training_run": str(self.run) if self.run else None, "adapter_sha256": self.adapter_sha256,
                "system_prompt": self.system_prompt, "prompt": prompt, "answer": answer, "generation": options,
                "input_tokens": len(ids), "output_tokens": len(new_ids), "ended_with_eos": ended,
                "hit_token_limit": len(new_ids) >= options["max_new_tokens"],
                "timing_seconds": {"load": round(self.load_seconds, 3), "generate": round(seconds, 3)},
                "tokens_per_second": round(len(new_ids) / seconds, 2) if seconds else None,
                "gpu": torch.cuda.get_device_name(device), "compute_dtype": self.loaded.compute_dtype,
                "quantization": "nf4_double_quant",
                "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 1048576, 2)}
