"""Load the original instruct checkpoint in verified 4-bit mode."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class LoadedModel:
    model: object
    tokenizer: object
    device: object
    compute_dtype: str
    linear4bit_count: int


def load_base_model(config: dict) -> LoadedModel:
    import torch
    from bitsandbytes.nn import Linear4bit
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for 4-bit inference; CPU fallback is disabled.")

    device_id = int(config.get("device", 0))
    if device_id < 0 or device_id >= torch.cuda.device_count():
        raise ValueError(f"Invalid CUDA device {device_id}; found {torch.cuda.device_count()} device(s).")
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    dtype_option = config.get("compute_dtype", "auto")
    if dtype_option not in ("auto", "bfloat16", "float16"):
        raise ValueError("compute_dtype must be auto, bfloat16, or float16")
    if dtype_option == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support bfloat16.")
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if dtype_option == "auto" else getattr(torch, dtype_option)

    source = config["name_or_path"]
    common = {
        "local_files_only": bool(config.get("local_files_only", True)),
        "trust_remote_code": bool(config.get("trust_remote_code", False)),
    }
    if config.get("cache_dir"):
        cache_dir = Path(config["cache_dir"])
        if not cache_dir.is_absolute():
            cache_dir = PROJECT_ROOT / cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        common["cache_dir"] = str(cache_dir)
    if config.get("revision"):
        common["revision"] = config["revision"]
    tokenizer = AutoTokenizer.from_pretrained(source, **common)
    if not tokenizer.chat_template:
        raise RuntimeError("Tokenizer has no chat template; refusing to invent one.")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        source, quantization_config=quantization, device_map={"": device_id},
        torch_dtype=dtype, **common,
    )
    count = sum(isinstance(module, Linear4bit) for module in model.modules())
    if not getattr(model, "is_loaded_in_4bit", False) or count == 0:
        raise RuntimeError("4-bit loading could not be verified.")
    model.eval()
    return LoadedModel(model, tokenizer, device, str(dtype), count)
