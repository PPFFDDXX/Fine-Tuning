"""Check customer-support SFT masks, EOS, and length before training."""

import argparse
import json
import sys
from pathlib import Path

import yaml
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bitext_support.data.formatting import generation_ids, sft_features  # noqa: E402
from bitext_support.evaluation.generate import read_jsonl  # noqa: E402
from bitext_support.training.sft import padded_batch  # noqa: E402
from bitext_support.training.sft_full import encode_split  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sft_customer_support_v2.yaml")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    model = yaml.safe_load((ROOT / config["model_config"]).read_text(encoding="utf-8"))["model"]
    common = {"cache_dir": str((ROOT / model["cache_dir"]).resolve()),
              "revision": model["revision"], "local_files_only": True,
              "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(model["name_or_path"], **common)
    model_config = AutoConfig.from_pretrained(model["name_or_path"], **common)
    generation = GenerationConfig.from_pretrained(model["name_or_path"], **common)
    stop_ids = generation.eos_token_id
    stop_ids = stop_ids if isinstance(stop_ids, list) else [stop_ids]
    if tokenizer.eos_token_id not in stop_ids:
        raise ValueError("Tokenizer EOS is absent from generation stop IDs")
    limit = min(config["max_seq_length"], model_config.max_position_embeddings)
    results = {}
    for split in ("train", "validation"):
        rows = read_jsonl(ROOT / config[f"{split}_file"])
        features, audit = encode_split(tokenizer, rows, limit, split)
        if audit["kept_count"] != len(rows):
            raise ValueError(f"{split}: {len(rows) - audit['kept_count']} overlength rows")
        lengths = []
        for row in rows:
            feature = sft_features(tokenizer, row)
            padded_batch([feature], tokenizer.eos_token_id)
            ids, labels = feature["input_ids"], feature["labels"]
            boundary = next(i for i, label in enumerate(labels) if label != -100)
            if ids[:boundary] != generation_ids(tokenizer, row):
                raise ValueError(f"{split}/{row['sample_id']}: wrong prompt boundary")
            if any(label != -100 for label in labels[:boundary]) or labels[boundary:] != ids[boundary:]:
                raise ValueError(f"{split}/{row['sample_id']}: incorrect supervision mask")
            eos_positions = [i for i in range(boundary, len(ids)) if ids[i] == tokenizer.eos_token_id]
            if not eos_positions or tokenizer.decode(ids[eos_positions[-1] + 1:], skip_special_tokens=False).strip():
                raise ValueError(f"{split}/{row['sample_id']}: missing EOS or nonblank suffix")
            lengths.append(len(ids))
        results[split] = {"rows": len(rows), "kept": len(features), "max_tokens": max(lengths),
                          "eos_ok": len(rows)}
    print(json.dumps({"revision": model["revision"], "limit": limit,
                      "eos_id": tokenizer.eos_token_id, "splits": results}, indent=2))


if __name__ == "__main__":
    main()
