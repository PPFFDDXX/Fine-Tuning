"""Prepare, audit, or publish reviewed customer-support SFT data."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bitext_support.data.customer_support import audit, file_hash, prepare, read_rows, load_review_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "audit", "publish"))
    parser.add_argument("--config", default="configs/customer_support_data_v2.json")
    parser.add_argument("--output", help="Override output directory inside the project")
    parser.add_argument("--local-jsonl", help="Offline source with instruction and intent fields")
    parser.add_argument("--review-from", help="Reuse matching reviews and exclude rejected candidates")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    output = (ROOT / (args.output or config["output_dir"])).resolve()
    if ROOT not in output.parents:
        raise ValueError("Output must be inside this project")
    if args.command == "prepare":
        if output.exists():
            raise ValueError(f"Output already exists: {output}; choose --output")
        if args.local_jsonl:
            path = (ROOT / args.local_jsonl).resolve()
            raw = read_rows(path)
            provenance = {"type": "local", "path": str(path), "sha256": file_hash(path), "license": "not verified"}
        else:
            from datasets import load_dataset
            from huggingface_hub import HfApi
            info = HfApi().dataset_info(config["repo_id"], revision=config["revision"])
            raw = load_dataset(config["repo_id"], revision=info.sha, split=config["source_split"], cache_dir=str(ROOT / "data/cache"))
            provenance = {"repo_id": config["repo_id"], "revision": info.sha, "split": config["source_split"],
                          "license": getattr(info.card_data, "license", None)}
        previous = load_review_history(ROOT / args.review_from) if args.review_from else []
        report = prepare(raw, config, output, provenance, previous)
    else:
        report = audit(output, publish=args.command == "publish")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Artifacts: {output}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
