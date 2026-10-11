"""Compare Base and SFT scores on the same approved examples."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("format_valid_rate", "intent_accuracy", "intent_macro_f1",
           "order_id_exact_accuracy", "order_id_present_f1", "joint_accuracy")


def load(value):
    path = ROOT / value
    report = json.loads((path / "report.json").read_text(encoding="utf-8"))
    content = (path / "predictions.jsonl").read_bytes()
    if hashlib.sha256(content).hexdigest() != report["predictions_sha256"]:
        raise ValueError(f"Predictions changed after scoring: {path}")
    predictions = [json.loads(line) for line in content.decode("utf-8").splitlines()]
    if len(predictions) != report["n"]:
        raise ValueError(f"Incomplete predictions: {path}")
    return report, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="outputs/customer_support/base_validation_pinned")
    parser.add_argument("--sft", required=True, help="Completed SFT evaluation output directory")
    args = parser.parse_args()
    base, base_rows = load(args.base)
    sft, sft_rows = load(args.sft)
    b, s = base["provenance"], sft["provenance"]
    if b["model_kind"] != "base" or s["model_kind"] != "sft":
        raise ValueError("Expected Base and SFT reports")
    for key in ("split", "data_sha256", "model_config_sha256", "requested_max_new_tokens", "limit"):
        if b.get(key) != s.get(key):
            raise ValueError(f"Evaluation settings differ: {key}")
    provenance_keys = (("source_sha256", "selection_sha256", "system_prompt_sha256") if b["split"] == "external"
                       else ("manifest_sha256", "review_sha256"))
    for key in provenance_keys:
        if not b.get(key) or b.get(key) != s.get(key):
            raise ValueError(f"Evaluation provenance differs: {key}")
    if [(r["sample_id"], r["target"], r["text"]) for r in base_rows] != [
            (r["sample_id"], r["target"], r["text"]) for r in sft_rows]:
        raise ValueError("Base and SFT predictions use different examples or order")
    print("Metric                         Base      SFT    Change")
    for key in METRICS:
        print(f"{key:29} {base[key]:7.3f} {sft[key]:8.3f} {sft[key]-base[key]:+9.3f}")


if __name__ == "__main__":
    main()
