"""Generate Base or SFT predictions on an approved validation or test split."""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bitext_support.data.customer_support import file_hash, read_rows
from bitext_support.evaluation.customer_support import score


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def approved_rows(data_root, split_name):
    audit = json.loads((data_root / "audit.json").read_text(encoding="utf-8"))
    if audit.get("content_approved") is not True:
        raise ValueError("Dataset content review is not approved")
    if file_hash(data_root / "review_queue.jsonl") != audit["review_sha256"]:
        raise ValueError("Review queue changed since the approved audit")
    split = data_root / "sft" / f"{split_name}.jsonl"
    rows = read_rows(split)
    if len(rows) != audit["counts"][split_name]:
        raise ValueError(f"{split_name} count changed")
    candidates = [r for r in read_rows(data_root / "candidates.jsonl") if r["split"] == split_name]
    if rows != candidates:
        raise ValueError(f"{split_name} rows changed")
    if any(r["messages"][-1]["content"] != json.dumps(r["target"], ensure_ascii=False, sort_keys=True, separators=(",", ":")) for r in rows):
        raise ValueError("Gold message differs from target")
    return rows, split


def external_rows(data_root):
    split = data_root / "selected_50.jsonl"
    rows = read_rows(split)
    spec = json.loads((ROOT / "configs/external_twitter_50_labels.json").read_text(encoding="utf-8"))
    expected = [(tweet_id, intent, order_id) for intent, items in spec["labels"].items()
                for tweet_id, order_id in items.items()]
    if len(rows) != 50 or len(expected) != 50 or len({r["sample_id"] for r in rows}) != 50:
        raise ValueError("External set must contain 50 unique reviewed examples")
    actual = [(r["source_tweet_id"], r["target"]["intent"], r["target"]["order_id"]) for r in rows]
    if actual != expected or any(r["sample_id"] != f'twcs_{r["source_tweet_id"]}' for r in rows):
        raise ValueError("External rows differ from reviewed labels or ordering")
    if any(r["target"]["order_id"] is not None and r["target"]["order_id"] not in r["text"] for r in rows):
        raise ValueError("External order ID is not in source text")
    system = (ROOT / "prompts/system_customer_support.txt").read_text(encoding="utf-8").strip()
    return [{**r, "messages": [{"role": "system", "content": system}]} for r in rows], split, spec


def write_report(output, predictions, metadata):
    metrics, scored = score(predictions)
    times = [r["timing_seconds"]["generate"] for r in predictions
             if isinstance(r.get("timing_seconds"), dict) and isinstance(r["timing_seconds"].get("generate"), (int, float))]
    tokens = [r["output_tokens"] for r in predictions if isinstance(r.get("output_tokens"), int)]
    metrics["runtime"] = {"timed_examples": len(times),
                          "mean_generation_seconds": statistics.mean(times) if times else None,
                          "median_generation_seconds": statistics.median(times) if times else None,
                          "total_generation_seconds": sum(times),
                          "total_output_tokens": sum(tokens),
                          "output_tokens_per_second": sum(tokens) / sum(times) if times and sum(times) else None,
                          "peak_allocated_mib_max": max((r["peak_allocated_mib"] for r in predictions
                                                          if isinstance(r.get("peak_allocated_mib"), (int, float))), default=None)}
    scored_path = output / "scored_predictions.jsonl"
    with scored_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in scored:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics["provenance"] = metadata
    metrics["predictions_sha256"] = file_hash(output / "predictions.jsonl")
    metrics["scored_predictions_sha256"] = file_hash(scored_path)
    dump(output / "report.json", metrics)
    lines = [f"# Customer support {metadata['model_kind'].upper()} {metadata['split']}", "", f"Scored {metrics['n']} {metadata['split']} examples.", "",
             "| Metric | Value |", "|---|---:|",
             f"| JSON/schema valid | {metrics['format_valid_rate']:.1%} |",
             f"| Intent accuracy | {metrics['intent_accuracy']:.1%} |",
             f"| Intent Macro-F1 | {metrics['intent_macro_f1']:.3f} |",
             f"| Order ID exact accuracy | {metrics['order_id_exact_accuracy']:.1%} |",
             f"| Present order ID F1 | {metrics['order_id_present_f1']:.3f} |",
             f"| Joint exact accuracy | {metrics['joint_accuracy']:.1%} |", "",
             "Invalid JSON/schema counts as incorrect for all end-to-end task metrics. Present-order F1 uses exact ID matches; a wrong ID creates one false positive and one false negative.", "",
             "## Per-intent errors", "", "| Intent | Support | Intent correct | Order ID correct | Joint correct |", "|---|---:|---:|---:|---:|"]
    for label, item in metrics["by_intent"].items():
        lines.append(f"| {label} | {item['n']} | {item['intent_correct']} | {item['slot_correct']} | {item['joint_correct']} |")
    if times:
        lines += ["", "## Runtime", "", f"Mean generation latency: {metrics['runtime']['mean_generation_seconds']:.3f} s/example.",
                  f"Generation throughput: {metrics['runtime']['output_tokens_per_second']:.1f} output tokens/s."]
    source_note = ("This external set contains real English customer tweets and Amazon-style order identifiers."
                   if metadata["split"] == "external" else
                   "This split contains synthetic English support messages and ORD-formatted identifiers.")
    lines += ["", "See `scored_predictions.jsonl` for every prompt, answer, and error flag. " + source_note, ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "score"))
    parser.add_argument("--data-root", default="data/processed/customer_support/v2")
    parser.add_argument("--model-config", default="configs/inference.yaml")
    parser.add_argument("--output", default="outputs/customer_support/base_validation")
    parser.add_argument("--kind", choices=("base", "sft"), default="base")
    parser.add_argument("--split", choices=("validation", "test", "external"), default="validation")
    parser.add_argument("--training-run", help="Completed full SFT run directory; required for --kind sft")
    parser.add_argument("--limit", type=int, help="Optional smoke subset; use another output directory")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()
    if (args.kind == "sft") != bool(args.training_run):
        parser.error("--kind sft requires --training-run; Base must not use it")
    if args.split in ("test", "external") and args.limit is not None:
        parser.error("Test and external evaluation must use the complete split")
    data_root = (ROOT / ("data/external/customer_support_twitter" if args.split == "external" else args.data_root)).resolve()
    output = (ROOT / args.output).resolve()
    if ROOT.resolve() not in output.parents:
        raise ValueError("Output must stay inside project")
    if args.split == "external":
        rows, split, external_spec = external_rows(data_root)
    else:
        rows, split = approved_rows(data_root, args.split)
    if args.limit is not None:
        if args.limit < 1 or args.limit > len(rows):
            raise ValueError("limit must be within validation count")
        rows = rows[:args.limit]
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "predictions.jsonl"
    metadata = {"split": args.split, "model_kind": args.kind, "data_sha256": file_hash(split),
                "model_config_sha256": file_hash(ROOT / args.model_config),
                "requested_max_new_tokens": args.max_new_tokens, "limit": args.limit}
    if args.split == "external":
        source_manifest = json.loads((data_root / "source_manifest.json").read_text(encoding="utf-8"))
        if source_manifest["sha256"] != external_spec["source_sha256"]:
            raise ValueError("External source version differs from reviewed labels")
        metadata["source_sha256"] = external_spec["source_sha256"]
        metadata["selection_sha256"] = file_hash(ROOT / "configs/external_twitter_50_labels.json")
        metadata["system_prompt_sha256"] = file_hash(ROOT / "prompts/system_customer_support.txt")
    else:
        metadata["manifest_sha256"] = file_hash(data_root / "manifest.json")
        metadata["review_sha256"] = file_hash(data_root / "review_queue.jsonl")
    if args.kind == "sft":
        run = (ROOT / args.training_run).resolve()
        if not (run / "train_report.json").is_file():
            raise ValueError(f"Completed SFT export missing: {run / 'train_report.json'}")
        training = json.loads((run / "train_report.json").read_text(encoding="utf-8"))
        if training.get("status") != "trained_pending_reload":
            raise ValueError("SFT run is incomplete")
        if args.split == "validation" and training.get("data_sha256", {}).get("validation") != metadata["data_sha256"]:
            raise ValueError("SFT used a different validation split")
        training_data_root = ROOT / args.data_root if args.split == "external" else data_root
        train_path = training_data_root / "sft" / "train.jsonl"
        val_path = training_data_root / "sft" / "validation.jsonl"
        if training.get("data_sha256") != {"train": file_hash(train_path), "validation": file_hash(val_path)}:
            raise ValueError("SFT was trained on a different dataset")
        reload_path = run / "reload_report.json"
        if not reload_path.is_file() or json.loads(reload_path.read_text(encoding="utf-8")).get("status") != "passed":
            raise ValueError("SFT Adapter must pass independent reload verification")
        metadata["training_run"] = str(run.relative_to(ROOT))
        metadata["train_report_sha256"] = file_hash(run / "train_report.json")
        metadata["adapter_sha256"] = training["adapter_sha256"]
    meta_path = output / "run_config.json"
    if meta_path.exists():
        if json.loads(meta_path.read_text(encoding="utf-8")) != metadata:
            raise ValueError("Existing output uses different data or generation config")
    elif predictions_path.exists():
        raise ValueError("Existing predictions have no run configuration")
    else:
        dump(meta_path, metadata)
    predictions = read_rows(predictions_path) if predictions_path.exists() else []
    if len(predictions) > len(rows) or any(p["sample_id"] != r["sample_id"] or p["target"] != r["target"] for p, r in zip(predictions, rows)):
        raise ValueError("Existing prediction prefix does not match validation split")
    if args.command == "run" and len(predictions) < len(rows):
        import yaml
        from bitext_support.inference.predictor import Predictor
        config = yaml.safe_load((ROOT / args.model_config).read_text(encoding="utf-8"))
        system = rows[0]["messages"][0]["content"]
        if any(r["messages"][0]["content"] != system for r in rows):
            raise ValueError("Mixed system prompts")
        predictor = Predictor(config, kind=args.kind, training_run=args.training_run, system_prompt=system)
        with predictions_path.open("a", encoding="utf-8", newline="\n") as stream:
            for index, row in enumerate(rows[len(predictions):], len(predictions) + 1):
                try:
                    result = predictor.generate(row["text"], max_new_tokens=args.max_new_tokens, do_sample=False)
                    answer, error = result["answer"], None
                except RuntimeError as exc:
                    if str(exc) != "Generation returned an empty answer":
                        raise
                    answer, error, result = "", str(exc), {}
                record = {"sample_id": row["sample_id"], "text": row["text"], "target": row["target"],
                          "answer": answer, "generation_error": error,
                          "output_tokens": result.get("output_tokens"), "hit_token_limit": result.get("hit_token_limit"),
                          "timing_seconds": result.get("timing_seconds"),
                          "peak_allocated_mib": result.get("peak_allocated_mib")}
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                predictions.append(record)
                if index % 10 == 0 or index == len(rows):
                    print(f"Generated {index}/{len(rows)}", flush=True)
    if len(predictions) != len(rows):
        raise ValueError(f"Incomplete predictions: {len(predictions)}/{len(rows)}")
    metrics = write_report(output, predictions, metadata)
    print(json.dumps({k: metrics[k] for k in ("n", "format_valid_rate", "intent_accuracy", "intent_macro_f1", "order_id_exact_accuracy", "order_id_present_f1", "joint_accuracy")}, indent=2))
    print(f"Report: {output / 'report.md'}")


if __name__ == "__main__":
    main()
