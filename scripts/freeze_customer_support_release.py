"""Freeze and verify the customer-support model, datasets, and evaluations."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bitext_support.evaluation.customer_support import score

ARTIFACT = ROOT / "reports/customer_support_release/freeze.json"
DATA = ROOT / "data/processed/customer_support/v2"
EXTERNAL = ROOT / "data/external/customer_support_twitter"
EVALS = {
    "base_validation": ROOT / "outputs/customer_support/base_validation_pinned",
    "sft_validation": ROOT / "outputs/customer_support/sft_validation",
    "base_test": ROOT / "outputs/customer_support/base_test",
    "sft_test": ROOT / "outputs/customer_support/sft_test",
    "base_external": ROOT / "outputs/customer_support/base_external",
    "sft_external": ROOT / "outputs/customer_support/sft_external",
}
METRICS = ("format_valid_rate", "intent_accuracy", "intent_macro_f1",
           "order_id_exact_accuracy", "order_id_present_f1", "joint_accuracy")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path):
    path = path.resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError(f"Path outside project: {path}")
    return path.relative_to(ROOT).as_posix()


def pinned_file(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": relative(path), "sha256": sha(path), "bytes": path.stat().st_size}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect():
    source_manifest = read_json(EXTERNAL / "source_manifest.json")
    if sha(EXTERNAL / "twcs.csv") != source_manifest["sha256"]:
        raise ValueError("Raw external CSV differs from its source manifest")
    files = {
        "inference_config": ROOT / "configs/inference.yaml",
        "training_config": ROOT / "configs/sft_customer_support_v2.yaml",
        "system_prompt": ROOT / "prompts/system_customer_support.txt",
        "data_manifest": DATA / "manifest.json",
        "data_audit": DATA / "audit.json",
        "train_data": DATA / "sft/train.jsonl",
        "validation_data": DATA / "sft/validation.jsonl",
        "test_data": DATA / "sft/test.jsonl",
        "external_source_manifest": EXTERNAL / "source_manifest.json",
        "external_source_csv": EXTERNAL / "twcs.csv",
        "external_selection_labels": ROOT / "configs/external_twitter_50_labels.json",
        "external_selected_data": EXTERNAL / "selected_50.jsonl",
    }
    pinned = {key: pinned_file(path) for key, path in files.items()}
    model_config = __import__("yaml").safe_load(files["inference_config"].read_text(encoding="utf-8"))
    if not model_config["model"].get("revision") or model_config["model"].get("local_files_only") is not True:
        raise ValueError("Base model must use a pinned revision in offline mode")
    reports = {}
    for name, directory in EVALS.items():
        report_path, predictions_path = directory / "report.json", directory / "predictions.jsonl"
        report = read_json(report_path)
        predictions = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()]
        if sha(predictions_path) != report["predictions_sha256"] or len(predictions) != report["n"]:
            raise ValueError(f"Prediction integrity failed: {name}")
        calculated, _ = score(predictions)
        if any(abs(calculated[key] - report[key]) > 1e-12 for key in METRICS):
            raise ValueError(f"Reported metrics differ from predictions: {name}")
        provenance = report["provenance"]
        expected_kind, expected_split = name.split("_", 1)
        if provenance["model_kind"] != expected_kind or provenance["split"] != expected_split:
            raise ValueError(f"Wrong evaluation kind or split: {name}")
        data_key = "external_selected_data" if expected_split == "external" else f"{expected_split}_data"
        if provenance["data_sha256"] != pinned[data_key]["sha256"]:
            raise ValueError(f"Evaluation data changed: {name}")
        if provenance["model_config_sha256"] != pinned["inference_config"]["sha256"]:
            raise ValueError(f"Inference config changed: {name}")
        reports[name] = {"report": pinned_file(report_path),
                         "predictions": pinned_file(predictions_path),
                         "scored_predictions": pinned_file(directory / "scored_predictions.jsonl"),
                         "n": report["n"], "metrics": {key: report[key] for key in METRICS},
                         "provenance": provenance}
        if reports[name]["scored_predictions"]["sha256"] != report["scored_predictions_sha256"]:
            raise ValueError(f"Scored predictions changed: {name}")
    for split in ("validation", "test", "external"):
        base, sft = (reports[f"{kind}_{split}"]["provenance"] for kind in ("base", "sft"))
        for key in ("data_sha256", "model_config_sha256", "requested_max_new_tokens", "limit"):
            if base[key] != sft[key]:
                raise ValueError(f"Base/SFT evaluation setting mismatch: {split}, {key}")
        extra = (("source_sha256", "selection_sha256", "system_prompt_sha256") if split == "external"
                 else ("manifest_sha256", "review_sha256"))
        for key in extra:
            if base[key] != sft[key]:
                raise ValueError(f"Base/SFT provenance mismatch: {split}, {key}")
        if split == "external":
            if (base["source_sha256"] != pinned["external_source_csv"]["sha256"] or
                    base["selection_sha256"] != pinned["external_selection_labels"]["sha256"] or
                    base["system_prompt_sha256"] != pinned["system_prompt"]["sha256"]):
                raise ValueError("External source, selection, or prompt changed")
        else:
            if (base["manifest_sha256"] != pinned["data_manifest"]["sha256"] or
                    base["review_sha256"] != sha(DATA / "review_queue.jsonl")):
                raise ValueError(f"Internal audit or manifest changed: {split}")
    sft_provenance = [reports[f"sft_{split}"]["provenance"] for split in ("validation", "test", "external")]
    if len({p["training_run"] for p in sft_provenance}) != 1:
        raise ValueError("SFT evaluations refer to different training runs")
    run = ROOT / sft_provenance[0]["training_run"]
    train_report_path, reload_path = run / "train_report.json", run / "reload_report.json"
    train_report, reload_report = read_json(train_report_path), read_json(reload_path)
    if train_report["status"] != "trained_pending_reload" or reload_report["status"] != "passed":
        raise ValueError("SFT training or independent reload check did not pass")
    if train_report["data_sha256"] != {"train": pinned["train_data"]["sha256"],
                                       "validation": pinned["validation_data"]["sha256"]}:
        raise ValueError("SFT training dataset changed")
    if train_report["model"]["revision"] != model_config["model"]["revision"]:
        raise ValueError("SFT and Base revisions differ")
    adapter = pinned_file(run / "adapter/adapter_model.safetensors")
    adapter_config = pinned_file(run / "adapter/adapter_config.json")
    for provenance in sft_provenance:
        if (provenance["train_report_sha256"] != sha(train_report_path) or
                provenance["adapter_sha256"] != train_report["adapter_sha256"]):
            raise ValueError("Evaluation uses a different SFT adapter")
    return {"format_version": 1, "base_model": {"name_or_path": model_config["model"]["name_or_path"],
                                                  "revision": model_config["model"]["revision"]},
            "sft": {"training_run": relative(run), "train_report": pinned_file(train_report_path),
                    "reload_report": pinned_file(reload_path), "adapter": adapter,
                    "adapter_config": adapter_config, "adapter_tensor_sha256": train_report["adapter_sha256"]},
            "files": pinned, "evaluations": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Check all frozen files without modifying the manifest")
    args = parser.parse_args()
    current = collect()
    if args.verify:
        if read_json(ARTIFACT) != current:
            raise ValueError("Current project artifacts differ from frozen manifest")
        print(f"Freeze verified: {relative(ARTIFACT)}")
    else:
        if ARTIFACT.exists():
            raise FileExistsError(f"Freeze already exists; use --verify: {ARTIFACT}")
        ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT.write_text(json.dumps(current, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Freeze created: {relative(ARTIFACT)}")


if __name__ == "__main__":
    main()
