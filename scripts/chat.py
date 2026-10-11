"""Single-turn inference using Base or a completed SFT/DPO export."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--kind", choices=("base", "sft", "dpo"), default="base")
    parser.add_argument("--training-run", type=Path)
    parser.add_argument("--system-prompt", type=Path, default=Path("prompts/system_general.txt"))
    parser.add_argument("--config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument("--model", help="Base mode only; Adapter mode uses the training report")
    parser.add_argument("--revision", help="Base mode only")
    parser.add_argument("--device", type=int)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--max-input-tokens", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ultrafeedback/inference"))
    args = parser.parse_args()
    if not args.prompt.strip():
        parser.error("--prompt must contain non-whitespace text")
    if args.kind != "base" and (args.model or args.revision):
        parser.error("Adapter mode uses the model/revision from its training report")
    from bitext_support.inference.predictor import Predictor, resolve_model, validate_generation
    import yaml

    try:
        config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
        if args.model:
            config["model"]["name_or_path"] = args.model
        if args.revision:
            config["model"]["revision"] = args.revision
        if args.device is not None:
            config["model"]["device"] = args.device
        if args.allow_download:
            config["model"]["local_files_only"] = False
        for key in ("max_input_tokens", "max_new_tokens", "do_sample", "temperature", "top_p", "repetition_penalty"):
            value = getattr(args, key)
            if value is not None:
                config["generation"][key] = value
        validate_generation(config["generation"])
        resolve_model(config, args.kind, args.training_run)
        system = (ROOT / args.system_prompt).read_text(encoding="utf-8")
        predictor = Predictor(config, args.kind, args.training_run, system)
        report = predictor.generate(args.prompt)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    report["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    directory = ROOT / args.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{args.kind}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(report["answer"])
    print(f"\nModel: {args.kind} | GPU: {report['gpu']} | dtype: {report['compute_dtype']}")
    print(f"Load: {report['timing_seconds']['load']}s | generation: {report['timing_seconds']['generate']}s"
          f" | peak allocated: {report['peak_allocated_mib']} MiB | report: {path}")


if __name__ == "__main__":
    main()
