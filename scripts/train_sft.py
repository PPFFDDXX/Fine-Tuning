"""Run smoke/full SFT training or verify an exported Adapter."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bitext_support.training.sft import project_path, train_smoke, verify_reload  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("smoke", help="Train on a small, fixed subset")
    train.add_argument("--config", default="configs/sft_ultrafeedback_v2_smoke.yaml")
    full = sub.add_parser("full", help="Train on all eligible train rows and select by validation loss")
    full.add_argument("--config", default="configs/sft_ultrafeedback_v2.yaml")
    full.add_argument("--resume", help="Path to a completed step-* checkpoint directory")
    verify = sub.add_parser("verify", help="Reload the Adapter in a fresh process")
    verify.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    if args.command in ("smoke", "full"):
        import yaml
        config = yaml.safe_load(project_path(args.config).read_text(encoding="utf-8"))
        if args.command == "full":
            from bitext_support.training.sft_full import train_full
            run_dir = train_full(config, project_path(args.resume) if args.resume else None)
        else:
            run_dir = train_smoke(config)
        print(f"Saved: {run_dir}")
        print(f'Next: python scripts/train_sft.py verify --run-dir "{run_dir}"')
    else:
        report = verify_reload(project_path(args.run_dir))
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
