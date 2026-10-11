"""Start the local UltraFeedback single-turn inference API."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("base", "sft", "dpo"), default="base")
    parser.add_argument("--training-run", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument("--system-prompt", type=Path, default=Path("prompts/system_general.txt"))
    parser.add_argument("--customer-support", action="store_true", help="Enable the structured /analyze endpoint")
    parser.add_argument("--device", type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    import yaml
    import uvicorn
    from bitext_support.inference.api import create_app

    try:
        config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
        if args.device is not None:
            config["model"]["device"] = args.device
        system = (ROOT / args.system_prompt).read_text(encoding="utf-8")
        app = create_app(config, args.kind, args.training_run, system,
                         customer_support=args.customer_support)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
