import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pamap2_forger.config import load_config, save_resolved_config
from pamap2_forger.trainer import train_language_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a GPT-2/Gemma language model on PAMAP2 SDForger text.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--train-jsonl", default=None, help="Optional override for training JSONL.")
    parser.add_argument("--val-jsonl", default=None, help="Optional override for validation JSONL.")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_dir = config.data.output_dir
    save_resolved_config(config, Path(config.training.output_dir) / "resolved_config.yaml")
    best_path = train_language_model(
        train_jsonl=args.train_jsonl or f"{dataset_dir}/train_text.jsonl",
        val_jsonl=args.val_jsonl or f"{dataset_dir}/val_text.jsonl",
        config=config.training,
    )
    print(best_path)


if __name__ == "__main__":
    main()
