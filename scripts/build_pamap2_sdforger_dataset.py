import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pamap2_forger.config import load_config, save_resolved_config
from pamap2_forger.dataset import build_sdforger_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a PAMAP2 SDForger-style dataset.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    manifest = build_sdforger_dataset(config.data, config.embedding)
    save_resolved_config(config, Path(config.data.output_dir) / "resolved_config.yaml")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
