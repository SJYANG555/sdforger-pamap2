import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pamap2_forger.utils import ensure_dir, write_json


def sample_split(source_dir: Path, split: str, count: int, seed: int) -> dict:
    metadata = pd.read_csv(source_dir / f"{split}_metadata.csv")
    embeddings = pd.read_csv(source_dir / f"{split}_embeddings.csv")
    windows = np.load(source_dir / f"{split}_windows.npy")
    text_records = [json.loads(line) for line in (source_dir / f"{split}_text.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    rng = np.random.default_rng(seed)
    sample_size = min(count, len(metadata))
    indices = np.sort(rng.choice(len(metadata), size=sample_size, replace=False))

    sampled_metadata = metadata.iloc[indices].reset_index(drop=True)
    sampled_embeddings = embeddings.iloc[indices].reset_index(drop=True)
    sampled_windows = windows[indices]
    sampled_text = [text_records[int(i)] for i in indices]

    return {
        "metadata": sampled_metadata,
        "embeddings": sampled_embeddings,
        "windows": sampled_windows,
        "text": sampled_text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a tiny smoke-test dataset from an existing PAMAP2 SDForger dataset.")
    parser.add_argument("--config", required=True, help="Smoke YAML config.")
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    smoke_cfg = raw["smoke"]
    source_dir = Path(smoke_cfg["source_dataset_dir"])
    output_dir = ensure_dir(smoke_cfg["output_dataset_dir"])
    seed = int(smoke_cfg.get("random_seed", 42))

    split_sizes = {
        "train": int(smoke_cfg["train_samples"]),
        "val": int(smoke_cfg["val_samples"]),
        "test": int(smoke_cfg["test_samples"]),
    }

    manifest = {
        "source_dataset_dir": str(source_dir.resolve()),
        "output_dataset_dir": str(output_dir.resolve()),
        "split_sizes": split_sizes,
        "splits": {},
    }

    for offset, (split, size) in enumerate(split_sizes.items()):
        sampled = sample_split(source_dir, split, size, seed + offset)
        sampled["metadata"].to_csv(output_dir / f"{split}_metadata.csv", index=False)
        sampled["embeddings"].to_csv(output_dir / f"{split}_embeddings.csv", index=False)
        np.save(output_dir / f"{split}_windows.npy", sampled["windows"])
        with open(output_dir / f"{split}_text.jsonl", "w", encoding="utf-8") as fp:
            for record in sampled["text"]:
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")

        manifest["splits"][split] = {
            "num_windows": int(len(sampled["metadata"])),
            "activities": sampled["metadata"]["activity_name"].value_counts().sort_index().to_dict(),
            "subjects": sorted(sampled["metadata"]["subject_id"].unique().tolist()),
        }

    for artifact_name in [
        "dataset_manifest.json",
        "split_manifest.json",
        "reconstruction_metadata.json",
        "reducer.pkl",
        "resolved_config.yaml",
    ]:
        src = source_dir / artifact_name
        if src.exists():
            (output_dir / artifact_name).write_bytes(src.read_bytes())

    write_json(output_dir / "smoke_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

