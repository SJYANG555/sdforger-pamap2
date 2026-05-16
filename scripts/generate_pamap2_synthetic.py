import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pamap2_forger.config import load_config, save_resolved_config
from pamap2_forger.generate import generate_synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PAMAP2 synthetic windows from a trained LM.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--model-path", required=True, help="Fine-tuned model path.")
    parser.add_argument("--prompt-metadata", default=None, help="Optional metadata CSV to condition generation.")
    parser.add_argument("--reference-embeddings", default=None, help="Optional embeddings CSV for norm filtering.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save generated embeddings and decoded windows.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_dir = Path(config.data.output_dir)
    prompt_metadata = args.prompt_metadata or str(dataset_dir / f"{config.generation.prompt_split}_metadata.csv")
    reference_embeddings = args.reference_embeddings or str(dataset_dir / "train_embeddings.csv")
    output_dir = args.output_dir or config.generation.output_dir
    save_resolved_config(config, Path(output_dir) / "resolved_config.yaml")
    result = generate_synthetic_dataset(
        model_path=args.model_path,
        reference_embeddings_path=reference_embeddings,
        prompt_metadata_path=prompt_metadata,
        reducer_path=dataset_dir / config.embedding.reducer_artifact_name,
        output_dir=output_dir,
        config=config.generation,
        text_template=config.embedding.text_template,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
