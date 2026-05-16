import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_step(command, name: str) -> None:
    print(f"[smoke] start: {name}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    print(f"[smoke] done: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the minimal GPT-2 smoke test pipeline.")
    parser.add_argument("--config", default="config/pamap2_sdforger_gpt2_smoke.yaml")
    parser.add_argument("--model-path", default=None, help="Optional override for generation checkpoint path.")
    args = parser.parse_args()

    config_path = args.config
    model_path = args.model_path or "outputs/checkpoints/gpt2_smoke/best"

    run_step([sys.executable, "scripts/build_pamap2_sdforger_smoke_dataset.py", "--config", config_path], "build_smoke_dataset")
    run_step([sys.executable, "scripts/train_pamap2_lm.py", "--config", config_path], "train_gpt2_smoke")
    run_step(
        [
            sys.executable,
            "scripts/generate_pamap2_synthetic.py",
            "--config",
            config_path,
            "--model-path",
            model_path,
            "--output-dir",
            "outputs/generated/gpt2_smoke",
        ],
        "generate_smoke",
    )
    run_step(
        [
            sys.executable,
            "scripts/evaluate_similarity.py",
            "--config",
            config_path,
            "--real-windows",
            "artifacts/pamap2_sdforger_dataset_smoke/test_windows.npy",
            "--real-metadata",
            "artifacts/pamap2_sdforger_dataset_smoke/test_metadata.csv",
            "--synthetic-windows",
            "outputs/generated/gpt2_smoke/generated_windows.npy",
            "--synthetic-metadata",
            "outputs/generated/gpt2_smoke/generated_embeddings.csv",
            "--output-dir",
            "outputs/evaluation/similarity/gpt2_smoke",
        ],
        "evaluate_similarity_smoke",
    )
    run_step(
        [
            sys.executable,
            "scripts/evaluate_utility.py",
            "--config",
            config_path,
            "--real-train-windows",
            "artifacts/pamap2_sdforger_dataset_smoke/train_windows.npy",
            "--real-train-metadata",
            "artifacts/pamap2_sdforger_dataset_smoke/train_metadata.csv",
            "--real-test-windows",
            "artifacts/pamap2_sdforger_dataset_smoke/test_windows.npy",
            "--real-test-metadata",
            "artifacts/pamap2_sdforger_dataset_smoke/test_metadata.csv",
            "--synthetic-windows",
            "outputs/generated/gpt2_smoke/generated_windows.npy",
            "--synthetic-metadata",
            "outputs/generated/gpt2_smoke/generated_embeddings.csv",
            "--output-dir",
            "outputs/evaluation/utility/gpt2_smoke",
        ],
        "evaluate_utility_smoke",
    )
    run_step(
        [
            sys.executable,
            "scripts/make_plots.py",
            "--config",
            config_path,
            "--training-log-csv",
            "outputs/checkpoints/gpt2_smoke/training_log_history.csv",
            "--real-windows",
            "artifacts/pamap2_sdforger_dataset_smoke/test_windows.npy",
            "--real-metadata",
            "artifacts/pamap2_sdforger_dataset_smoke/test_metadata.csv",
            "--synthetic-windows",
            "outputs/generated/gpt2_smoke/generated_windows.npy",
            "--synthetic-metadata",
            "outputs/generated/gpt2_smoke/generated_embeddings.csv",
            "--real-embeddings",
            "artifacts/pamap2_sdforger_dataset_smoke/test_embeddings.csv",
            "--synthetic-embeddings",
            "outputs/generated/gpt2_smoke/generated_embeddings.csv",
            "--output-dir",
            "outputs/plots/gpt2_smoke",
        ],
        "make_plots_smoke",
    )


if __name__ == "__main__":
    main()

