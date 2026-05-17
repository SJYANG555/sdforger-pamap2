import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mmfit_forger.dataset import (  # noqa: E402
    DEFAULT_SELECTED_ACTIVITIES,
    DEFAULT_SENSOR_STREAMS,
    MMFitBuildConfig,
    MMFitEmbeddingConfig,
    build_mmfit_sdforger_dataset,
)
from pamap2_forger.config import load_config, save_resolved_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an MM-Fit SDForger-style exercise dataset.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp)
    pipeline_config = load_config(args.config)
    mmfit_raw = raw.get("mmfit", {})

    build_config = MMFitBuildConfig(
        source_path=pipeline_config.data.preprocessed_dir,
        output_dir=pipeline_config.data.output_dir,
        selected_activities=mmfit_raw.get("selected_activities", DEFAULT_SELECTED_ACTIVITIES),
        sensor_streams=mmfit_raw.get("sensor_streams", DEFAULT_SENSOR_STREAMS),
        train_sessions=pipeline_config.data.train_subjects or [1, 2, 3, 4, 6, 7, 8, 16, 17, 18],
        val_sessions=pipeline_config.data.val_subjects or [14, 15, 19],
        test_sessions=pipeline_config.data.test_subjects or [0, 5, 12, 13, 20],
        window_seconds=float(mmfit_raw.get("window_seconds", 5.0)),
        stride_seconds=float(mmfit_raw.get("stride_seconds", 2.5)),
        video_fps=float(mmfit_raw.get("video_fps", 30.0)),
        output_sample_rate_hz=float(mmfit_raw.get("output_sample_rate_hz", 50.0)),
        random_seed=pipeline_config.data.random_seed,
    )
    embedding_config = MMFitEmbeddingConfig(
        method=pipeline_config.embedding.method,
        n_components=pipeline_config.embedding.n_components,
        variance_explained=pipeline_config.embedding.variance_explained,
        standardization=pipeline_config.embedding.standardization,
        fastica_max_iter=pipeline_config.embedding.fastica_max_iter,
        fastica_tol=pipeline_config.embedding.fastica_tol,
        input_tokens_precision=pipeline_config.embedding.input_tokens_precision,
        reducer_artifact_name=pipeline_config.embedding.reducer_artifact_name,
        scaler_artifact_name=pipeline_config.embedding.scaler_artifact_name,
    )

    manifest = build_mmfit_sdforger_dataset(build_config, embedding_config)
    save_resolved_config(pipeline_config, Path(pipeline_config.data.output_dir) / "resolved_config.yaml")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

