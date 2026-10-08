from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "training" / "prepared"
DEFAULT_OUTPUT = PROJECT_ROOT / "training" / "runs"

# These were the three additional experiments in the completed expanded run.
# The other five completed model checkpoints were verified and reused.
EXPERIMENTS_TO_TRAIN = (
    "mixed_ai50_real50_300_existing_labels",
    "mixed_real75_ai25_300_existing_labels",
    "mixed_ai75_real25_300_existing_labels",
)

EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 8
SEED = 42
OPTIMIZER = "AdamW"
INITIAL_LEARNING_RATE = 0.001
PRETRAINED_WEIGHTS = "yolov8n.pt"


def train_experiment(data_root: Path, output_root: Path, experiment_name: str) -> None:
    """Fit one mixture, choose its best validation checkpoint, then test it."""
    from ultralytics import YOLO

    experiment_dir = data_root / experiment_name
    data_yaml = experiment_dir / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"Could not find {data_yaml}. Prepare the dataset before training."
        )

    run_dir = output_root / experiment_name
    if run_dir.exists():
        raise FileExistsError(
            f"{run_dir} already exists. Choose a new output folder rather than "
            "overwriting a previous training run."
        )

    print(f"\nTraining {experiment_name}")
    model = YOLO(PRETRAINED_WEIGHTS)
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        optimizer=OPTIMIZER,
        lr0=INITIAL_LEARNING_RATE,
        seed=SEED,
        device=0,
        deterministic=True,
        amp=True,
        cache=False,
        workers=2,
        patience=0,
        save=True,
        save_period=10,
        plots=True,
        project=str(output_root),
        name=experiment_name,
    )

    best_checkpoint = run_dir / "weights" / "best.pt"
    if not best_checkpoint.is_file():
        raise FileNotFoundError(f"Training did not produce the expected checkpoint: {best_checkpoint}")

    print(f"Selected checkpoint: {best_checkpoint}")
    print("Evaluating the selected checkpoint on this experiment's held-out test split.")
    best_model = YOLO(str(best_checkpoint))
    test_metrics = best_model.val(
        data=str(data_yaml),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=0,
        plots=True,
    )
    print("Test metrics:", test_metrics.results_dict)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=EXPERIMENTS_TO_TRAIN,
        default=list(EXPERIMENTS_TO_TRAIN),
        help="Choose which of the three additional real/AI mixtures to train.",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    for experiment_name in args.experiments:
        train_experiment(args.data_root, args.output, experiment_name)


if __name__ == "__main__":
    main()
