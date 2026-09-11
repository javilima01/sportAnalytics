import argparse
import math

from src.config import setup_logger


def parse_segment(value):
    try:
        start, end = map(float, value.split(":"))
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError
        return start, end
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Expected start:end in minutes, with 0 <= start < end."
        ) from error


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Image tagging, label checking, and YOLO training")

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Command: train, generate, visualize, validate, research",
    )

    # --- TRAIN ---
    train_parser = subparsers.add_parser(
        "train", help="Train and validate a YOLO model on tagged images"
    )
    train_parser.add_argument("--data", required=True, type=str, help="Path to dataset YAML")
    train_parser.add_argument(
        "--model", default="yolov8x.pt", type=str, help="Model checkpoint or name"
    )
    train_parser.add_argument("--epochs", default=50, type=int, help="Number of training epochs")
    train_parser.add_argument("--imgsz", default=1280, type=int, help="Training image size")
    train_parser.add_argument("--batch", default=6, type=int, help="Batch size")
    train_parser.add_argument(
        "--project", default="training/finetune", type=str, help="Project folder for outputs"
    )
    train_parser.add_argument(
        "--name", default="yolov8x_football", type=str, help="Experiment name"
    )
    train_parser.add_argument(
        "--device",
        default=None,
        help="Device ('cpu', 'mps', or CUDA index); auto-select by default",
    )
    train_parser.add_argument("--seed", default=0, type=int, help="Training random seed")
    train_parser.add_argument("--patience", default=100, type=int, help="Early stopping patience")
    train_parser.add_argument("--workers", default=8, type=int, help="Number of dataloader workers")

    # --- GENERATE DATASET ---
    gen_parser = subparsers.add_parser(
        "generate", help="Generate a YOLO-format dataset using a trained model"
    )
    gen_parser.add_argument(
        "--model", required=True, type=str, help="Path to trained YOLO model (e.g. best.pt)"
    )
    gen_parser.add_argument("--video", required=True, type=str, help="Video path or YouTube URL")
    gen_parser.add_argument(
        "--output", default="datasets/generated_dataset", type=str, help="Output dataset directory"
    )
    gen_parser.add_argument(
        "--sample_prob", default=0.1, type=float, help="Frame sampling probability [0-1]"
    )
    gen_parser.add_argument(
        "--splits", nargs=3, default=[0.7, 0.2, 0.1], type=float, help="Train/val/test split ratios"
    )
    gen_parser.add_argument(
        "--imgsz", default=1280, type=int, help="Resize frames before YOLO inference"
    )
    gen_parser.add_argument(
        "--conf", default=0.25, type=float, help="Minimum detection confidence [0-1]"
    )
    gen_parser.add_argument(
        "--device",
        default=None,
        help="Device ('cpu', 'mps', or CUDA index); auto-select by default",
    )
    gen_parser.add_argument(
        "--seed", default=0, type=int, help="Frame sampling and split random seed"
    )
    gen_parser.add_argument(
        "--include_empty",
        action="store_true",
        help="Keep frames without detections for manual tagging or background examples",
    )
    gen_parser.add_argument(
        "--segments",
        nargs="+",
        type=parse_segment,
        default=None,
        help="List of video segments in start:end (minutes) format, e.g. 5:10 15:25 60:65",
    )

    # --- VISUALIZE ---
    vis_parser = subparsers.add_parser(
        "visualize", help="Visualize or edit YOLO-format dataset bounding boxes"
    )
    vis_parser.add_argument(
        "--dataset", required=True, type=str, help="Path to YOLO dataset folder"
    )
    vis_parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test", "valid"],
        help="Dataset split to visualize",
    )
    vis_parser.add_argument(
        "--max_images", default=5, type=int, help="Maximum number of random images to show"
    )
    vis_parser.add_argument(
        "--save_dir",
        default=None,
        type=str,
        help="Directory to save visualizations instead of displaying",
    )
    vis_parser.add_argument(
        "--model",
        default=None,
        type=str,
        help="YOLO checkpoint; draw live predictions instead of stored labels",
    )
    vis_parser.add_argument(
        "--conf", default=0.25, type=float, help="Minimum prediction confidence [0-1]"
    )
    vis_parser.add_argument("--imgsz", default=1280, type=int, help="Prediction image size")
    vis_parser.add_argument(
        "--device",
        default=None,
        help="Device ('cpu', 'mps', or CUDA index); auto-select by default",
    )
    vis_parser.add_argument("--edit", action="store_true", help="Open interactive label editor")

    # --- VALIDATE ---
    val_parser = subparsers.add_parser(
        "validate", help="Validate one or more YOLO models without training"
    )
    val_parser.add_argument(
        "--model",
        nargs="*",
        default=[],
        help="One or more checkpoint paths (e.g. best.pt); directories expand to *.pt",
    )
    val_parser.add_argument(
        "--models-dir",
        default=None,
        type=str,
        help="Directory scanned for *.pt checkpoints (non-recursive)",
    )
    val_parser.add_argument("--data", required=True, type=str, help="Path to dataset YAML")
    val_parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to validate on",
    )
    val_parser.add_argument("--imgsz", default=1280, type=int, help="Validation image size")
    val_parser.add_argument("--batch", default=6, type=int, help="Batch size")
    val_parser.add_argument(
        "--device",
        default=None,
        help="Device ('cpu', 'mps', or CUDA index); auto-select by default",
    )
    val_parser.add_argument("--workers", default=8, type=int, help="Number of dataloader workers")

    research = subparsers.add_parser(
        "research", help="Acquire agent-labeled data and run experiments"
    )
    actions = research.add_subparsers(dest="action", required=True)
    for action in ("init", "auto", "acquire", "run", "status", "finalize"):
        command = actions.add_parser(action)
        command.add_argument("--config", default="research.yaml", help="Campaign YAML path")
        if action == "init":
            command.add_argument(
                "--config-only",
                action="store_true",
                help="Create configuration without starting the autonomous workflow",
            )
        if action == "run":
            command.add_argument(
                "--phase", choices=("explore", "promote", "confirm"), default="explore"
            )
    args = parser.parse_args(argv)
    if args.command == "visualize" and args.edit and args.model:
        parser.error("--edit cannot be combined with --model; predictions are not labels.")
    return args


def collect_model_paths(model_entries, models_dir):
    """Expand --model entries and --models-dir into an ordered, de-duplicated file list."""
    from pathlib import Path

    paths = []
    seen = set()

    def add_file(path):
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            seen.add(key)
            paths.append(path)

    for entry in model_entries or []:
        path = Path(entry)
        if path.is_dir():
            for checkpoint in sorted(path.glob("*.pt")):
                if checkpoint.is_file():
                    add_file(checkpoint)
        else:
            add_file(path)
    if models_dir:
        directory = Path(models_dir)
        if not directory.is_dir():
            raise NotADirectoryError(f"Models directory not found: {models_dir}")
        for checkpoint in sorted(directory.glob("*.pt")):
            if checkpoint.is_file():
                add_file(checkpoint)
    if not paths:
        raise ValueError("No models to validate: pass --model and/or --models-dir.")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Model checkpoint(s) not found: {', '.join(missing)}")
    return paths


def metric_row(metrics):
    """Normalize an Ultralytics metrics object into a flat {column: value} mapping."""
    row = {}
    for key, value in dict(getattr(metrics, "results_dict", {})).items():
        column = key.removeprefix("metrics/").removesuffix("(B)")
        try:
            row[column] = float(value)
        except (TypeError, ValueError):
            continue
    speed = getattr(metrics, "speed", None) or {}
    try:
        row["inference_ms"] = float(speed["inference"])
    except (KeyError, TypeError, ValueError):
        pass
    return row


def render_table(headers, rows):
    """Render a table with a left-aligned first column and right-aligned values."""
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def line(cells):
        return "  ".join(
            cell.rjust(widths[index]) if index else cell.ljust(widths[index])
            for index, cell in enumerate(cells)
        )

    divider = "  ".join("-" * width for width in widths)
    return "\n".join([line(headers), divider, *(line(row) for row in rows)])


PREFERRED_COLUMNS = ("precision", "recall", "mAP50", "mAP50-95", "fitness", "inference_ms")


def format_results_table(summary, failures):
    columns = [name for name in PREFERRED_COLUMNS if any(name in row for row in summary.values())]
    columns += sorted({name for row in summary.values() for name in row} - set(columns))
    models = list(summary)
    best = {}
    if len(models) > 1:
        for column in columns:
            values = [summary[model][column] for model in models if column in summary[model]]
            best[column] = min(values) if column.endswith("_ms") else max(values)

    def cell(model, column):
        value = summary[model].get(column)
        if value is None:
            return "-"
        text = f"{value:.1f}" if column.endswith("_ms") else f"{value:.4f}"
        return text + ("*" if best.get(column) == value else "")

    headers = ["model", *columns]
    rows = [[model, *(cell(model, column) for column in columns)] for model in models]
    lines = [render_table(headers, rows)] if summary else []
    if best:
        lines.append("* best in column")
    if failures:
        lines.append("")
        lines.extend(f"FAILED {model}: {error}" for model, error in failures.items())
    return "\n".join(lines) or "No models validated."


def run_validate(args):
    from src.models import TrainConfig
    from src.trainer import YOLOFineTuner

    logger = setup_logger("YOLOValidate")
    model_paths = collect_model_paths(args.model, args.models_dir)
    summary = {}
    failures = {}
    for model_path in model_paths:
        cfg = TrainConfig(
            data=args.data,
            model=str(model_path),
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
        )
        trainer = YOLOFineTuner(cfg)
        try:
            logger.info("Validating %s on split '%s'...", model_path, args.split)
            summary[str(model_path)] = metric_row(trainer.validate(split=args.split))
        except Exception as exc:
            failures[str(model_path)] = str(exc)
            logger.error("Validation failed for %s: %s", model_path, exc)
        finally:
            trainer.close()
    print(format_results_table(summary, failures), flush=True)
    if failures:
        raise SystemExit(1)
    return summary


def run_research(args):
    import json
    from pathlib import Path

    import yaml

    from src.research.config import Campaign, load_campaign
    from src.research.controller import finalize, initialize, run_campaign, status
    from src.research.runtime import campaign_lock

    if args.action == "init":
        path = Path(args.config)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or args.config_only:
            with path.open("x") as handle:
                yaml.safe_dump(Campaign().model_dump(mode="json"), handle, sort_keys=False)
            print(f"Created {path.resolve()}.", flush=True)
        else:
            print(f"Using {path.resolve()}.", flush=True)
        if args.config_only:
            return
    cfg = load_campaign(args.config)
    if args.action == "status":
        print(json.dumps(status(cfg), indent=2))
        return
    with campaign_lock(cfg.output_dir), campaign_lock(cfg.dataset_dir):
        initialize(cfg)
        if args.action in ("init", "auto"):
            from src.research.autonomous import run_autonomous

            result = run_autonomous(cfg)
        elif args.action == "acquire":
            from src.research.acquisition import acquire

            result = acquire(cfg)
        elif args.action == "finalize":
            result = finalize(cfg)
        else:
            result = run_campaign(cfg, phase=args.phase)
        print(json.dumps(result, indent=2))


def run_train(args):
    from src.models import TrainConfig
    from src.trainer import YOLOFineTuner

    cfg = TrainConfig(
        data=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=args.project,
        name=args.name,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
    )

    trainer = YOLOFineTuner(cfg)
    trainer.logger.info("Configuration:")
    for k, v in cfg.model_dump().items():
        trainer.logger.info(f"  {k}: {v}")

    try:
        trainer.logger.info("Starting training process...")
        trainer.train()
        trainer.logger.info("Running validation after training...")
        metrics = trainer.validate()
        trainer.logger.info(f"Final validation metrics: {metrics}")
    finally:
        trainer.close()


def run_generate(args):
    from src.creation import DatasetCreator

    logger = setup_logger("DatasetCreator")

    creator = DatasetCreator(
        model_path=args.model,
        output_dir=args.output,
        sample_prob=args.sample_prob,
        splits=tuple(args.splits),
        imgsz=args.imgsz,
        logger=logger,
        conf=args.conf,
        device=args.device,
        seed=args.seed,
        include_empty=args.include_empty,
    )

    creator.create_from_video(args.video, segments=args.segments)


def run_visualize(args):
    from src.visualizer import Visualizer

    logger = setup_logger("Visualizer")
    vis = Visualizer(
        dataset_dir=args.dataset,
        split=args.split,
        max_images=args.max_images,
        logger=logger,
        model_path=args.model,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        window_name="YOLO Predictions" if args.model else "YOLO Label Editor",
    )

    if args.edit:
        logger.info("Starting interactive label editor...")
        vis.edit()
    else:
        vis.visualize(save_dir=args.save_dir)


def main():
    args = parse_args()
    if args.command == "train":
        run_train(args)
    elif args.command == "generate":
        run_generate(args)
    elif args.command == "visualize":
        run_visualize(args)
    elif args.command == "validate":
        run_validate(args)
    elif args.command == "research":
        run_research(args)
    else:
        raise ValueError("Invalid command. Use 'train', 'generate', 'visualize', or 'validate'.")


if __name__ == "__main__":
    main()
