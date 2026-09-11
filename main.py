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
        dest="command", required=True, help="Command: train, generate, or visualize"
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
    vis_parser.add_argument("--edit", action="store_true", help="Open interactive label editor")

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
    return parser.parse_args(argv)


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
        dataset_dir=args.dataset, split=args.split, max_images=args.max_images, logger=logger
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
    elif args.command == "research":
        run_research(args)
    else:
        raise ValueError("Invalid command. Use 'train', 'generate', or 'visualize'.")


if __name__ == "__main__":
    main()
