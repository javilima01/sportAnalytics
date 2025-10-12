import argparse
from src.models import TrainConfig
from src.trainer import YOLOFineTuner
from src.creation import DatasetCreator
from src.visualizer import Visualizer
from src.config import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Training, Dataset Generation, and Visualization Pipeline")

    subparsers = parser.add_subparsers(dest="command", required=True, help="Command: train, generate, or visualize")

    # --- TRAIN ---
    train_parser = subparsers.add_parser("train", help="Fine-tune a YOLOv8 model")
    train_parser.add_argument("--data", required=True, type=str, help="Path to dataset YAML")
    train_parser.add_argument("--model", default="yolov8x.pt", type=str, help="Model checkpoint or name")
    train_parser.add_argument("--epochs", default=50, type=int, help="Number of training epochs")
    train_parser.add_argument("--imgsz", default=1280, type=int, help="Training image size")
    train_parser.add_argument("--batch", default=6, type=int, help="Batch size")
    train_parser.add_argument("--project", default="training/finetune", type=str, help="Project folder for outputs")
    train_parser.add_argument("--name", default="yolov8x_football", type=str, help="Experiment name")
    train_parser.add_argument("--device", default="0", type=str, help="Device index (e.g. '0' or 'cpu')")
    train_parser.add_argument("--patience", default=100, type=int, help="Early stopping patience")
    train_parser.add_argument("--workers", default=8, type=int, help="Number of dataloader workers")

    # --- GENERATE DATASET ---
    gen_parser = subparsers.add_parser("generate", help="Generate a YOLO-format dataset using a trained model")
    gen_parser.add_argument("--model", required=True, type=str, help="Path to trained YOLO model (e.g. best.pt)")
    gen_parser.add_argument("--video", required=True, type=str, help="Video path or YouTube URL")
    gen_parser.add_argument("--output", default="datasets/generated_dataset", type=str, help="Output dataset directory")
    gen_parser.add_argument("--sample_prob", default=0.1, type=float, help="Frame sampling probability [0-1]")
    gen_parser.add_argument("--splits", nargs=3, default=[0.7, 0.2, 0.1], type=float, help="Train/val/test split ratios")
    gen_parser.add_argument("--imgsz", default=1280, type=int, help="Resize frames before YOLO inference")
    gen_parser.add_argument(
        "--segments",
        nargs="+",
        default=None,
        help="List of video segments in start:end (minutes) format, e.g. 5:10 15:25 60:65"
    )

    # --- VISUALIZE / EDIT DATASET ---
    vis_parser = subparsers.add_parser("visualize", help="Visualize or edit YOLO-format dataset bounding boxes")
    vis_parser.add_argument("--dataset", required=True, type=str, help="Path to YOLO dataset folder")
    vis_parser.add_argument("--split", default="train", choices=["train", "val", "test", "valid"], help="Dataset split to visualize")
    vis_parser.add_argument("--max_images", default=5, type=int, help="Maximum number of random images to show (view mode only)")
    vis_parser.add_argument("--save_dir", default=None, type=str, help="Directory to save visualizations instead of displaying")
    vis_parser.add_argument("--edit", action="store_true", help="Open interactive label editor")

    return parser.parse_args()


def run_train(args):
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
        workers=args.workers
    )

    trainer = YOLOFineTuner(cfg)
    trainer.logger.info("Configuration:")
    for k, v in cfg.model_dump().items():
        trainer.logger.info(f"  {k}: {v}")

    trainer.train()
    trainer.validate()
    trainer.export_best()
    trainer.close()


def run_generate(args):
    logger = setup_logger("DatasetCreator")

    segments = None
    if args.segments:
        segments = []
        for seg in args.segments:
            try:
                start, end = map(float, seg.split(":"))
                if end <= start:
                    raise ValueError
                segments.append((start, end))
            except Exception:
                raise ValueError(f"Invalid segment format '{seg}'. Expected start:end in minutes (e.g. 5:10).")

    creator = DatasetCreator(
        model_path=args.model,
        output_dir=args.output,
        sample_prob=args.sample_prob,
        splits=tuple(args.splits),
        imgsz=args.imgsz,
        logger=logger
    )

    creator.create_from_video(args.video, segments=segments)


def run_visualize(args):
    logger = setup_logger("Visualizer")
    vis = Visualizer(
        dataset_dir=args.dataset,
        split=args.split,
        max_images=args.max_images,
        logger=logger
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
    else:
        raise ValueError("Invalid command. Use 'train', 'generate', or 'visualize'.")


if __name__ == "__main__":
    main()
