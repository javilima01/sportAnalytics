"""Fine-tune and validate detection checkpoints on reviewed images."""

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO

from .config import setup_logger
from .models import TrainConfig


class YOLOFineTuner:
    def __init__(self, cfg: TrainConfig):
        if not Path(cfg.data).is_file():
            raise FileNotFoundError(f"Dataset YAML not found: {cfg.data}")
        self.cfg = cfg
        self.logger = setup_logger("YOLOTrainer")
        self.model = YOLO(cfg.model)
        if self.model.task != "detect":
            raise ValueError("Training requires a detection model.")
        self.writer = None

    def train(self):
        results = self.model.train(**self.cfg.training_args())
        directory = Path(self.model.trainer.save_dir)
        if self.writer is not None:
            self.writer.close()
        self.writer = SummaryWriter(
            log_dir=self.cfg.tensorboard_dir or str(directory / "tensorboard")
        )
        self._log_metrics("train", results)
        self.logger.info("Training completed: %s", directory)
        return results

    def validate(self, split="val"):
        metrics = self.model.val(
            data=self.cfg.data,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device,
            batch=self.cfg.batch,
            workers=self.cfg.workers,
            split=split,
        )
        self._log_metrics(split, metrics)
        return metrics

    def _log_metrics(self, stage, results):
        if self.writer is not None:
            for key, value in getattr(results, "results_dict", {}).items():
                self.writer.add_scalar(f"{stage}/{key}", value)
            self.writer.flush()

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
