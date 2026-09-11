"""Fine-tune and validate detection checkpoints on reviewed images."""

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO

from .config import setup_logger
from .models import TrainConfig
from .precision import enable_mixed_precision, resolve_device


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
        arguments = self.cfg.training_args()
        if self.cfg.precision != "fp32":
            device = resolve_device(self.cfg.device)
            arguments["device"] = str(device)
            self.logger.info("Training precision %s on %s", self.cfg.precision, device)
            enable_mixed_precision(self.model, self.cfg.precision, device)
        results = self.model.train(**arguments)
        directory = Path(self.model.trainer.save_dir)
        if self.writer is not None:
            self.writer.close()
        self.writer = SummaryWriter(
            log_dir=self.cfg.tensorboard_dir or str(directory / "tensorboard")
        )
        self._log_metrics("train", results)
        self.logger.info("Training completed: %s", directory)
        return results

    def validate(self, split="val", half=None):
        half = self.cfg.half if half is None else half
        device = self.cfg.device
        if half:
            resolved = resolve_device(device)
            if resolved.type == "cpu":
                raise ValueError("Half-precision validation requires an MPS or CUDA device.")
            device = str(resolved)
        metrics = self.model.val(
            data=self.cfg.data,
            imgsz=self.cfg.imgsz,
            device=device,
            batch=self.cfg.batch,
            workers=self.cfg.workers,
            split=split,
            half=half,
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
