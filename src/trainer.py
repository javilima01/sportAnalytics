from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO
from src.models import TrainConfig
from src.config import setup_logger


class YOLOFineTuner:
    """Encapsulates YOLOv8 fine-tuning, validation, and export."""

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.logger = setup_logger(name="YOLOTrainer")
        self.model = YOLO(cfg.model_ckpt)
        self.writer = SummaryWriter(log_dir=cfg.tensorboard_dir)

    def train(self):
        """Fine-tune YOLO model using configuration values."""
        self.logger.info(f"Starting fine-tuning: {self.cfg.model_ckpt}")

        # Dump config and remove non-YOLO keys before passing
        params = self.cfg.model_dump()
        # remove non-arg fields that YOLO.train() doesn’t use
        for key in ["model_ckpt", "export_format", "tensorboard_dir"]:
            params.pop(key, None)

        results = self.model.train(**params)

        if hasattr(results, "results_dict"):
            for key, val in results.results_dict.items():
                self.writer.add_scalar(f"train/{key}", val, self.cfg.epochs)
        self.writer.flush()
        self.logger.info(f"Training completed. Output directory: {self.cfg.save_dir}")
        return results

    def validate(self):
        """Run validation on the validation split."""
        self.logger.info("Running validation...")
        metrics = self.model.val(
            data=self.cfg.data_yaml,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device
        )
        if hasattr(metrics, "results_dict"):
            for key, val in metrics.results_dict.items():
                self.writer.add_scalar(f"val/{key}", val)
        self.writer.flush()
        self.logger.info("Validation completed.")
        return metrics

    def export_best(self):
        """Export best weights to chosen format."""
        if not self.cfg.export_format:
            self.logger.warning("No export format specified, skipping export.")
            return
        weight_path = f"{self.cfg.save_dir}/weights/best.pt"
        self.logger.info(f"Exporting weights from {weight_path} to {self.cfg.export_format}")
        model = YOLO(weight_path)
        model.export(format=self.cfg.export_format)
        self.logger.info("Export completed.")

    def close(self):
        """Release logger and writer resources."""
        self.writer.close()
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
