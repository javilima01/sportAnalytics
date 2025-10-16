from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO
import torch
import torch.nn.utils.prune as prune
from .models import TrainConfig
from .config import setup_logger


class YOLOFineTuner:
    """Encapsulates YOLOv8 fine-tuning, validation, pruning, quantization, and export."""

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.logger = setup_logger(name="YOLOTrainer")
        self.model = YOLO(cfg.model)
        self.writer = SummaryWriter(log_dir=cfg.tensorboard_dir)

    def train(self):
        """Fine-tune YOLO model using configuration values."""
        self.logger.info(f"Starting fine-tuning: {self.cfg.model}")

        params = self.cfg.model_dump()
        for key in ["model", "export_format", "tensorboard_dir"]:
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
            data=self.cfg.data,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device
        )
        if hasattr(metrics, "results_dict"):
            for key, val in metrics.results_dict.items():
                self.writer.add_scalar(f"val/{key}", val)
        self.writer.flush()
        self.logger.info("Validation completed.")
        return metrics

    def prune(self, amount: float = 0.3):
        """Prunes model parameters by a given amount."""
        self.logger.info(f"Pruning model with amount={amount}")
        torch_model = self.model.model
        for name, module in torch_model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name="weight", amount=amount)
                prune.remove(module, "weight")
        self.logger.info("Model pruning completed.")

    def export_quantized(self, output_dir: str = None):
        """Exports the model in a quantized form."""
        self.logger.info("Exporting quantized model...")
        if output_dir is None:
            output_dir = f"{self.cfg.save_dir}/weights/quantized.pt"

        quantized_model = torch.quantization.quantize_dynamic(
            self.model.model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
        )
        torch.save(quantized_model.state_dict(), output_dir)
        self.logger.info(f"Quantized model saved at: {output_dir}")
        return output_dir

    def export_onnx(self, output_path: str = None):
        """Exports the model to ONNX format."""
        self.logger.info("Exporting model to ONNX format...")
        if output_path is None:
            output_path = f"{self.cfg.save_dir}/weights/model.onnx"
        self.model.export(format="onnx", opset=12, dynamic=True, simplify=True)
        self.logger.info(f"ONNX model exported to: {output_path}")
        return output_path

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
