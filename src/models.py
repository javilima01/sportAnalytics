from pydantic import BaseModel, Field
from typing import Optional


class TrainConfig(BaseModel):
    """Configuration schema for YOLOv8 fine-tuning using Pydantic."""

    data: str = Field("data.yaml", description="Path to the dataset YAML file that defines image paths and class names.")
    model_ckpt: str = Field("models/yolov8x.pt", description="Pretrained YOLO checkpoint to start fine-tuning from.")
    epochs: int = Field(50, ge=1, description="Number of full training epochs.")
    imgsz: int = Field(640, ge=128, description="Image size used for both training and validation.")
    batch: int = Field(16, ge=1, description="Number of images per batch.")
    lr0: float = Field(0.001, gt=0, description="Initial learning rate for the optimizer.")
    weight_decay: float = Field(0.0005, ge=0, description="Weight decay (L2 regularization) factor.")
    patience: int = Field(20, ge=0, description="Early stopping patience in epochs with no improvement.")
    freeze: int = Field(0, ge=0, description="Number of model backbone layers to freeze during fine-tuning.")
    device: str = Field("0", description="Device for training, e.g., '0' for first GPU or 'cpu' for CPU mode.")
    project: str = Field("training/finetune", description="Project folder where YOLO training outputs are saved.")
    name: str = Field("yolov8x_football", description="Experiment subfolder name under the project directory.")
    cos_lr: bool = Field(True, description="Use cosine learning rate schedule if True.")
    amp: bool = Field(True, description="Use automatic mixed precision (float16) if supported.")
    pretrained: bool = Field(True, description="Load pretrained weights before fine-tuning.")
    optimizer: str = Field("auto", description="Optimizer type (e.g., 'SGD', 'Adam', 'auto').")
    workers: int = Field(4, ge=0, description="Number of dataloader worker threads.")
    export_format: Optional[str] = Field("pt", description="Export format for best weights after training (e.g., 'pt', 'onnx', 'engine').")
    tensorboard_dir: str = Field("training/tensorboard", description="Directory to store TensorBoard logs for visualization.")

    @property
    def save_dir(self) -> str:
        """Output directory for current training run."""
        return f"{self.project}/{self.name}"

    class Config:
        extra = "forbid"
        validate_assignment = True
