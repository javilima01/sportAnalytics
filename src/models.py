from pydantic import BaseModel, Field
from typing import Optional


class TrainConfig(BaseModel):
    """
    Full YOLOv8 training configuration aligned with Ultralytics' trainer parameters.
    Use `cfg.model_dump()` to pass directly into `YOLO().train(**cfg.model_dump())`.
    """

    # --- core training ---
    task: str = Field("detect", description="YOLOv8 task type: detect, segment, classify, or pose.")
    mode: str = Field("train", description="Mode for YOLO engine (train, val, predict, etc.).")
    model: str = Field("models/yolov8x.pt", description="Path or name of pretrained YOLO checkpoint.")
    data: str = Field("datasets/roboflow/data.yaml", description="Path to dataset YAML (train/val/test splits).")
    epochs: int = Field(50, ge=1, description="Number of full training epochs.")
    imgsz: int = Field(1280, ge=128, description="Image size for training and validation.")
    batch: int = Field(3, ge=1, description="Batch size for training.")
    device: Optional[str] = Field("0", description="Device index or 'cpu' for training.")
    workers: int = Field(8, ge=0, description="Number of DataLoader workers.")
    project: Optional[str] = Field("training/finetune", description="Root folder for saving results.")
    name: str = Field("yolov8x_football", description="Experiment name within the project directory.")
    pretrained: bool = Field(True, description="Load pretrained weights before training.")
    optimizer: str = Field("auto", description="Optimizer type: auto, SGD, Adam, etc.")
    deterministic: bool = Field(True, description="Ensure deterministic training for reproducibility.")
    seed: int = Field(0, ge=0, description="Random seed.")
    patience: int = Field(100, ge=0, description="Early stopping patience (epochs with no improvement).")

    # --- learning rate & schedule ---
    lr0: float = Field(0.01, gt=0, description="Initial learning rate.")
    lrf: float = Field(0.01, gt=0, description="Final learning rate (lr0 * lrf).")
    momentum: float = Field(0.937, ge=0, description="Optimizer momentum factor.")
    weight_decay: float = Field(0.0005, ge=0, description="L2 regularization weight decay.")
    warmup_epochs: float = Field(3.0, ge=0, description="Number of warmup epochs.")
    warmup_momentum: float = Field(0.8, ge=0, description="Initial momentum during warmup.")
    warmup_bias_lr: float = Field(0.1, ge=0, description="Initial bias LR during warmup.")
    cos_lr: bool = Field(False, description="Use cosine LR schedule instead of linear decay.")
    amp: bool = Field(True, description="Use automatic mixed precision if supported.")

    # --- model behavior ---
    single_cls: bool = Field(False, description="Treat dataset as a single class.")
    rect: bool = Field(False, description="Use rectangular training batches.")
    cache: bool = Field(False, description="Cache images for faster training (True/False/ram/disk).")
    resume: bool = Field(False, description="Resume training from last checkpoint.")
    freeze: Optional[int] = Field(None, description="Number of layers to freeze (backbone).")
    multi_scale: bool = Field(False, description="Use multi-scale training.")
    overlap_mask: bool = Field(True, description="Apply masks overlapping in segmentation tasks.")
    dropout: float = Field(0.0, ge=0, description="Dropout probability in the head.")
    save_period: int = Field(-1, description="Checkpoint saving period (-1 disables periodic saving).")

    # --- validation ---
    val: bool = Field(True, description="Run validation during training.")
    split: str = Field("val", description="Dataset split for validation.")
    plots: bool = Field(True, description="Save training plots and images.")
    save_json: bool = Field(False, description="Save COCO-style JSON results on validation.")
    conf: Optional[float] = Field(None, description="Confidence threshold for validation inference.")
    iou: float = Field(0.7, ge=0, le=1, description="IoU threshold for NMS.")
    max_det: int = Field(300, ge=1, description="Maximum detections per image during validation.")

    # --- augmentation ---
    hsv_h: float = Field(0.015, description="HSV-Hue augmentation gain.")
    hsv_s: float = Field(0.7, description="HSV-Saturation augmentation gain.")
    hsv_v: float = Field(0.4, description="HSV-Value augmentation gain.")
    translate: float = Field(0.1, description="Image translation augmentation factor.")
    scale: float = Field(0.5, description="Image scale augmentation factor.")
    fliplr: float = Field(0.5, description="Left-right flip probability.")
    mosaic: float = Field(1.0, description="Mosaic augmentation probability.")
    mixup: float = Field(0.0, description="MixUp augmentation probability.")
    erasing: float = Field(0.4, description="Random erasing augmentation probability.")
    auto_augment: str = Field("randaugment", description="Auto augmentation policy.")
    close_mosaic: int = Field(10, description="Disable mosaic in the last N epochs.")
    crop_fraction: float = Field(1.0, description="Crop fraction for random cropping.")
    perspective: float = Field(0.0, description="Random perspective distortion.")
    degrees: float = Field(0.0, description="Image rotation degrees.")
    shear: float = Field(0.0, description="Shear augmentation degrees.")

    # --- other ---
    export_format: Optional[str] = Field("pt", description="Format to export best weights after training.")
    tensorboard_dir: str = Field("training/tensorboard", description="TensorBoard logs output directory.")

    @property
    def save_dir(self) -> str:
        """Output directory for this run."""
        if self.project:
            return f"{self.project}/{self.name}"
        return self.name

    class Config:
        extra = "forbid"
        validate_assignment = True
