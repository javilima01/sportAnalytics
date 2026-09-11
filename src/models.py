"""Application training options; Ultralytics owns model/augmentation defaults."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    data: str
    model: str = "yolov8x.pt"
    epochs: int = Field(50, ge=1)
    imgsz: int = Field(1280, ge=32)
    batch: int = Field(6, ge=1)
    device: str | None = None
    workers: int = Field(8, ge=0)
    project: str = "training/finetune"
    name: str = "yolov8x_football"
    patience: int = Field(100, ge=0)
    seed: int = Field(0, ge=0)
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    half: bool = False
    tensorboard_dir: str | None = None

    def training_args(self):
        return self.model_dump(
            exclude={"model", "tensorboard_dir", "precision", "half"}, exclude_none=True
        )
