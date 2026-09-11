"""Validated campaign settings; paths resolve relative to the configuration file."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Source(Settings):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    url: str
    match_id: str = Field(min_length=1)
    split: Literal["train", "val", "test"]
    start_minutes: float = Field(0, ge=0)
    end_minutes: float = Field(2, gt=0)

    @model_validator(mode="after")
    def interval(self):
        if self.end_minutes <= self.start_minutes:
            raise ValueError("Source end must follow start.")
        return self


class Recipe(Settings):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    hypothesis: str = Field(min_length=1)
    model: str = "models/yolov8n.pt"
    imgsz: int = Field(640, ge=32, le=2048, multiple_of=32)
    batch: int = Field(4, ge=1, le=32)
    epochs: int = Field(100, ge=1, le=1000)
    lr0: float = Field(0.001, gt=0, le=0.1)
    lrf: float = Field(0.01, gt=0, le=1)
    optimizer: Literal["AdamW", "SGD", "auto"] = "AdamW"
    weight_decay: float = Field(0.0005, ge=0, le=0.01)
    warmup_epochs: float = Field(3.0, ge=0, le=20)
    patience: int = Field(30, ge=1, le=500)
    dropout: float = Field(0.0, ge=0, le=0.5)
    mosaic: float = Field(0.5, ge=0, le=1)
    close_mosaic: int = Field(0, ge=0, le=100)
    mixup: float = Field(0.0, ge=0, le=1)
    copy_paste: float = Field(0.0, ge=0, le=1)
    # Random erasing keeps labels while deleting pixels; harmful for tiny objects.
    erasing: float = Field(0.4, ge=0, le=1)
    hsv_h: float = Field(0.015, ge=0, le=1)
    hsv_s: float = Field(0.7, ge=0, le=1)
    hsv_v: float = Field(0.4, ge=0, le=1)
    degrees: float = Field(0, ge=0, le=15)
    translate: float = Field(0.1, ge=0, le=0.9)
    scale: float = Field(0.5, ge=0, le=1)
    shear: float = Field(0.0, ge=0, le=10)
    perspective: float = Field(0.0, ge=0, le=0.001)
    flipud: float = Field(0.0, ge=0, le=1)
    fliplr: float = Field(0.5, ge=0, le=1)


class Acquisition(Settings):
    queries: list[str] = Field(default_factory=lambda: ["football full match wide camera"])
    sources: list[Source] = Field(default_factory=list)
    max_sources: int = Field(6, ge=3, le=100)
    max_frames_per_source: int = Field(12, ge=1, le=1000)
    minutes: float = Field(30, gt=0)
    max_rounds: int = Field(3, ge=1, le=10)
    download_gb: float = Field(10, gt=0)
    storage_gb: float = Field(20, gt=0)
    attempts_per_source: int = Field(2, ge=1, le=5)
    teacher: str = "models/yolov8l.pt"
    teacher_imgsz: int = Field(1280, ge=32, le=2048, multiple_of=32)
    agent_timeout_seconds: float = Field(120, gt=0)
    agent_attempts_per_image: int = Field(2, ge=1, le=5)


class Evaluation(Settings):
    ball_class_id: int = Field(1, ge=0)
    min_matches: int = Field(1, ge=1)
    min_ball_instances: int = Field(1, ge=1)
    nms_iou: float = Field(0.7, gt=0, le=1)
    max_det: int = Field(300, ge=1)
    thresholds: dict[str, float] = Field(
        default_factory=lambda: {
            "ap50_95": 0.70,
            "ap50": 0.85,
            "precision": 0.90,
            "recall": 0.90,
            "f1": 0.85,
        }
    )
    benchmark_frames: int = Field(0, ge=0)
    warmup_frames: int = Field(50, ge=0)
    max_p95_ms: float | None = Field(None, gt=0)

    @model_validator(mode="after")
    def thresholds_valid(self):
        if set(self.thresholds) != {"ap50_95", "ap50", "precision", "recall", "f1"}:
            raise ValueError("All five quality thresholds are required.")
        if any(not 0 < value < 1 for value in self.thresholds.values()):
            raise ValueError("Quality thresholds must be between zero and one.")
        if self.max_p95_ms is not None and self.benchmark_frames < 200:
            raise ValueError("A latency gate requires at least 200 benchmark frames.")
        return self


class Budget(Settings):
    exploration_minutes: float = Field(15, gt=0, le=1440)
    promotion_minutes: float = Field(30, gt=0, le=1440)
    confirmation_minutes: float = Field(120, gt=0)
    evaluation_reserve_seconds: float = Field(120, gt=0)
    max_trials: int = Field(20, ge=1)
    max_exploration_trials: int = Field(8, ge=1)
    max_hours: float = Field(10, gt=0)
    final_test_minutes: float = Field(30, gt=0)
    confirmation_epochs: int = Field(200, ge=1)


class Fallback(Settings):
    enabled: bool = True
    executable: str = "opencode"
    model: str = Field(
        "opencode/muse-spark-1.3-contributor-free", pattern=r"^opencode/[a-zA-Z0-9._-]+-free$"
    )
    codex_retry_seconds: float = Field(300, ge=30)
    opencode_retry_seconds: float = Field(60, ge=30)
    max_wait_hours: float = Field(24, gt=0, le=168)


class DataGrowth(Settings):
    enabled: bool = True
    min_train_images: int = Field(48, ge=1)
    max_rounds: int = Field(3, ge=0, le=100)
    sources_per_round: int = Field(3, ge=1, le=10)
    trials_per_round: int = Field(2, ge=1)


class Diagnostics(Settings):
    enabled: bool = True
    max_actions: int = Field(12, ge=0, le=100)
    minutes: float = Field(5, gt=0, le=15)
    max_minutes: float = Field(30, gt=0)
    max_images: int = Field(16, ge=2, le=64)
    max_decisions: int = Field(32, ge=1, le=200)


class Autonomy(Settings):
    """The agent may raise working allowances up to these user-owned ceilings."""

    enabled: bool = True
    allow_benchmark_growth: bool = True
    allow_label_review: bool = True
    max_trial_minutes: float = Field(240, gt=0, le=1440)
    max_hours: float = Field(72, gt=0)
    max_trials: int = Field(80, ge=1)
    max_exploration_trials: int = Field(60, ge=1)
    max_data_rounds: int = Field(30, ge=0, le=100)
    max_acquisition_minutes: float = Field(60, gt=0)
    max_download_gb: float = Field(50, gt=0)
    max_storage_gb: float = Field(100, gt=0)
    max_diagnostic_actions: int = Field(50, ge=0, le=100)
    max_diagnostic_minutes: float = Field(15, gt=0, le=15)
    max_diagnostic_total_minutes: float = Field(120, gt=0)
    max_decisions: int = Field(200, ge=1, le=200)


class Campaign(Settings):
    campaign_id: str = Field("football-pilot", pattern=r"^[a-zA-Z0-9_-]+$")
    output_dir: Path = Path("experiments/football-pilot")
    prior_campaign: Path | None = None
    dataset_dir: Path = Path("datasets/football-pilot")
    device: str = "mps"
    workers: int = Field(2, ge=0)
    # Training compute precision. fp32 is the default; fp16/bf16 use autocast via
    # src/precision.py (Ultralytics gates AMP to CUDA). bf16 needs no loss scaling.
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    # Inference precision for teacher labeling, evaluation, and benchmarks.
    half: bool = False
    names: list[str] = Field(default_factory=lambda: ["player", "ball"], min_length=2)
    taxonomy: str = (
        "Player includes active outfield players and goalkeepers. Exclude referees, "
        "spectators and staff. Ball is the match football, including partially visible balls "
        "only when identifiable. Reject ambiguous frames rather than guessing."
    )
    codex_executable: str = "codex"
    codex_model: str | None = "gpt-6-astra"
    codex_reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = "high"
    fallback: Fallback = Field(default_factory=Fallback)
    proposals: Literal["codex", "queue"] = "codex"
    acquisition: Acquisition = Field(default_factory=Acquisition)
    evaluation: Evaluation = Field(default_factory=Evaluation)
    budget: Budget = Field(default_factory=Budget)
    data_growth: DataGrowth = Field(default_factory=DataGrowth)
    diagnostics: Diagnostics = Field(default_factory=Diagnostics)
    autonomy: Autonomy = Field(default_factory=Autonomy)
    recipes: list[Recipe] = Field(
        default_factory=lambda: [
            Recipe(id="nano640", hypothesis="Establish a small-model baseline."),
            Recipe(id="nano960", hypothesis="Higher resolution improves ball recall.", imgsz=960),
            *[
                Recipe(
                    id=f"{size}640",
                    hypothesis=f"Measure whether YOLOv8{size} meets the fixed quality gates.",
                    model=f"models/yolov8{size}.pt",
                )
                for size in ("s", "m", "l", "x")
            ],
        ]
    )

    @model_validator(mode="after")
    def unique_ids(self):
        for values in (self.recipes, self.acquisition.sources):
            if len({item.id for item in values}) != len(values):
                raise ValueError("Recipe and source IDs must be unique.")
        if len(set(self.names)) != len(self.names):
            raise ValueError("Class names must be unique.")
        if self.evaluation.ball_class_id >= len(self.names):
            raise ValueError("Ball class ID is outside the taxonomy.")
        return self


def base_models(cfg):
    """Configured checkpoints plus existing architecture/weight files in their directories."""
    models = {r.model for r in cfg.recipes}
    for directory in {Path(model).parent for model in models}:
        for pattern in ("*.pt", "*.yaml"):
            models.update(str(path) for path in sorted(directory.glob(pattern)))
    return models


def load_campaign(path):
    path = Path(path).resolve()
    cfg = Campaign.model_validate(yaml.safe_load(path.read_text()))
    for field in ("output_dir", "dataset_dir"):
        value = getattr(cfg, field)
        setattr(cfg, field, (path.parent / value).resolve())
    if cfg.prior_campaign is not None:
        cfg.prior_campaign = (path.parent / cfg.prior_campaign).resolve()
        if cfg.prior_campaign == cfg.output_dir:
            raise ValueError("Prior campaign must differ from the new output directory.")
    if (
        cfg.output_dir == cfg.dataset_dir
        or cfg.output_dir in cfg.dataset_dir.parents
        or cfg.dataset_dir in cfg.output_dir.parents
    ):
        raise ValueError("Dataset and experiment output directories must be separate.")
    for recipe in cfg.recipes:
        if not Path(recipe.model).is_absolute():
            recipe.model = str((path.parent / recipe.model).resolve())
    cfg.acquisition.teacher = str((path.parent / cfg.acquisition.teacher).resolve())
    for source in cfg.acquisition.sources:
        if not source.url.startswith(("http://", "https://")):
            source.url = str((path.parent / source.url).resolve())
    return cfg
