"""Disposable training/evaluation worker, supervised by the campaign controller."""

import os
import platform
import signal
import sys
import threading
import time
from importlib.metadata import version
from pathlib import Path

from ultralytics import YOLO

from .config import Campaign, Recipe
from .evaluation import evaluate, localization_report
from .progress import TrainingProgress
from .runtime import file_hash, read_json, save_json


def schedule_fraction(epoch, epochs, elapsed, seconds):
    """Reach the final LR by the epoch cap or 90% of the wall-time allowance."""
    return min(1.0, max(epoch / max(1, epochs - 1), elapsed / (seconds * 0.9)))


def train_job(job):
    cfg = Campaign.model_validate(job["campaign"])
    recipe = Recipe.model_validate(job["recipe"])
    folder = Path(job["folder"])
    model = YOLO(recipe.model)
    start = time.monotonic()
    training_seconds = job["training_seconds"]
    progress = TrainingProgress(folder, training_seconds, cfg.evaluation.ball_class_id)
    progress.write(
        f"starting {recipe.id} ({recipe.model}), device={cfg.device}, "
        f"imgsz={recipe.imgsz}, seed={job['seed']}, "
        f"up to {recipe.epochs} epochs / {training_seconds / 60:.1f} min training"
    )
    progress.attach(model)
    timed_out = False

    def adapt_schedule(trainer):
        def factor(epoch):
            fraction = schedule_fraction(
                epoch, recipe.epochs, time.monotonic() - start, training_seconds
            )
            return 1.0 - fraction * (1.0 - trainer.args.lrf)

        trainer.lf = factor  # Ultralytics warmup uses the same factor as the epoch scheduler.
        trainer.scheduler.lr_lambdas = [factor] * len(trainer.optimizer.param_groups)

    def stop_at_budget(trainer):
        nonlocal timed_out
        if time.monotonic() - start >= training_seconds:
            timed_out = True
            trainer.stop = True

    model.add_callback("on_train_start", adapt_schedule)
    model.add_callback("on_train_batch_end", stop_at_budget)
    model.add_callback("on_train_epoch_end", stop_at_budget)
    arguments = recipe.model_dump(exclude={"id", "hypothesis", "model"})
    model.train(
        **arguments,
        data=str(cfg.dataset_dir / "data.yaml"),
        device=cfg.device,
        workers=cfg.workers,
        seed=job["seed"],
        project=str(folder),
        name="training",
        exist_ok=True,
        plots=False,
        amp=False,
        save=True,
        val=True,
    )
    checkpoint = Path(model.trainer.best)
    if not checkpoint.is_file():
        raise RuntimeError("Training did not produce best.pt.")
    # Reload so parameter count is measured before prediction fuses layers.
    best = YOLO(str(checkpoint))
    parameters = sum(p.numel() for p in best.model.parameters())
    progress.write("training finished; evaluating best.pt on validation images")
    metrics, frames = evaluate(
        best, cfg.dataset_dir, "val", cfg.evaluation, cfg.device, recipe.imgsz
    )
    metrics.update(
        localization=localization_report(frames, cfg.evaluation.ball_class_id),
        parameters=parameters,
        checkpoint_bytes=checkpoint.stat().st_size,
        checkpoint=str(checkpoint),
        checkpoint_sha256=file_hash(checkpoint),
        seed=job["seed"],
        epochs_completed=model.trainer.epoch + 1,
        runtime_seconds=time.monotonic() - start,
        training={
            "requested_epochs": recipe.epochs,
            "completed_epochs": model.trainer.epoch + 1,
            "seconds_allowed": training_seconds,
            "stop_reason": "time_budget"
            if timed_out
            else "epochs_complete"
            if model.trainer.epoch + 1 >= recipe.epochs
            else "early_stopping",
            "schedule": "epoch-or-time-linear-v1",
            "first_epoch": progress.history[0] if progress.history else None,
            "last_epoch": progress.history[-1] if progress.history else None,
        },
    )
    save_json(folder / "predictions.json", frames)
    save_json(folder / "metrics.json", metrics)
    progress.write(
        f"validation complete | macro AP50-95={metrics['macro']['ap50_95']:.4f}, "
        f"ball AP50-95={metrics['ball']['ap50_95']:.4f}, "
        f"ball precision={metrics['ball']['precision']:.4f}, "
        f"ball recall={metrics['ball']['recall']:.4f} | checkpoint={checkpoint}"
    )


def test_job(job):
    cfg = Campaign.model_validate(job["campaign"])
    model = YOLO(job["checkpoint"])
    metrics, frames = evaluate(
        model,
        cfg.dataset_dir,
        "test",
        cfg.evaluation,
        cfg.device,
        job["imgsz"],
        confidence=job["confidence"],
    )
    save_json(Path(job["folder"]) / "predictions.json", frames)
    save_json(Path(job["folder"]) / "metrics.json", metrics)


def main():
    parent = os.getppid()

    def monitor_parent():
        while True:
            time.sleep(0.5)
            if os.getppid() != parent:
                os.killpg(os.getpgrp(), signal.SIGTERM)
                return

    # Workers are launched in their own session; do not leave training behind on controller death.
    if os.getpgrp() == os.getpid():
        threading.Thread(target=monitor_parent, daemon=True).start()
    job = read_json(sys.argv[1])
    save_json(
        Path(job["folder"]) / "environment.json",
        {
            "platform": platform.platform(),
            "python": sys.version,
            "packages": {
                name: version(name) for name in ("torch", "ultralytics", "numpy", "pydantic")
            },
        },
    )
    if job["kind"] == "diagnostic":
        from .diagnostics import diagnostic_job

        diagnostic_job(job)
    elif job["kind"] == "extract":
        from .acquisition import extract_source

        extract_source(job)
    elif job["kind"] == "train":
        train_job(job)
    elif job["kind"] == "test":
        test_job(job)
    elif job["kind"] == "prepare":
        Path(job["model"]).parent.mkdir(parents=True, exist_ok=True)
        model = YOLO(job["model"])
        if model.task != "detect":
            raise ValueError("Starting checkpoint must be a detection model.")
    else:
        raise ValueError("Unknown worker job.")


if __name__ == "__main__":
    main()
