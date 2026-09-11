"""Readable training progress, retained on disk and forwarded by the controller."""

import time
from pathlib import Path

from .runtime import save_json


class TrainingProgress:
    def __init__(self, folder, training_seconds, ball_class_id=1):
        self.path = Path(folder) / "progress.log"
        self.started = time.monotonic()
        self.training_seconds = training_seconds
        self.last_batch_log = self.started
        self.batch = 0
        self.last_epoch = -1
        self.ball_class_id = ball_class_id
        self.history = []

    def write(self, message):
        elapsed = time.monotonic() - self.started
        line = f"[training/{self.path.parent.name}] {elapsed / 60:.1f} min | {message}"
        with self.path.open("a") as output:
            print(line, file=output, flush=True)
        print(line, flush=True)

    def epoch_start(self, trainer):
        self.batch = 0
        self.last_batch_log = time.monotonic()

    def batch_end(self, trainer):
        self.batch += 1
        now = time.monotonic()
        if now - self.last_batch_log >= 30:
            self.write(
                f"epoch {trainer.epoch + 1}/{trainer.epochs}, "
                f"batch {self.batch}/{len(trainer.train_loader)}"
            )
            self.last_batch_log = now

    def epoch_end(self, trainer):
        # Ultralytics also calls on_fit_epoch_end after revalidating best.pt.
        if trainer.epoch == self.last_epoch:
            return
        self.last_epoch = trainer.epoch
        losses = trainer.label_loss_items(trainer.tloss, prefix="train")
        scores = {**losses, **trainer.metrics}
        box_metrics = getattr(
            getattr(getattr(trainer, "validator", None), "metrics", None), "box", None
        )
        ball = None
        if box_metrics is not None:
            for index, cls in enumerate(box_metrics.ap_class_index):
                if int(cls) == self.ball_class_id:
                    ball = dict(
                        zip(
                            ("precision", "recall", "ap50", "ap50_95"),
                            map(float, box_metrics.class_result(index)),
                        )
                    )
                    scores.update({f"val/ball_{k}": v for k, v in ball.items()})
        learning_rates = getattr(trainer, "lr", {})
        scores.update(learning_rates)
        self.history.append(
            {
                "epoch": trainer.epoch + 1,
                "elapsed_seconds": time.monotonic() - self.started,
                "losses": {k: float(v) for k, v in losses.items()},
                "learning_rates": learning_rates,
                "ball": ball,
            }
        )
        save_json(self.path.parent / "training-history.json", self.history)
        values = ", ".join(f"{key}={float(value):.4f}" for key, value in scores.items())
        remaining = max(0, self.training_seconds - (time.monotonic() - self.started))
        self.write(
            f"epoch {trainer.epoch + 1}/{trainer.epochs} complete | {values} | "
            f"training budget remaining {remaining / 60:.1f} min"
        )

    def attach(self, model):
        model.add_callback("on_train_epoch_start", self.epoch_start)
        model.add_callback("on_train_batch_end", self.batch_end)
        model.add_callback("on_fit_epoch_end", self.epoch_end)
