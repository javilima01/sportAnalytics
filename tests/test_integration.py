"""Real OpenCV + YOLO + TensorBoard smoke test using a tiny synthetic dataset."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from ultralytics import YOLO

from src.creation import DatasetCreator
from src.dataset import write_labels
from src.models import TrainConfig
from src.trainer import YOLOFineTuner
from src.visualizer import Visualizer


@pytest.mark.integration
def test_real_inference_training_and_checkpoint_reload(tmp_path, monkeypatch):
    # Only disable optional internet checks/font downloads; model operations are real.
    import ultralytics.data.utils
    import ultralytics.utils.checks

    monkeypatch.setattr(ultralytics.utils.checks, "check_pip_update_available", lambda: False)
    monkeypatch.setattr(ultralytics.data.utils, "check_font", lambda *args, **kwargs: None)
    monkeypatch.chdir(tmp_path)
    checkpoint = tmp_path / "initial.pt"
    model = YOLO("yolov8n.yaml")
    model.save(checkpoint)
    video = tmp_path / "video.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 64))
    assert writer.isOpened()
    for _ in range(4):
        image = np.zeros((64, 64, 3), np.uint8)
        cv2.rectangle(image, (16, 16), (48, 48), (255, 255, 255), -1)
        writer.write(image)
    writer.release()
    creator = DatasetCreator(
        checkpoint,
        tmp_path / "dataset",
        sample_prob=1,
        splits=(1, 0, 0),
        imgsz=64,
        device="cpu",
        include_empty=True,
    )
    creator.create_from_video(video)
    images = sorted((creator.output_dir / "images/train").glob("*.jpg"))
    assert len(images) == 4
    for image in images:
        write_labels(
            creator.output_dir / "labels/train" / f"{image.stem}.txt",
            [(0, (16, 16, 48, 48))],
            (64, 64),
        )
    # Move a distinct generated frame into validation.
    val_image = images.pop()
    val_image.rename(creator.output_dir / "images/val" / val_image.name)
    (creator.output_dir / "labels/train" / f"{val_image.stem}.txt").rename(
        creator.output_dir / "labels/val" / f"{val_image.stem}.txt"
    )
    Visualizer(creator.output_dir).visualize(tmp_path / "previews")
    assert len(list((tmp_path / "previews").glob("*.jpg"))) == 3
    cfg = TrainConfig(
        data=str(creator.output_dir / "data.yaml"),
        model=str(checkpoint),
        epochs=1,
        imgsz=64,
        batch=2,
        workers=0,
        device="cpu",
        project=str(tmp_path / "training"),
        name="smoke",
    )
    trainer = YOLOFineTuner(cfg)
    try:
        trainer.train()
        metrics = trainer.validate()
        assert "metrics/mAP50(B)" in metrics.results_dict
        run_dir = Path(trainer.model.trainer.save_dir)
        best = run_dir / "weights/best.pt"
        assert best.is_file()
        assert list((run_dir / "tensorboard").glob("events.out.tfevents.*"))
        reloaded = YOLO(str(best))
        results = reloaded.predict(
            np.zeros((64, 64, 3), np.uint8), imgsz=64, device="cpu", verbose=False
        )
        assert results[0].boxes is not None
    finally:
        trainer.close()
