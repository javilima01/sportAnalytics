"""Opt-in real YouTube download, CPU tagging, and CLI preview validation."""

import os
import subprocess
import sys
from pathlib import Path

import cv2
import pytest

from src.dataset import read_labels, read_names


@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("RUN_YOUTUBE_SMOKE") != "1", reason="Requires YouTube and model download access"
)
def test_youtube_to_labeled_images(tmp_path):
    root = Path(__file__).resolve().parents[1]
    dataset = tmp_path / "dataset"
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    environment = {
        **os.environ,
        "TMPDIR": str(downloads),
        "TMP": str(downloads),
        "TEMP": str(downloads),
    }
    result = subprocess.run(
        [
            sys.executable,
            str(root / "main.py"),
            "generate",
            "--model",
            str(root / "models/yolov8n.pt"),
            "--video",
            "https://www.youtube.com/watch?v=jNQXAC9IVRw",
            "--output",
            str(dataset),
            "--segments",
            "0:0.1",
            "--sample_prob",
            "0.1",
            "--splits",
            "1",
            "0",
            "0",
            "--imgsz",
            "640",
            "--device",
            "cpu",
            "--seed",
            "0",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    images = sorted(dataset.glob("images/train/*.jpg"))
    labels = sorted(dataset.glob("labels/train/*.txt"))
    assert images and len(images) == len(labels)
    names = read_names(dataset / "data.yaml")
    for path in images:
        image = cv2.imread(str(path))
        assert image is not None
        assert read_labels(dataset / "labels/train" / f"{path.stem}.txt", image.shape, names)
    assert not list(downloads.glob("image-tagging-*"))
    preview = tmp_path / "previews"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "main.py"),
            "visualize",
            "--dataset",
            str(dataset),
            "--save_dir",
            str(preview),
            "--max_images",
            "2",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(list(preview.glob("*.jpg"))) == min(2, len(images))
