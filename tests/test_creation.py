import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
import torch
from ultralytics.engine.results import Boxes
from yt_dlp import YoutubeDL

from src.creation import DatasetCreator


@pytest.fixture
def model(monkeypatch):
    model = Mock(task="detect", names={0: "player", 1: "ball: white"})
    boxes = Boxes(torch.tensor([[16, 8, 48, 40, 0.9, 1]]), orig_shape=(48, 64))
    model.predict.return_value = [SimpleNamespace(boxes=boxes)]
    monkeypatch.setattr("src.creation.YOLO", Mock(return_value=model))
    return model


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "match.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 48))
    assert writer.isOpened()
    for value in (0, 80, 160, 240):
        writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    writer.release()
    return path


def test_real_video_generation_and_safe_rerun(tmp_path, model, video):
    creator = DatasetCreator("best.pt", tmp_path / "data", sample_prob=1, splits=(1, 0, 0))
    creator.create_from_video(video, segments=[(0, 0.01), (0.005, 0.02)])
    images = sorted(creator.output_dir.glob("images/train/*.jpg"))
    labels = sorted(creator.output_dir.glob("labels/train/*.txt"))
    assert len(images) == len(labels) == 4
    assert model.predict.call_count == 4
    assert cv2.imread(str(images[0])).shape == (48, 64, 3)
    assert labels[0].read_text() == "1 0.500000 0.500000 0.500000 0.666667\n"
    labels[0].write_text("reviewed")
    creator.splits = (0, 1, 0)
    creator.create_from_video(video)
    assert labels[0].read_text() == "reviewed"
    assert not list(creator.output_dir.glob("images/val/*.jpg"))
    assert model.predict.call_count == 4


@pytest.mark.parametrize("include_empty,expected", [(False, 0), (True, 4)])
def test_background_frames(tmp_path, model, video, include_empty, expected):
    model.predict.return_value = [SimpleNamespace(boxes=Boxes(torch.empty((0, 6)), (48, 64)))]
    creator = DatasetCreator(
        "best.pt", tmp_path / "data", sample_prob=1, include_empty=include_empty
    )
    creator.create_from_video(video)
    labels = list(creator.output_dir.glob("labels/*/*.txt"))
    assert len(labels) == expected
    assert all(path.read_text() == "" for path in labels)


@pytest.mark.parametrize(
    "options",
    [
        {"sample_prob": -1},
        {"sample_prob": float("nan")},
        {"splits": (0.2, 0.2, 0.2)},
        {"splits": (-1, 1, 1)},
        {"conf": 2},
        {"imgsz": 0},
    ],
)
def test_bad_options_before_loading_model(tmp_path, model, options):
    with pytest.raises(ValueError):
        DatasetCreator("best.pt", tmp_path / "data", **options)
    assert not (tmp_path / "data").exists()


def test_incompatible_classes_do_not_overwrite_yaml(tmp_path, model):
    root = tmp_path / "data"
    root.mkdir()
    yaml = root / "data.yaml"
    yaml.write_text("names: [different]\n")
    with pytest.raises(ValueError, match="differ"):
        DatasetCreator("best.pt", root)
    assert yaml.read_text() == "names: [different]\n"


def test_capture_released_on_inference_error(tmp_path, model, monkeypatch):
    capture = Mock()
    capture.get.side_effect = [5, 4]
    capture.read.return_value = (True, np.zeros((48, 64, 3), dtype=np.uint8))
    monkeypatch.setattr("src.creation.cv2.VideoCapture", Mock(return_value=capture))
    model.predict.side_effect = RuntimeError("inference error")
    creator = DatasetCreator("best.pt", tmp_path / "data", sample_prob=1)
    with pytest.raises(RuntimeError, match="inference error"):
        creator.create_from_video("match.avi")
    capture.release.assert_called_once()


def test_failed_image_write_creates_no_label(tmp_path, video, model, monkeypatch):
    creator = DatasetCreator("best.pt", tmp_path / "data", sample_prob=1)
    monkeypatch.setattr("src.creation.cv2.imwrite", lambda *args: False)
    with pytest.raises(OSError, match="Cannot save image"):
        creator.create_from_video(video)
    assert not list(creator.output_dir.glob("labels/*/*.txt"))


def test_download_cleanup_and_command(tmp_path, model, monkeypatch):
    creator = DatasetCreator("best.pt", tmp_path / "data")
    download_paths = []

    def download(command, **kwargs):
        assert "--no-playlist" in command
        assert command[command.index("--js-runtimes") + 1].startswith("deno:")
        # Real format selection must accept video-only streams, as YouTube may
        # provide no combined audio/video MP4 at all.
        with YoutubeDL({"format": command[command.index("-f") + 1], "quiet": True}) as downloader:
            selected = downloader.process_ie_result(
                {
                    "id": "test",
                    "title": "test",
                    "formats": [
                        {
                            "format_id": "video",
                            "url": "https://example.com/video.mp4",
                            "ext": "mp4",
                            "vcodec": "avc1.4d400c",
                            "acodec": "none",
                        },
                    ],
                },
                download=False,
            )
            assert selected["format_id"] == "video"
        path = Path(command[command.index("-o") + 1])
        path.write_bytes(b"video")
        download_paths.append(path)

    monkeypatch.setattr("src.creation.subprocess.run", download)
    monkeypatch.setattr(creator, "_process_video", Mock(side_effect=RuntimeError("failed")))
    with pytest.raises(RuntimeError):
        creator.create_from_video("https://example.com/video")
    assert not download_paths[0].parent.exists()
    monkeypatch.setattr(
        "src.creation.subprocess.run",
        Mock(side_effect=subprocess.CalledProcessError(1, "yt-dlp", stderr="download error")),
    )
    with pytest.raises(RuntimeError, match="download error"):
        creator.create_from_video("https://example.com/video")


def test_downloader_finds_runtime_in_unactivated_virtualenv(tmp_path, model, monkeypatch):
    creator = DatasetCreator("best.pt", tmp_path / "data")
    runtime = str(Path(sys.executable).parent / "deno")
    which = Mock(return_value=runtime)
    monkeypatch.setattr("src.creation.shutil.which", which)
    run = Mock()
    monkeypatch.setattr("src.creation.subprocess.run", run)
    creator._download_youtube("https://example.com/video", tmp_path)
    which.assert_called_once_with("deno", path=str(Path(sys.executable).parent))
    command = run.call_args.args[0]
    assert command[command.index("--js-runtimes") + 1] == f"deno:{runtime}"


def test_missing_runtime_reports_installation_help(tmp_path, model, monkeypatch):
    creator = DatasetCreator("best.pt", tmp_path / "data")
    monkeypatch.setattr("src.creation.shutil.which", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="Install requirements.txt"):
        creator._download_youtube("https://example.com/video", tmp_path)


def test_clip_download_bounds_time_and_resolution(tmp_path, monkeypatch):
    run = Mock()
    monkeypatch.setattr("src.creation.subprocess.run", run)
    monkeypatch.setattr("src.creation.shutil.which", lambda *a, **k: "/bin/deno")
    monkeypatch.setattr("imageio_ffmpeg.get_ffmpeg_exe", lambda: "/bin/ffmpeg")
    DatasetCreator._download_youtube(
        None, "https://example.com/video", tmp_path, section=(1200, 1320), max_height=1280
    )
    command = run.call_args.args[0]
    assert command[command.index("--download-sections") + 1] == "*1200-1320"
    assert command[command.index("--ffmpeg-location") + 1] == "/bin/ffmpeg"
    assert "--force-keyframes-at-cuts" in command
    assert "--ignore-config" in command
    # Exercise yt-dlp's selector: a high-resolution stream must not bypass the cap.
    with YoutubeDL({"format": command[command.index("-f") + 1], "quiet": True}) as downloader:
        info = downloader.process_ie_result(
            {
                "id": "test",
                "title": "test",
                "formats": [
                    {
                        "format_id": str(height),
                        "height": height,
                        "url": f"https://example.com/{height}.mp4",
                        "ext": "mp4",
                        "vcodec": "avc1.4d400c",
                        "acodec": "none",
                    }
                    for height in (720, 1080, 2160)
                ],
            },
            download=False,
        )
    assert info["format_id"] == "1080"
