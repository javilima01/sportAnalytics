"""Create reviewable YOLO detection datasets from videos."""

import hashlib
import math
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

from .config import setup_logger
from .dataset import SPLITS, atomic_write, class_names, read_names


class DatasetCreator:
    def __init__(
        self,
        model_path,
        output_dir="datasets/generated_dataset",
        sample_prob=0.1,
        splits=(0.7, 0.2, 0.1),
        imgsz=1280,
        logger=None,
        conf=0.25,
        device=None,
        seed=0,
        include_empty=False,
    ):
        if not math.isfinite(sample_prob) or not 0 <= sample_prob <= 1:
            raise ValueError("sample_prob must be between 0 and 1.")
        if (
            len(splits) != 3
            or any(not math.isfinite(v) or v < 0 for v in splits)
            or not math.isclose(sum(splits), 1)
        ):
            raise ValueError("splits must contain three nonnegative ratios summing to 1.")
        if imgsz < 32:
            raise ValueError("imgsz must be at least 32.")
        if not math.isfinite(conf) or not 0 <= conf <= 1:
            raise ValueError("conf must be between 0 and 1.")
        if Path(model_path).suffix.lower() != ".pt":
            raise ValueError("Image tagging requires a YOLO .pt checkpoint.")
        self.logger = logger or setup_logger("DatasetCreator")
        self.output_dir = Path(output_dir)
        self.sample_prob, self.splits, self.imgsz = sample_prob, splits, imgsz
        self.conf, self.device, self.include_empty = conf, device, include_empty
        self.random = random.Random(seed)
        self.model = YOLO(str(model_path))
        if self.model.task != "detect":
            raise ValueError("Image tagging requires a detection model.")
        self.names = class_names(self.model.names)
        yaml_path = self.output_dir / "data.yaml"
        if yaml_path.exists() and read_names(yaml_path) != self.names:
            raise ValueError("Existing dataset class names differ from the model.")
        for split in SPLITS:
            for kind in ("images", "labels"):
                (self.output_dir / kind / split).mkdir(parents=True, exist_ok=True)
        self._write_yaml()

    def _download_youtube(self, url, directory):
        output = Path(directory) / "video.mp4"
        # Find the runtime beside Python even when the virtualenv is not activated.
        runtime = shutil.which("deno", path=str(Path(sys.executable).parent)) or shutil.which(
            "deno"
        )
        if runtime is None:
            raise RuntimeError("YouTube downloading requires Deno. Install requirements.txt first.")
        command = [
            sys.executable,
            "-m",
            "yt_dlp",
            "--no-playlist",
            "--js-runtimes",
            f"deno:{runtime}",
            "-f",
            # No audio/merging is needed; prefer H.264 for OpenCV compatibility.
            "bestvideo[ext=mp4][vcodec^=avc1]/best[ext=mp4]/bestvideo[ext=mp4]",
            "-o",
            str(output),
            url,
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"Video download failed: {error.stderr}") from error
        return output

    def create_from_video(self, video_source, segments=None):
        if segments is not None:
            if not segments or any(
                not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end)
                for start, end in segments
            ):
                raise ValueError("Segments must have finite times with 0 <= start < end.")
        source = str(video_source)
        is_url = source.startswith(("https://", "http://"))
        identity = source if is_url else str(Path(source).resolve())
        prefix = (
            ("video" if is_url else Path(source).stem)
            + "_"
            + hashlib.sha256(identity.encode()).hexdigest()[:12]
        )
        with tempfile.TemporaryDirectory(prefix="image-tagging-") as directory:
            video = self._download_youtube(source, directory) if is_url else Path(source)
            self._process_video(video, prefix, segments)
        return self.output_dir

    def _process_video(self, video, prefix, segments):
        cap = cv2.VideoCapture(str(video))
        saved = 0
        try:
            if not cap.isOpened():
                raise FileNotFoundError(f"Cannot open video: {video}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if not math.isfinite(fps) or fps <= 0 or not math.isfinite(count) or count < 1:
                raise ValueError(f"Video has invalid FPS or frame count: {video}")
            count = int(count)
            # Merge overlapping segments so a frame is considered only once.
            ranges = (
                [(0, count)]
                if segments is None
                else sorted(
                    (min(count, int(start * 60 * fps)), min(count, math.ceil(end * 60 * fps)))
                    for start, end in segments
                )
            )
            merged = []
            for start, end in ranges:
                if start >= end:
                    continue
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
                else:
                    merged.append((start, end))
            if not merged:
                raise ValueError("No requested segments overlap the video.")
            for start, end in merged:
                if start and not cap.set(cv2.CAP_PROP_POS_FRAMES, start):
                    raise RuntimeError(f"Cannot seek to frame {start}.")
                for index in range(start, end):
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError(f"Cannot read video frame {index}.")
                    if self.random.random() >= self.sample_prob:
                        continue
                    split = self.random.choices(SPLITS, weights=self.splits)[0]
                    stem = f"{prefix}_{index:06d}"
                    # Existing frames may have been manually reviewed; never overwrite them.
                    if any(
                        (self.output_dir / kind / candidate / f"{stem}{suffix}").exists()
                        for candidate in SPLITS
                        for kind, suffix in (("images", ".jpg"), ("labels", ".txt"))
                    ):
                        continue
                    result = self.model.predict(
                        frame, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False
                    )[0]
                    if result.boxes is None:
                        raise ValueError("Model did not return detection boxes.")
                    if not len(result.boxes) and not self.include_empty:
                        continue
                    image_path = self.output_dir / "images" / split / f"{stem}.jpg"
                    label_path = self.output_dir / "labels" / split / f"{stem}.txt"
                    if not cv2.imwrite(str(image_path), frame):
                        raise OSError(f"Cannot save image: {image_path}")
                    try:
                        rows = [
                            f"{int(cls)} " + " ".join(f"{v:.6f}" for v in coordinates)
                            for cls, coordinates in zip(
                                result.boxes.cls.tolist(), result.boxes.xywhn.tolist()
                            )
                        ]
                        atomic_write(label_path, "\n".join(rows) + ("\n" if rows else ""))
                    except Exception:
                        image_path.unlink(missing_ok=True)
                        raise
                    saved += 1
        finally:
            cap.release()
        self.logger.info(
            "Dataset generation complete: %s images saved to %s", saved, self.output_dir
        )

    def _write_yaml(self):
        path = self.output_dir / "data.yaml"
        data = {
            "path": str(self.output_dir.resolve()),
            **{split: f"images/{split}" for split in SPLITS},
            "names": self.names,
        }
        # Preserve additional metadata when extending an existing dataset.
        if path.exists():
            existing = yaml.safe_load(path.read_text())
            for split in SPLITS:
                if existing.get(split) != data[split]:
                    raise ValueError("Generation requires images/<split> dataset paths.")
            data = {**existing, **data}
        atomic_write(path, yaml.safe_dump(data, sort_keys=False))
        return path
