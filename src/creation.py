import cv2
import random
import tempfile
import subprocess
from pathlib import Path
from typing import Union, List, Tuple
from ultralytics import YOLO
from .config import setup_logger


class DatasetCreator:
    """
    Generate a YOLO-format dataset automatically using a trained YOLO model.

    - Accepts local video paths or YouTube URLs.
    - Can process specific time segments (start:end in minutes).
    - Samples frames probabilistically to limit dataset size.
    - Saves predictions in YOLO format.
    - Automatically creates the dataset structure and YAML file.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        output_dir: Union[str, Path] = "datasets/generated_dataset",
        sample_prob: float = 0.1,
        splits: tuple[float, float, float] = (0.7, 0.2, 0.1),
        imgsz: int = 1280,
        logger=None,
    ):
        """
        Args:
            model_path: Path to YOLO weights (e.g., 'runs/train/best.pt').
            output_dir: Root directory where dataset will be created.
            sample_prob: Probability to keep a given frame (0-1).
            splits: Train/val/test ratio tuple (must sum to 1).
            imgsz: Resize frames before inference (e.g., 640, 1280).
            logger: Optional logger instance; defaults to setup_logger().
        """
        self.model = YOLO(str(model_path))
        self.output_dir = Path(output_dir)
        self.sample_prob = sample_prob
        self.splits = splits
        self.imgsz = imgsz
        self.logger = logger or setup_logger("DatasetCreator")

        self.logger.info(f"Loaded YOLO model: {model_path}")
        self._prepare_folders()

    def _prepare_folders(self):
        """Create folder structure for YOLO dataset."""
        for split in ["train", "val", "test"]:
            (self.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    def _download_youtube(self, url: str) -> Path:
        """Download a YouTube video to a temporary file using yt-dlp."""
        tmp_dir = Path(tempfile.mkdtemp())
        out_path = tmp_dir / "video.mp4"
        cmd = ["yt-dlp", "-f", "mp4", "-o", str(out_path), url]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to download YouTube video: {url}\n{e.stderr.decode()}"
            )
        self.logger.info(f"Downloaded YouTube video to {out_path}")
        return out_path

    def _choose_split(self) -> str:
        """Randomly assign a frame to train, val, or test split."""
        r = random.random()
        if r < self.splits[0]:
            return "train"
        elif r < self.splits[0] + self.splits[1]:
            return "val"
        return "test"

    def _save_label_file(self, label_path: Path, boxes):
        """Write YOLO-format label file for one image."""
        with label_path.open("w") as f:
            for box in boxes:
                cls = int(box.cls)
                xywhn = box.xywhn.view(-1).tolist()
                f.write(f"{cls} {' '.join(f'{x:.6f}' for x in xywhn)}\n")

    def create_from_video(
        self,
        video_source: Union[str, Path],
        segments: List[Tuple[float, float]] = None,
    ):
        """
        Generate dataset from a local video or YouTube URL.
        You can restrict processing to specific segments.

        Args:
            video_source: Path to a local video file or a YouTube URL.
            segments: Optional list of (start_min, end_min) tuples specifying which parts of the video to process.
        """
        if isinstance(video_source, str) and video_source.startswith("http"):
            video_source = self._download_youtube(video_source)

        video_source = Path(video_source)
        cap = cv2.VideoCapture(str(video_source))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_source}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0
        self.logger.info(
            f"Processing video: {video_source.name} | {total_frames} frames | {duration_sec/60:.1f} min | {fps:.2f} FPS"
        )

        if not segments:
            segments = [(0, duration_sec / 60)]  # default: whole video

        frame_idx, saved = 0, 0

        for (start_min, end_min) in segments:
            start_frame = int(start_min * 60 * fps)
            end_frame = int(end_min * 60 * fps)
            end_frame = min(end_frame, total_frames - 1)

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self.logger.info(f"Processing segment {start_min:.2f}–{end_min:.2f} min ({start_frame}-{end_frame} frames)")

            while cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if random.random() > self.sample_prob:
                    continue

                frame_resized = cv2.resize(frame, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
                results = self.model.predict(frame_resized, imgsz=self.imgsz, verbose=False)
                boxes = results[0].boxes
                if not boxes or len(boxes) == 0:
                    continue

                split = self._choose_split()
                img_dir = self.output_dir / "images" / split
                label_dir = self.output_dir / "labels" / split

                img_name = f"{video_source.stem}_{frame_idx:06d}.jpg"
                label_name = img_name.replace(".jpg", ".txt")

                img_path = img_dir / img_name
                label_path = label_dir / label_name

                cv2.imwrite(str(img_path), frame)
                self._save_label_file(label_path, boxes)
                saved += 1

                if saved % 50 == 0:
                    self.logger.info(f"Saved {saved} labeled frames so far...")

        cap.release()
        self.logger.info(f"Dataset generation complete. Total labeled frames: {saved}")

        yaml_path = self._write_yaml()
        self.logger.info(f"Dataset YAML created at {yaml_path}")
        return self.output_dir

    def _write_yaml(self) -> Path:
        """Write the YOLO dataset YAML file."""
        yaml_path = self.output_dir / "data.yaml"
        names = self.model.names

        lines = [
            f"path: {self.output_dir}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
        ]
        lines += [f"  {i}: {name}" for i, name in names.items()]

        yaml_path.write_text("\n".join(lines))
        return yaml_path
