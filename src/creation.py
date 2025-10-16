import cv2
import random
import tempfile
import subprocess
from pathlib import Path
from typing import Union, List, Tuple
import torch
import onnxruntime as ort
from ultralytics import YOLO
from .config import setup_logger


class DatasetCreator:
    """
    Generate a YOLO-format dataset automatically using a trained YOLO model.
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
        self.logger = logger or setup_logger("DatasetCreator")
        self.model_path = Path(model_path)
        self.model_type = self.model_path.suffix.lower().replace('.', '')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(self.model_path)

        self.output_dir = Path(output_dir)
        self.sample_prob = sample_prob
        self.splits = splits
        self.imgsz = imgsz

        self.logger.info(f"Loaded model on {self.device}: {model_path}")
        self._prepare_folders()

    def load_model(self, model_path: Union[str, Path]):
        model_path = Path(model_path)
        ext = model_path.suffix.lower()

        if ext == ".pt":
            try:
                model = YOLO(str(model_path))
                model.to(self.device)
                self.logger.info("Loaded YOLO model (.pt)")
                return model
            except Exception:
                self.logger.info("Detected quantized PyTorch model (.pt)")
                state_dict = torch.load(model_path, map_location=self.device)
                dummy_model = YOLO("yolov8n.pt").model
                dummy_model.load_state_dict(state_dict, strict=False)
                dummy_model.to(self.device)
                dummy_model.eval()
                return dummy_model

        elif ext == ".onnx":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
            self.logger.info(f"Loaded ONNX model (.onnx) with providers: {providers}")
            session = ort.InferenceSession(str(model_path), providers=providers)
            return session

        else:
            raise ValueError(f"Unsupported model type: {ext}")

    def _prepare_folders(self):
        for split in ["train", "val", "test"]:
            (self.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    def _download_youtube(self, url: str) -> Path:
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
        r = random.random()
        if r < self.splits[0]:
            return "train"
        elif r < self.splits[0] + self.splits[1]:
            return "val"
        return "test"

    def _save_label_file(self, label_path: Path, boxes):
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
            segments = [(0, duration_sec / 60)]

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
                results = self._run_inference(frame_resized)
                boxes = results if isinstance(results, list) else results[0].boxes
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

    def _run_inference(self, frame):
        if self.model_type == "onnx":
            img = frame.transpose(2, 0, 1)
            img = img[None].astype("float32") / 255.0
            ort_inputs = {self.model.get_inputs()[0].name: img}
            return self.model.run(None, ort_inputs)
        elif isinstance(self.model, YOLO):
            return self.model.predict(frame, imgsz=self.imgsz, device=str(self.device), verbose=False)
        else:
            with torch.no_grad():
                tensor = (
                    torch.from_numpy(frame)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                    .to(self.device)
                    / 255.0
                )
                return self.model(tensor)

    def _write_yaml(self) -> Path:
        yaml_path = self.output_dir / "data.yaml"
        names = getattr(self.model, "names", None)
        if not names:
            names = {0: "object"}

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
