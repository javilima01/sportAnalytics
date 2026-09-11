"""Inspect, edit, and predict YOLO detections with OpenCV."""

import math
import random
from pathlib import Path

import cv2
from ultralytics import YOLO

from .config import setup_logger
from .dataset import IMAGE_SUFFIXES, class_names, read_labels, read_names, write_labels


class Visualizer:
    def __init__(
        self,
        dataset_dir,
        split="train",
        names=None,
        max_images=5,
        window_name="YOLO Label Editor",
        logger=None,
        model_path=None,
        conf=0.25,
        imgsz=1280,
        device=None,
        half=False,
    ):
        if max_images < 1:
            raise ValueError("max_images must be positive.")
        self.dataset_dir = Path(dataset_dir)
        self.names = (
            class_names(names) if names is not None else read_names(self.dataset_dir / "data.yaml")
        )
        self.max_images, self.window_name = max_images, window_name
        self.logger = logger or setup_logger("Visualizer")
        self.model = None
        if model_path is not None:
            if Path(model_path).suffix.lower() != ".pt":
                raise ValueError("Prediction preview requires a YOLO .pt checkpoint.")
            if not math.isfinite(conf) or not 0 <= conf <= 1:
                raise ValueError("conf must be between 0 and 1.")
            if imgsz < 32:
                raise ValueError("imgsz must be at least 32.")
            self.model = YOLO(str(model_path))
            if self.model.task != "detect":
                raise ValueError("Prediction preview requires a detection model.")
            self.prediction_names = class_names(self.model.names)
        self.conf, self.imgsz, self.device = conf, imgsz, device
        self.half = half
        for image_dir, label_dir in (
            (self.dataset_dir / split / "images", self.dataset_dir / split / "labels"),
            (self.dataset_dir / "images" / split, self.dataset_dir / "labels" / split),
        ):
            if image_dir.is_dir():
                self.image_dir, self.label_dir = image_dir, label_dir
                break
        else:
            raise FileNotFoundError(f"Cannot locate images for split {split!r}.")
        self.image_paths = sorted(
            path
            for path in self.image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if len({path.stem for path in self.image_paths}) != len(self.image_paths):
            raise ValueError(
                "Images with the same stem would share a label file; rename them first."
            )
        self.index = 0
        self.current_image = None
        self.boxes = []
        self.start_point = self.cursor = self.selected_box = None
        self.image_modified = False

    def _draw_boxes(self, image, boxes):
        drawn = image.copy()
        for index, (cls, coordinates) in enumerate(boxes):
            x1, y1, x2, y2 = map(round, coordinates)
            color = (0, 0, 255) if index == self.selected_box else (0, 255, 0)
            cv2.rectangle(drawn, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                drawn,
                self.names[cls],
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
        if self.start_point is not None and self.cursor is not None:
            cv2.rectangle(drawn, self.start_point, self.cursor, (255, 0, 0), 1)
        return drawn

    def _delete_selected(self):
        if self.selected_box is not None:
            del self.boxes[self.selected_box]
            self.selected_box = None
            self.image_modified = True

    def _mouse_callback(self, event, x, y, flags, param):
        if self.current_image is None:
            return
        height, width = self.current_image.shape[:2]
        x, y = max(0, min(width, x)), max(0, min(height, y))
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MBUTTONDOWN):
            self.start_point = self.cursor = None
            self.selected_box = next(
                (
                    i
                    for i in reversed(range(len(self.boxes)))
                    if self.boxes[i][1][0] <= x <= self.boxes[i][1][2]
                    and self.boxes[i][1][1] <= y <= self.boxes[i][1][3]
                ),
                None,
            )
            if event == cv2.EVENT_MBUTTONDOWN or flags & cv2.EVENT_FLAG_CTRLKEY:
                self._delete_selected()
            elif self.selected_box is None:
                self.start_point = self.cursor = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.start_point is not None:
            self.cursor = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.start_point is not None:
            x1, y1 = self.start_point
            self.start_point = self.cursor = None
            if abs(x - x1) > 5 and abs(y - y1) > 5:
                self.boxes.append((0, (min(x1, x), min(y1, y), max(x1, x), max(y1, y))))
                self.selected_box = len(self.boxes) - 1
                self.image_modified = True

    def _predict(self, image):
        result = self.model.predict(
            image,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            half=self.half,
            verbose=False,
        )[0]
        boxes = result.boxes
        return [
            (int(cls), float(score), tuple(coordinates))
            for cls, score, coordinates in zip(
                boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()
            )
        ]

    def _draw_predictions(self, image, predictions):
        drawn = image.copy()
        for cls, confidence, coordinates in predictions:
            x1, y1, x2, y2 = map(round, coordinates)
            cv2.rectangle(drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                drawn,
                f"{self.prediction_names[cls]} {confidence:.2f}",
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        return drawn

    def visualize(self, save_dir=None):
        if not self.image_paths:
            self.logger.warning("No images found in %s", self.image_dir)
            return
        output = Path(save_dir) if save_dir else None
        if output:
            if output.resolve() == self.image_dir.resolve():
                raise ValueError("Save previews outside the source image directory.")
            output.mkdir(parents=True, exist_ok=True)
        try:
            for path in random.sample(
                self.image_paths, min(self.max_images, len(self.image_paths))
            ):
                image = cv2.imread(str(path))
                if image is None:
                    self.logger.warning("Cannot read %s", path)
                    continue
                if self.model is not None:
                    preview = self._draw_predictions(image, self._predict(image))
                else:
                    label_path = self.label_dir / f"{path.stem}.txt"
                    if not label_path.exists():
                        self.logger.warning("Missing labels for %s", path.name)
                    boxes = read_labels(label_path, image.shape, self.names)
                    preview = self._draw_boxes(image, boxes)
                if output:
                    if not cv2.imwrite(str(output / path.name), preview):
                        raise OSError(f"Cannot save preview for {path}.")
                else:
                    cv2.imshow(self.window_name, preview)
                    if cv2.waitKeyEx(0) == 27:
                        break
        finally:
            if output is None:
                cv2.destroyAllWindows()

    def _save_current(self, force=False):
        if self.current_image is not None and (force or self.image_modified):
            path = self.label_dir / f"{self.image_paths[self.index].stem}.txt"
            write_labels(path, self.boxes, self.current_image.shape)
            self.image_modified = False
            self.logger.info("Saved %s", path.name)

    def edit(self):
        """Edit all images; save changes on navigation, Esc, and window close."""
        if self.model is not None:
            raise ValueError("Prediction previews are read-only; drop --model to edit labels.")
        if not self.image_paths:
            self.logger.warning("No images found in %s", self.image_dir)
            return
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        try:
            cv2.setMouseCallback(self.window_name, self._mouse_callback)
            while self.image_paths:
                path = self.image_paths[self.index]
                self.current_image = cv2.imread(str(path))
                if self.current_image is None:
                    self.logger.warning("Cannot read %s", path)
                    self.image_paths.pop(self.index)
                    self.index = max(0, min(self.index, len(self.image_paths) - 1))
                    continue
                self.boxes = read_labels(
                    self.label_dir / f"{path.stem}.txt", self.current_image.shape, self.names
                )
                self.start_point = self.cursor = self.selected_box = None
                self.image_modified = False
                while True:
                    cv2.imshow(self.window_name, self._draw_boxes(self.current_image, self.boxes))
                    key = cv2.waitKeyEx(30)
                    if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                        self._save_current()
                        return
                    if key == 27:
                        self._save_current()
                        return
                    if key == ord("s"):
                        self._save_current(force=True)
                    elif key == ord("c") and self.selected_box is not None:
                        cls, box = self.boxes[self.selected_box]
                        self.boxes[self.selected_box] = ((cls + 1) % len(self.names), box)
                        self.image_modified = True
                    elif key == ord("x"):
                        self._delete_selected()
                    elif key == ord("d"):
                        path.unlink()
                        (self.label_dir / f"{path.stem}.txt").unlink(missing_ok=True)
                        self.image_paths.pop(self.index)
                        self.image_modified = False
                        self.index = max(0, min(self.index, len(self.image_paths) - 1))
                        break
                    # OpenCV key codes for Linux, Windows, and macOS; N/P work everywhere.
                    elif key in (
                        ord("n"),
                        83,
                        65363,
                        2555904,
                        63235,
                        ord("p"),
                        81,
                        65361,
                        2424832,
                        63234,
                    ):
                        self._save_current()
                        delta = 1 if key in (ord("n"), 83, 65363, 2555904, 63235) else -1
                        self.index = max(0, min(self.index + delta, len(self.image_paths) - 1))
                        break
        finally:
            cv2.destroyAllWindows()
