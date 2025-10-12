import cv2
import random
from pathlib import Path
from typing import Optional, List, Tuple
from .config import setup_logger


class Visualizer:
    """
    Visualize YOLO-format datasets by drawing bounding boxes on sample images.

    Supports both:
    - YOLOv8 default structure:
        dataset/
            images/train, images/val, images/test
            labels/train, labels/val, labels/test
    - Roboflow structure:
        dataset/
            train/images, train/labels
            valid/images, valid/labels
            test/images, test/labels
    """

    def __init__(
        self,
        dataset_dir: str,
        split: str = "train",
        names: Optional[List[str]] = None,
        max_images: int = 5,
        window_name: str = "YOLO Dataset Visualization",
        logger=None
    ):
        """
        Args:
            dataset_dir: Root directory of the YOLO dataset.
            split: Dataset split to visualize (train, valid/val, test).
            names: Optional list of class names; if None, read from data.yaml.
            max_images: Maximum number of random images to visualize.
            window_name: Name for the OpenCV window.
            logger: Optional logger, defaults to setup_logger().
        """
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.names = names or self._read_names_from_yaml()
        self.max_images = max_images
        self.window_name = window_name
        self.logger = logger or setup_logger("Visualizer")

        # Support both YOLOv8 and Roboflow layouts
        yolo8_img_dir = self.dataset_dir / "images" / split
        yolo8_lbl_dir = self.dataset_dir / "labels" / split
        roboflow_img_dir = self.dataset_dir / split / "images"
        roboflow_lbl_dir = self.dataset_dir / split / "labels"

        if roboflow_img_dir.exists() and roboflow_lbl_dir.exists():
            self.image_dir = roboflow_img_dir
            self.label_dir = roboflow_lbl_dir
        elif yolo8_img_dir.exists() and yolo8_lbl_dir.exists():
            self.image_dir = yolo8_img_dir
            self.label_dir = yolo8_lbl_dir
        else:
            raise FileNotFoundError(
                f"Could not locate images/labels for split '{split}' "
                f"in YOLOv8 or Roboflow structure under {dataset_dir}"
            )

        self.logger.info(f"Visualizer initialized for split: {split}")
        self.logger.info(f"Images: {self.image_dir}")
        self.logger.info(f"Labels: {self.label_dir}")

    def _read_names_from_yaml(self) -> List[str]:
        """Reads class names from the dataset's data.yaml file."""
        yaml_path = self.dataset_dir / "data.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found in {self.dataset_dir}")

        names = []
        with yaml_path.open("r") as f:
            lines = f.readlines()
            start = False
            for line in lines:
                if line.strip().startswith("names:"):
                    start = True
                    continue
                if start and line.strip().startswith(("-", " ")):
                    parts = line.strip().split(":")
                    if len(parts) == 2:
                        names.append(parts[1].strip())
        return names

    def _read_labels(self, label_path: Path, img_shape: Tuple[int, int]) -> List[Tuple[int, Tuple[int, int, int, int]]]:
        """Reads YOLO label file and converts normalized xywh to pixel coordinates."""
        h, w = img_shape[:2]
        boxes = []
        with label_path.open("r") as f:
            for line in f:
                values = line.strip().split()
                if len(values) != 5:
                    continue
                cls, x, y, bw, bh = map(float, values)
                cls = int(cls)
                x1 = int((x - bw / 2) * w)
                y1 = int((y - bh / 2) * h)
                x2 = int((x + bw / 2) * w)
                y2 = int((y + bh / 2) * h)
                boxes.append((cls, (x1, y1, x2, y2)))
        return boxes

    def _draw_boxes(self, image, boxes):
        """Draw bounding boxes and labels on the image."""
        for cls, (x1, y1, x2, y2) in boxes:
            color = tuple(random.randint(0, 255) for _ in range(3))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = self.names[cls] if self.names and cls < len(self.names) else str(cls)
            cv2.putText(image, label, (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        return image

    def visualize(self, save_dir: Optional[str] = None):
        """
        Visualize random samples from the YOLO dataset.
        If save_dir is provided, saves visualizations instead of displaying them.
        """
        image_paths = list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png"))
        if not image_paths:
            self.logger.warning(f"No images found in {self.image_dir}")
            return

        random.shuffle(image_paths)
        sample_paths = image_paths[: self.max_images]

        for img_path in sample_paths:
            label_path = self.label_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                self.logger.warning(f"Label file missing for {img_path.name}")
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                self.logger.warning(f"Failed to read {img_path}")
                continue

            boxes = self._read_labels(label_path, image.shape)
            image = self._draw_boxes(image, boxes)

            if save_dir:
                save_path = Path(save_dir)
                save_path.mkdir(parents=True, exist_ok=True)
                out_file = save_path / img_path.name
                cv2.imwrite(str(out_file), image)
            else:
                cv2.imshow(self.window_name, image)
                key = cv2.waitKey(0)
                if key == 27:  # ESC to exit early
                    break

        if not save_dir:
            cv2.destroyAllWindows()
