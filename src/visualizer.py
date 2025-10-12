import cv2
import random
from pathlib import Path
from typing import Optional, List, Tuple
from .config import setup_logger


class Visualizer:
    """
    Interactive YOLO label viewer and editor.
    Allows adding, deleting, and modifying bounding boxes,
    as well as removing images from the dataset.
    """

    def __init__(
        self,
        dataset_dir: str,
        split: str = "train",
        names: Optional[List[str]] = None,
        max_images: int = 5,
        window_name: str = "YOLO Label Editor",
        logger=None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.names = names or self._read_names_from_yaml()
        self.max_images = max_images
        self.window_name = window_name
        self.logger = logger or setup_logger("Visualizer")

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
                f"Could not locate image/label dirs for split '{split}'"
            )

        self.image_paths = sorted(
            list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png"))
        )
        self.index = 0
        self.current_image = None
        self.boxes = []
        self.start_point = None
        self.selected_box = None
        self.image_modified = False

    def _read_names_from_yaml(self) -> List[str]:
        yaml_path = self.dataset_dir / "data.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found in {self.dataset_dir}")
        names = []
        with yaml_path.open("r") as f:
            for line in f:
                if ":" in line and not line.strip().startswith("#"):
                    parts = line.strip().split(":")
                    if len(parts) == 2 and parts[0].strip().isdigit():
                        names.append(parts[1].strip())
        return names

    def _read_labels(
        self, label_path: Path, img_shape: Tuple[int, int]
    ) -> List[Tuple[int, Tuple[int, int, int, int]]]:
        h, w = img_shape[:2]
        boxes = []
        if not label_path.exists():
            return boxes
        with label_path.open("r") as f:
            for line in f:
                vals = line.strip().split()
                if len(vals) != 5:
                    continue
                cls, x, y, bw, bh = map(float, vals)
                cls = int(cls)
                x1 = int((x - bw / 2) * w)
                y1 = int((y - bh / 2) * h)
                x2 = int((x + bw / 2) * w)
                y2 = int((y + bh / 2) * h)
                boxes.append((cls, (x1, y1, x2, y2)))
        return boxes

    def _write_labels(self, label_path: Path, boxes, img_shape):
        h, w = img_shape[:2]
        with label_path.open("w") as f:
            for cls, (x1, y1, x2, y2) in boxes:
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    def _draw_boxes(self, img, boxes):
        drawn = img.copy()
        for idx, (cls, (x1, y1, x2, y2)) in enumerate(boxes):
            color = (0, 255, 0) if idx != self.selected_box else (0, 0, 255)
            cv2.rectangle(drawn, (x1, y1), (x2, y2), color, 2)
            label = self.names[cls] if cls < len(self.names) else str(cls)
            cv2.putText(
                drawn,
                label,
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
        return drawn

    def _mouse_callback(self, event, x, y, flags, param):
        # CTRL + left click deletes a box (fallback if middle click is unreliable)
        if event == cv2.EVENT_LBUTTONDOWN and (flags & cv2.EVENT_FLAG_CTRLKEY):
            for i, (cls, (x1, y1, x2, y2)) in enumerate(self.boxes):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    del self.boxes[i]
                    self.image_modified = True
                    return

        if event == cv2.EVENT_MBUTTONDOWN:
            for i, (cls, (x1, y1, x2, y2)) in enumerate(self.boxes):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    del self.boxes[i]
                    self.image_modified = True
                    return

        if event == cv2.EVENT_LBUTTONDOWN:
            for i, (cls, (x1, y1, x2, y2)) in enumerate(self.boxes):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.selected_box = i
                    return
            self.start_point = (x, y)
            self.selected_box = None

        elif event == cv2.EVENT_MOUSEMOVE and self.start_point:
            img = self.current_image.copy()
            cv2.rectangle(img, self.start_point, (x, y), (255, 0, 0), 1)
            cv2.imshow(self.window_name, img)

        elif event == cv2.EVENT_LBUTTONUP and self.start_point:
            x1, y1 = self.start_point
            x2, y2 = x, y
            self.start_point = None
            if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                self.boxes.append(
                    (0, (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
                )
                self.image_modified = True

    def visualize(self, save_dir: Optional[str] = None):
        """
        Display or save random images with their YOLO bounding boxes.
        Press ESC to exit when viewing interactively.
        """
        image_paths = list(self.image_dir.glob("*.jpg")) + list(
            self.image_dir.glob("*.png")
        )
        if not image_paths:
            self.logger.warning(f"No images found in {self.image_dir}")
            return

        random.shuffle(image_paths)
        sample_paths = image_paths[: self.max_images]

        for img_path in sample_paths:
            label_path = self.label_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                self.logger.warning(f"Missing label file for {img_path.name}")
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                self.logger.warning(f"Could not read {img_path}")
                continue

            boxes = self._read_labels(label_path, img.shape)
            vis_img = self._draw_boxes(img, boxes)

            if save_dir:
                save_path = Path(save_dir)
                save_path.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_path / img_path.name), vis_img)
            else:
                cv2.imshow(self.window_name, vis_img)
                key = cv2.waitKey(0)
                if key == 27:
                    break

        if not save_dir:
            cv2.destroyAllWindows()

    def edit(self):
        """
        Interactive label editing interface.
        Left-click + drag: create box
        Middle-click: delete box (or press X)
        C: change class
        S: save labels
        D: delete image
        Arrow keys: navigate
        ESC: exit
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        while 0 <= self.index < len(self.image_paths):
            img_path = self.image_paths[self.index]
            label_path = self.label_dir / (img_path.stem + ".txt")

            img = cv2.imread(str(img_path))
            if img is None:
                self.logger.warning(f"Failed to read {img_path}")
                self.index += 1
                continue

            self.current_image = img
            self.boxes = self._read_labels(label_path, img.shape)
            self.selected_box = None
            self.image_modified = False

            while True:
                display = self._draw_boxes(img, self.boxes)
                cv2.imshow(self.window_name, display)
                key = cv2.waitKey(0) & 0xFF

                # Exit editor
                if key == 27:
                    cv2.destroyAllWindows()
                    return

                # Save labels
                elif key == ord("s"):
                    self._write_labels(label_path, self.boxes, img.shape)
                    self.image_modified = False
                    self.logger.info(f"Saved {label_path.name}")

                # Change class of selected box
                elif (
                    key == ord("c")
                    and self.selected_box is not None
                    and 0 <= self.selected_box < len(self.boxes)
                ):
                    cls, box = self.boxes[self.selected_box]
                    new_cls = (cls + 1) % len(self.names)
                    self.boxes[self.selected_box] = (new_cls, box)
                    self.image_modified = True

                # Delete selected box using keyboard (X)
                elif (
                    key == ord("x")
                    and self.selected_box is not None
                    and 0 <= self.selected_box < len(self.boxes)
                ):
                    del self.boxes[self.selected_box]
                    self.selected_box = None
                    self.image_modified = True

                # Delete current image and label file
                elif key == ord("d"):
                    img_path.unlink(missing_ok=True)
                    label_path.unlink(missing_ok=True)
                    self.logger.info(f"Deleted {img_path.name}")
                    self.image_paths.pop(self.index)
                    if self.index >= len(self.image_paths):
                        self.index -= 1
                    break

                # Move to next image
                elif key == 83:  # right arrow
                    if self.image_modified:
                        self._write_labels(label_path, self.boxes, img.shape)
                    self.index += 1
                    break

                # Move to previous image
                elif key == 81:  # left arrow
                    if self.image_modified:
                        self._write_labels(label_path, self.boxes, img.shape)
                    self.index -= 1
                    break

        cv2.destroyAllWindows()
