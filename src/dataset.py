"""Shared YOLO dataset parsing and safe label persistence."""

import math
import os
import tempfile
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def class_names(names):
    """Normalize YAML list/mapping names and require contiguous detection IDs."""
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or not names:
        raise ValueError("Dataset names must be a nonempty list or numeric mapping.")
    if any(type(key) is not int for key in names):
        raise ValueError("Class IDs must be integers starting at zero.")
    if sorted(names) != list(range(len(names))):
        raise ValueError("Class IDs must be contiguous and start at zero.")
    if any(not isinstance(value, str) or not value.strip() for value in names.values()):
        raise ValueError("Class names must be nonempty strings.")
    return dict(sorted(names.items()))


def read_names(path):
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a dataset YAML mapping.")
    return class_names(data.get("names"))


def atomic_write(path, text):
    """Replace a file only after its complete contents have been written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_labels(path, shape, names):
    """Read normalized detection labels as floating-point pixel boxes.

    Reject malformed labels instead of silently dropping them during editing.
    """
    path = Path(path)
    if not path.exists():
        return []
    height, width = shape[:2]
    boxes = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            cls, cx, cy, bw, bh = map(float, line.split())
            if not all(math.isfinite(v) for v in (cls, cx, cy, bw, bh)):
                raise ValueError
            if not cls.is_integer() or int(cls) not in names:
                raise ValueError
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                raise ValueError
            boxes.append(
                (
                    int(cls),
                    (
                        (cx - bw / 2) * width,
                        (cy - bh / 2) * height,
                        (cx + bw / 2) * width,
                        (cy + bh / 2) * height,
                    ),
                )
            )
        except ValueError as error:
            raise ValueError(f"{path}:{number}: invalid YOLO detection label.") from error
    return boxes


def write_labels(path, boxes, shape):
    height, width = shape[:2]
    lines = []
    for cls, (x1, y1, x2, y2) in boxes:
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            raise ValueError("Box coordinates must be finite.")
        x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
        y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Boxes must have positive width and height.")
        values = (
            (x1 + x2) / (2 * width),
            (y1 + y2) / (2 * height),
            (x2 - x1) / width,
            (y2 - y1) / height,
        )
        lines.append(f"{cls} " + " ".join(f"{value:.6f}" for value in values))
    atomic_write(path, "\n".join(lines) + ("\n" if lines else ""))
