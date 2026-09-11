"""Freeze image/label provenance and reject contaminated evaluation splits."""

import json
import re
from pathlib import Path

import cv2

from ..dataset import IMAGE_SUFFIXES, read_labels, read_names
from .runtime import file_hash, value_hash


class InsufficientCoverage(ValueError):
    """More accepted source images are needed before this dataset can be frozen."""


def match_key(value):
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def audit_dataset(directory, evaluation, splits=("train", "val", "test")):
    root = Path(directory).resolve()
    names = read_names(root / "data.yaml")
    import yaml

    layout = yaml.safe_load((root / "data.yaml").read_text())
    if Path(layout.get("path", "")).resolve() != root or any(
        layout.get(split) != f"images/{split}" for split in ("train", "val", "test")
    ):
        raise ValueError("Dataset YAML must point to the manifest's local split directories.")
    if evaluation.ball_class_id not in names:
        raise ValueError("Required ball class is absent.")
    manifest = root / "manifest.jsonl"
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    groups, hashes, paths, source_splits = {}, {}, set(), {}
    counts = {
        split: {"images": 0, "matches": set(), "instances": [0] * len(names)} for split in splits
    }
    for record in records:
        split = record["split"]
        if split not in ("train", "val", "test"):
            raise ValueError("Unknown manifest split.")
        group = match_key(record["match_id"])
        if group in groups and groups[group] != split:
            raise ValueError(f"Match leakage: {group}")
        groups[group] = split
        source = record.get("source_url")
        if source:
            if source in source_splits and source_splits[source] != split:
                raise ValueError("Source video leakage across splits.")
            source_splits[source] = split
        if record["image_sha256"] in hashes and hashes[record["image_sha256"]] != split:
            raise ValueError("Duplicate image appears across splits.")
        hashes[record["image_sha256"]] = split
        image = (root / record["image"]).resolve()
        label = (root / record["label"]).resolve()
        if root not in image.parents or root not in label.parents:
            raise ValueError("Manifest path escapes the dataset.")
        if (
            image.parent != root / "images" / split
            or label != root / "labels" / split / f"{image.stem}.txt"
        ):
            raise ValueError("Manifest split and image/label paths disagree.")
        if image in paths:
            raise ValueError("Duplicate manifest image.")
        paths.add(image)
        if record["review_status"] != "agent_labeled":
            raise ValueError("Dataset contains unreviewed images.")
        if file_hash(image) != record["image_sha256"] or file_hash(label) != record["label_sha256"]:
            raise ValueError("Dataset contents changed after labeling.")
        if split not in counts:
            continue
        data = cv2.imread(str(image))
        if data is None:
            raise ValueError(f"Unreadable image: {image}")
        boxes = read_labels(label, data.shape, names)
        counts[split]["images"] += 1
        counts[split]["matches"].add(group)
        for cls, _ in boxes:
            counts[split]["instances"][cls] += 1
    actual = {
        p.resolve() for p in (root / "images").rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
    }
    if actual != paths:
        raise ValueError("Manifest does not cover the exact dataset image set.")
    expected_labels = {(root / record["label"]).resolve() for record in records}
    if {p.resolve() for p in (root / "labels").rglob("*.txt")} != expected_labels:
        raise ValueError("Manifest does not cover the exact dataset label set.")
    for split, count in counts.items():
        count["matches"] = len(count["matches"])
        if not count["images"] or any(value == 0 for value in count["instances"]):
            raise InsufficientCoverage(
                f"Split {split} has missing images or unsupported target classes."
            )
        if split != "train" and (
            count["matches"] < evaluation.min_matches
            or count["instances"][evaluation.ball_class_id] < evaluation.min_ball_instances
        ):
            raise InsufficientCoverage(f"Split {split} has insufficient evaluation coverage.")
    return {
        "version": value_hash(
            {"manifest": file_hash(manifest), "yaml": file_hash(root / "data.yaml")}
        ),
        "counts": counts,
        "benchmark_version": value_hash([r for r in records if r["split"] != "train"]),
        "label_provenance": "agent_labeled",
        "independent_ground_truth": False,
    }
