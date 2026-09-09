"""Bounded diagnostics on training/validation only; never deployment candidates."""

import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from ..dataset import read_labels, write_labels
from .config import Campaign, Recipe
from .controller import ROOT, load_state, remaining_seconds, source_fingerprint
from .runtime import file_hash, read_json, run_process, save_json


def diagnostic_allowance(cfg):
    records = read_json(cfg.output_dir / "diagnostics/state.json", [])
    spent = sum(
        r["reserved_seconds"] if r["status"] == "running" else r["seconds"] for r in records
    )
    if len(records) >= cfg.diagnostics.max_actions:
        return 0
    return max(
        0,
        min(
            cfg.diagnostics.minutes * 60,
            cfg.diagnostics.max_minutes * 60 - spent,
            remaining_seconds(cfg, load_state(cfg)),
        ),
    )


def run_diagnostic(cfg, kind, trial, directory, executor=run_process):
    """Reserve before launch; interrupted diagnostics are charged and never rerun."""
    if (cfg.output_dir / "final_test.json").exists():
        raise ValueError("Diagnostics cannot run after final testing.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "outcome.json"
    if result_path.exists():
        return read_json(result_path)
    ledger_path = cfg.output_dir / "diagnostics/state.json"
    ledger = read_json(ledger_path, [])
    existing = next((r for r in ledger if r["directory"] == str(directory)), None)
    if existing:
        existing.update(status="interrupted", seconds=existing["reserved_seconds"])
        save_json(ledger_path, ledger)
        result = {
            "status": "interrupted",
            "kind": kind,
            "reason": "Interrupted diagnostic retained and charged; not repeated.",
        }
        save_json(result_path, result)
        return result
    timeout = diagnostic_allowance(cfg)
    if timeout <= 0:
        raise ValueError("Diagnostic budget exhausted.")
    if trial and file_hash(trial["metrics"]["checkpoint"]) != trial["metrics"]["checkpoint_sha256"]:
        raise ValueError("Diagnostic checkpoint changed.")
    entry = {
        "directory": str(directory),
        "kind": kind,
        "status": "running",
        "reserved_seconds": timeout,
        "seconds": 0,
    }
    ledger.append(entry)
    save_json(ledger_path, ledger)
    job = {
        "kind": "diagnostic",
        "diagnostic": kind,
        "campaign": cfg.model_dump(mode="json"),
        "trial": trial,
        "folder": str(directory),
        "seconds": timeout,
    }
    save_json(directory / "job.json", job)
    save_json(
        directory / "provenance.json",
        {
            "dataset_version": read_json(cfg.output_dir / "snapshot.json")["version"],
            "source_hashes": source_fingerprint(),
            "trial_id": trial["id"] if trial else None,
        },
    )
    start = time.monotonic()
    result = {"kind": kind, "status": "interrupted"}
    try:
        executor(
            [sys.executable, "-m", "src.research.worker", str(directory / "job.json")],
            timeout=timeout,
            log=directory / "run.log",
            cwd=ROOT,
            progress_log=directory / "progress.log",
        )
        payload = read_json(directory / "result.json")
        if not isinstance(payload, dict):
            raise ValueError("Diagnostic produced no valid result.")
        result.update(status="completed", evidence=payload)
    except (RuntimeError, TimeoutError, ValueError, OSError) as error:
        result.update(status="failed", reason=str(error))
    finally:
        entry.update(
            status=result["status"],
            seconds=(timeout if result["status"] == "interrupted" else time.monotonic() - start),
        )
        save_json(ledger_path, ledger)
        save_json(result_path, result)
    return result


def selected_images(cfg, split):
    """Prefer ball-bearing frames from different recorded matches, then fill the sample."""
    if split not in ("train", "val"):
        raise ValueError("Diagnostics only inspect train and val.")
    records = [
        json.loads(line) for line in (cfg.dataset_dir / "manifest.jsonl").read_text().splitlines()
    ]
    records = [r for r in records if r["split"] == split]

    def has_ball(r):
        return any(
            line.split()[0] == str(cfg.evaluation.ball_class_id)
            for line in (cfg.dataset_dir / r["label"]).read_text().splitlines()
            if line.strip()
        )

    records.sort(key=lambda r: (not has_ball(r), r["image"]))
    chosen, deferred, matches = [], [], set()
    for record in records:
        if record["match_id"] in matches:
            deferred.append(record)
        else:
            chosen.append(record)
            matches.add(record["match_id"])
    return [cfg.dataset_dir / r["image"] for r in (chosen + deferred)[: cfg.diagnostics.max_images]]


def boxes_for(cfg, path, image):
    return read_labels(
        cfg.dataset_dir / "labels" / path.parent.name / f"{path.stem}.txt",
        image.shape,
        dict(enumerate(cfg.names)),
    )


def inspect_data(cfg, trial, folder):
    from PIL import Image
    from ultralytics.cfg import get_cfg
    from ultralytics.data.augment import RandomPerspective
    from ultralytics.data.dataset import YOLODataset

    recipe = Recipe.model_validate(trial["recipe"]) if trial else cfg.recipes[0]
    summary = {"imgsz": recipe.imgsz, "splits": {}, "annotation_accuracy_verified": False}
    for split in ("train", "val"):
        sizes, counts = [], np.zeros(len(cfg.names), dtype=int)
        for path in sorted((cfg.dataset_dir / "images" / split).iterdir()):
            if not path.is_file() or path.suffix.lower() not in (
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp",
            ):
                continue
            with Image.open(path) as image:
                w, h = image.size
            labels = read_labels(
                cfg.dataset_dir / "labels" / split / f"{path.stem}.txt",
                (h, w),
                dict(enumerate(cfg.names)),
            )
            for cls, (x1, y1, x2, y2) in labels:
                counts[cls] += 1
                if cls == cfg.evaluation.ball_class_id:
                    sizes.append(
                        [(x2 - x1) * recipe.imgsz / max(h, w), (y2 - y1) * recipe.imgsz / max(h, w)]
                    )
        summary["splits"][split] = {
            "class_counts": counts.tolist(),
            "ball_boxes": len(sizes),
            "median_ball_wh_pixels": np.median(sizes, axis=0).tolist() if sizes else None,
            "balls_at_most_4px_in_one_dimension": sum(min(b) <= 4 for b in sizes),
        }
        # Analytical crops of original pixels, with label outlines, for agent visual review.
        tiles = []
        for path in selected_images(cfg, split):
            im = cv2.imread(str(path))
            for cls, (x1, y1, x2, y2) in boxes_for(cfg, path, im):
                if cls != cfg.evaluation.ball_class_id or len(tiles) >= 16:
                    continue
                radius = max(40, int(max(x2 - x1, y2 - y1)))
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                left, top = max(0, cx - radius), max(0, cy - radius)
                crop = im[top : cy + radius, left : cx + radius].copy()
                cv2.rectangle(
                    crop,
                    (int(x1 - left), int(y1 - top)),
                    (int(x2 - left), int(y2 - top)),
                    (0, 0, 255),
                    1,
                )
                tile = np.full((240, 240, 3), 255, np.uint8)
                tile[:216] = cv2.resize(crop, (240, 216))
                cv2.putText(
                    tile, str(len(tiles)), (4, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
                )
                tiles.append(tile)
        if tiles:
            while len(tiles) % 4:
                tiles.append(np.full((240, 240, 3), 255, np.uint8))
            cv2.imwrite(
                str(folder / f"{split}-ball-crops.jpg"),
                np.vstack([np.hstack(tiles[i : i + 4]) for i in range(0, len(tiles), 4)]),
            )
    paths = selected_images(cfg, "train")
    listing = folder / "training-sample.txt"
    listing.write_text("\n".join(str(p.resolve()) for p in paths) + "\n")
    args = recipe.model_dump(exclude={"id", "hypothesis", "model"})
    random.seed(0)
    np.random.seed(0)
    before, after = [], []
    original = RandomPerspective.__call__

    def measure(transform, labels):
        before.extend(labels["cls"].reshape(-1).tolist())
        result = original(transform, labels)
        after.extend(result["cls"].reshape(-1).tolist())
        return result

    RandomPerspective.__call__ = measure
    retained, missing = 0, 0
    try:
        ds = YOLODataset(
            str(listing),
            imgsz=recipe.imgsz,
            batch_size=recipe.batch,
            data={"names": dict(enumerate(cfg.names)), "nc": len(cfg.names)},
            hyp=get_cfg(overrides=args),
            augment=True,
            stride=32,
        )
        for index in range(len(ds)):
            sample = ds[index]
            h, w = sample["img"].shape[1:]
            boxes = sample["bboxes"][
                sample["cls"].reshape(-1) == cfg.evaluation.ball_class_id
            ].numpy() * [w, h, w, h]
            for cx, cy, bw, bh in boxes:
                retained += 1
                eligible = False
                for stride in (8, 16, 32):
                    xs, ys = (
                        (np.arange(w // stride) + 0.5) * stride,
                        (np.arange(h // stride) + 0.5) * stride,
                    )
                    eligible |= bool(
                        ((xs > cx - bw / 2) & (xs < cx + bw / 2)).any()
                        and ((ys > cy - bh / 2) & (ys < cy + bh / 2)).any()
                    )
                missing += not eligible
    finally:
        RandomPerspective.__call__ = original
    summary["augmentation_sample"] = {
        "images": len(paths),
        "seed": 0,
        "ball_occurrences_before_affine": before.count(cfg.evaluation.ball_class_id),
        "ball_occurrences_after_affine": after.count(cfg.evaluation.ball_class_id),
        "retained_balls": retained,
        "balls_without_grid_center": missing,
        "assumed_yolov8_strides": [8, 16, 32],
        "note": "Sampled occurrences, not unique labels; geometry is necessary, not sufficient for assignment.",
    }
    return summary


def crop_overfit(cfg, trial, folder, seconds):
    from .worker import train_job

    data = folder / "sanity-data"
    sources = []
    for path in selected_images(cfg, "train"):
        im = cv2.imread(str(path))
        boxes = boxes_for(cfg, path, im)
        balls = [b for c, b in boxes if c == cfg.evaluation.ball_class_id]
        if not balls:
            continue
        x1, y1, x2, y2 = balls[0]
        h, w = im.shape[:2]
        left, top = (
            max(0, min(w - 320, int((x1 + x2) / 2) - 160)),
            max(0, min(h - 320, int((y1 + y2) / 2) - 160)),
        )
        crop = im[top : top + 320, left : left + 320]
        ch, cw = crop.shape[:2]
        labels = []
        for cls, (a, b, c, d) in boxes:
            a, b, c, d = max(0, a - left), max(0, b - top), min(cw, c - left), min(ch, d - top)
            if c > a and d > b:
                labels.append((cls, (a, b, c, d)))
        for split in ("train", "val"):
            dest = data / "images" / split
            dest.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dest / path.name), crop)
            write_labels(data / "labels" / split / f"{path.stem}.txt", labels, crop.shape)
        sources.append(path.name)
    if not sources:
        raise ValueError("No ball-bearing training crops available.")
    (data / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(data.resolve()),
                "names": cfg.names,
                "train": "images/train",
                "val": "images/val",
            }
        )
    )
    recipe = Recipe.model_validate(trial["recipe"]).model_copy(
        update={"id": "diagnostic-overfit", "imgsz": 320, "mosaic": 0, "scale": 0, "epochs": 200}
    )
    isolated = cfg.model_copy(update={"dataset_dir": data, "output_dir": folder, "workers": 0})
    train_job(
        {
            "campaign": isolated.model_dump(mode="json"),
            "recipe": recipe.model_dump(),
            "folder": str(folder),
            "seed": 0,
            "training_seconds": max(0.1, seconds - min(60, seconds * 0.2)),
        }
    )
    metrics = read_json(folder / "metrics.json")
    return {
        "interpretation": "Memorization check on the same training crops; NOT generalization or acceptance.",
        "generalization": False,
        "source_images": sources,
        "ball": metrics["ball"],
        "macro": metrics["macro"],
        "epochs": metrics["epochs_completed"],
    }


def diagnostic_job(job):
    from ultralytics import YOLO

    from .evaluation import evaluate

    cfg, folder = Campaign.model_validate(job["campaign"]), Path(job["folder"])
    trial, kind = job.get("trial"), job["diagnostic"]
    if kind == "inspect_data":
        result = inspect_data(cfg, trial, folder)
    elif kind == "evaluate_train":
        paths = selected_images(cfg, "train")
        metrics, _ = evaluate(
            YOLO(trial["metrics"]["checkpoint"]),
            cfg.dataset_dir,
            "train",
            cfg.evaluation,
            cfg.device,
            trial["recipe"]["imgsz"],
            image_paths=paths,
        )
        result = {
            "interpretation": "Sampled training-set fit, not generalization.",
            "images": len(paths),
            "ball": metrics["ball"],
            "macro": metrics["macro"],
            "confidence": metrics["confidence"],
            "validation_ball": trial["metrics"]["ball"],
        }
    elif kind == "overfit_crops":
        result = crop_overfit(cfg, trial, folder, job["seconds"])
    else:
        raise ValueError("Unknown diagnostic.")
    save_json(folder / "result.json", result)
