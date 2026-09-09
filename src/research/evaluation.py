"""Pinned AP calculation plus fixed-confidence, class-aware acceptance metrics."""

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.utils.metrics import ap_per_class

from ..dataset import IMAGE_SUFFIXES, read_labels, read_names


def match_detections(predictions, targets, thresholds):
    """Confidence-ordered, one-to-one matching; arrays: cls/conf/xyxy and cls/xyxy."""
    predictions = np.asarray(predictions, dtype=float).reshape(-1, 6)
    targets = np.asarray(targets, dtype=float).reshape(-1, 5)
    order = np.argsort(-predictions[:, 1], kind="stable")
    predictions = predictions[order]
    correct = np.zeros((len(predictions), len(thresholds)), dtype=bool)
    for column, threshold in enumerate(thresholds):
        used = set()
        for row, prediction in enumerate(predictions):
            candidates = [
                i
                for i, target in enumerate(targets)
                if i not in used and target[0] == prediction[0]
            ]
            if not candidates:
                continue
            boxes = targets[candidates, 1:]
            low = np.maximum(prediction[2:4], boxes[:, :2])
            high = np.minimum(prediction[4:6], boxes[:, 2:])
            intersection = np.maximum(high - low, 0).prod(axis=1)
            area = np.prod(prediction[4:6] - prediction[2:4])
            union = area + (boxes[:, 2:] - boxes[:, :2]).prod(axis=1) - intersection
            ious = intersection / np.maximum(union, 1e-12)
            best = int(np.argmax(ious))
            if ious[best] >= threshold:
                used.add(candidates[best])
                correct[row, column] = True
    return predictions, correct


def operating_metrics(predictions, correct, support, confidence):
    per_class = []
    for cls, total in enumerate(support):
        mask = (predictions[:, 0] == cls) & (predictions[:, 1] >= confidence)
        tp = int(correct[mask, 0].sum())
        fp, fn = int(mask.sum()) - tp, int(total) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / total if total else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "instances": int(total),
            }
        )
    macro = {
        key: float(np.mean([item[key] for item in per_class]))
        for key in ("precision", "recall", "f1")
    }
    return macro, per_class


def summarize(frames, names, cfg, confidence=None):
    predictions, matches, targets = [], [], []
    for frame in frames:
        pred, correct = match_detections(
            frame["predictions"], frame["targets"], np.linspace(0.5, 0.95, 10)
        )
        predictions.append(pred)
        matches.append(correct)
        targets.extend(int(target[0]) for target in frame["targets"])
    pred = np.concatenate(predictions) if predictions else np.empty((0, 6))
    correct = np.concatenate(matches) if matches else np.empty((0, 10), bool)
    support = np.bincount(targets, minlength=len(names))
    if len(support) != len(names) or np.any(support == 0):
        raise ValueError("Every target class must have evaluation ground truth.")
    if confidence is None:

        def score(value):
            macro, classes = operating_metrics(pred, correct, support, value)
            ball = classes[cfg.ball_class_id]
            ratios = [
                item[key] / cfg.thresholds[key]
                for item in (macro, ball)
                for key in ("precision", "recall")
            ]
            return min(ratios), macro["f1"], -value

        confidence = max([i / 100 for i in range(1, 100)], key=score)
    macro, classes = operating_metrics(pred, correct, support, confidence)
    ap = np.zeros((len(names), 10))
    if len(pred):
        result = ap_per_class(correct, pred[:, 1], pred[:, 0], np.asarray(targets), plot=False)
        for row, cls in enumerate(result[6]):
            ap[int(cls)] = result[5][row]
    for cls, item in enumerate(classes):
        item.update(name=names[cls], ap50=float(ap[cls, 0]), ap50_95=float(ap[cls].mean()))
    macro.update(ap50=float(ap[:, 0].mean()), ap50_95=float(ap.mean()))
    return {
        "macro": macro,
        "ball": classes[cfg.ball_class_id],
        "per_class": classes,
        "confidence": float(confidence),
        "evaluator": "confidence-greedy-v1-ultralytics-ap",
        "label_provenance": "agent_labeled",
        "independent_ground_truth": False,
    }


def acceptance(metrics, cfg):
    ratios, failures = [], []
    for group in ("macro", "ball"):
        for key, threshold in cfg.thresholds.items():
            value = metrics[group][key]
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid metric: {group}.{key}")
            ratios.append(value / threshold)
            if value <= threshold:
                failures.append(f"{group}.{key}")
    if cfg.max_p95_ms is not None:
        latency = metrics.get("p95_ms")
        if latency is None or not np.isfinite(latency) or latency > cfg.max_p95_ms:
            failures.append("p95_ms")
    return {"feasible": not failures, "failures": failures, "progress": min(ratios)}


def evaluate(model, dataset, split, cfg, device, imgsz, confidence=None, *, image_paths=None):
    names = read_names(Path(dataset) / "data.yaml")
    if model.names != names:
        raise ValueError("Model classes differ from the evaluation taxonomy.")
    frames = []
    available = sorted(
        p
        for p in (Path(dataset) / "images" / split).iterdir()
        if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if image_paths is None:
        image_paths = available
    else:
        image_paths = [Path(p) for p in image_paths]
        allowed = {p.resolve() for p in available}
        if not image_paths or any(p.resolve() not in allowed for p in image_paths):
            raise ValueError("Evaluation subset must contain only images from its requested split.")
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Unreadable image: {path}")
        labels = read_labels(
            Path(dataset) / "labels" / split / f"{path.stem}.txt", image.shape, names
        )
        results = model.predict(
            image,
            imgsz=imgsz,
            conf=0.001,
            iou=cfg.nms_iou,
            max_det=cfg.max_det,
            agnostic_nms=False,
            augment=False,
            device=device,
            verbose=False,
        )[0]
        boxes = results.boxes
        frames.append(
            {
                "image": path.name,
                "predictions": [
                    [int(cls), float(score), *xyxy]
                    for cls, score, xyxy in zip(
                        boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()
                    )
                ],
                "targets": [[cls, *box] for cls, box in labels],
            }
        )
    metrics = summarize(frames, names, cfg, confidence)
    metrics["p95_ms"] = None
    if cfg.benchmark_frames:
        timings = []
        for index in range(cfg.warmup_frames + cfg.benchmark_frames):
            sample = cv2.imread(str(image_paths[index % len(image_paths)]))
            if device == "mps":
                torch.mps.synchronize()
            elif device != "cpu" and torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            model.predict(
                sample,
                imgsz=imgsz,
                conf=metrics["confidence"],
                iou=cfg.nms_iou,
                max_det=cfg.max_det,
                device=device,
                verbose=False,
            )
            if device == "mps":
                torch.mps.synchronize()
            elif device != "cpu" and torch.cuda.is_available():
                torch.cuda.synchronize()
            if index >= cfg.warmup_frames:
                timings.append((time.perf_counter() - start) * 1000)
        metrics["p95_ms"] = float(np.percentile(timings, 95))
        metrics["p50_ms"] = float(np.percentile(timings, 50))
    metrics.update(acceptance(metrics, cfg))
    return metrics, frames
