"""Independent image-based relabeling with recoverable, versioned label transactions."""

import json
import time
from pathlib import Path

import cv2

from ..dataset import atomic_write, read_labels, write_labels
from .agent import Review
from .controller import freeze_dataset
from .manifest import match_key
from .runtime import file_hash, read_json, save_json


def commit_review(cfg, directory):
    journal_path = directory / "revision.json"
    journal = read_json(journal_path)
    manifest = cfg.dataset_dir / "manifest.jsonl"
    current = [json.loads(line) for line in manifest.read_text().splitlines()]
    if current not in (journal["before_records"], journal["after_records"]):
        raise ValueError("Dataset changed outside the pending label review.")
    for change in journal["changes"]:
        target = cfg.dataset_dir / change["label"]
        if target.read_text() not in (change["before"], change["after"]):
            raise ValueError("A pending review label was modified externally.")
    # The complete before/after transaction is durable before any live label changes.
    for change in journal["changes"]:
        atomic_write(cfg.dataset_dir / change["label"], change["after"])
    atomic_write(manifest, "".join(json.dumps(r) + "\n" for r in journal["after_records"]))
    return freeze_dataset(cfg, label_revision=journal_path)


def enough_coverage(cfg, records, changes):
    replacements = {c["label"]: c["after"] for c in changes}
    for split in ("train", "val", "test"):
        rows = [r for r in records if r["split"] == split]
        counts = [0] * len(cfg.names)
        for row in rows:
            text = replacements.get(row["label"])
            if text is None:
                text = (cfg.dataset_dir / row["label"]).read_text()
            for line in text.splitlines():
                if line.strip():
                    counts[int(line.split()[0])] += 1
        if not rows or not all(counts):
            return False
        if split != "train" and (
            len({match_key(r["match_id"]) for r in rows}) < cfg.evaluation.min_matches
            or counts[cfg.evaluation.ball_class_id] < cfg.evaluation.min_ball_instances
        ):
            return False
    return True


def review_labels(cfg, agent, directory, plan):
    """No student predictions are provided to the labeling agent, including for test data."""
    if not (cfg.autonomy.enabled and cfg.autonomy.allow_label_review and cfg.data_growth.enabled):
        raise ValueError("Autonomous label review is disabled.")
    if (cfg.output_dir / "final_test.json").exists():
        raise ValueError("Labels cannot change after final testing.")
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "result.json").exists():
        freeze_dataset(cfg)
        return read_json(directory / "result.json")
    if (directory / "revision.json").exists():
        snapshot = commit_review(cfg, directory)
        result = {**read_json(directory / "summary.json"), "dataset_version": snapshot["version"]}
        save_json(directory / "result.json", result)
        return result
    freeze_dataset(cfg)
    split = plan["split"]
    if split not in ("train", "val", "test"):
        raise ValueError("Unknown label-review split.")
    manifest = cfg.dataset_dir / "manifest.jsonl"
    before = [json.loads(line) for line in manifest.read_text().splitlines()]
    eligible = [r for r in before if r["split"] == split]
    names = plan.get("images", [])
    if names:
        if set(names) - {Path(r["image"]).name for r in eligible}:
            raise ValueError("Review images must belong to the requested split.")
        eligible = [r for r in eligible if Path(r["image"]).name in names]
    eligible.sort(key=lambda r: ("/label_reviews/" in r.get("agent_record", ""), r["image"]))
    after = {r["image"]: r.copy() for r in before}
    changes, rejected, reviewed = [], 0, 0
    deadline = time.monotonic() + cfg.acquisition.minutes * 60
    for record in eligible[: cfg.diagnostics.max_images]:
        if time.monotonic() >= deadline:
            break
        folder = directory / "images" / Path(record["image"]).stem
        result_path = folder / "validated.json"
        if result_path.exists():
            review = Review.model_validate(read_json(result_path))
        else:
            path = cfg.dataset_dir / record["image"]
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Unreadable review image: {path}")
            h, w = image.shape[:2]
            proposals = [
                {"class_name": cfg.names[c], "xyxy": [a / w, b / h, x / w, y / h]}
                for c, (a, b, x, y) in read_labels(
                    cfg.dataset_dir / record["label"], image.shape, dict(enumerate(cfg.names))
                )
            ]
            review = agent.label(
                path,
                proposals,
                folder,
                min(cfg.acquisition.agent_timeout_seconds, deadline - time.monotonic()),
            )
            if any(b.class_id >= len(cfg.names) for b in review.boxes):
                raise ValueError("Reviewer returned an unknown class.")
            save_json(result_path, review.model_dump())
        reviewed += 1
        if review.status != "accepted":
            rejected += 1
            continue  # Keep difficult/ambiguous examples; do not drop them to improve scores.
        staged = folder / "labels.txt"
        write_labels(
            staged,
            [
                (b.class_id, (b.x1 * 1000, b.y1 * 1000, b.x2 * 1000, b.y2 * 1000))
                for b in review.boxes
            ],
            (1000, 1000),
        )
        changes.append(
            {
                "label": record["label"],
                "before": (cfg.dataset_dir / record["label"]).read_text(),
                "after": staged.read_text(),
            }
        )
        after[record["image"]].update(
            label_sha256=file_hash(staged),
            agent_record=str(folder),
            agent_sha256=file_hash(result_path),
        )
    summary = {
        "split": split,
        "reviewed_images": reviewed,
        "rejected_retained": rejected,
        "changed_labels": sum(c["before"] != c["after"] for c in changes),
    }
    if not enough_coverage(cfg, list(after.values()), changes):
        result = {
            **summary,
            "status": "needs_more_data",
            "changed_labels": 0,
            "reason": "Collect more class coverage in this split, then retry this saved review.",
        }
        save_json(directory / "result.json", result)
        return result
    save_json(directory / "summary.json", summary)
    save_json(
        directory / "revision.json",
        {
            "before_records": before,
            "after_records": list(after.values()),
            "changes": changes,
            "mutable_fields": ["label_sha256", "agent_record", "agent_sha256"],
        },
    )
    snapshot = commit_review(cfg, directory)
    result = {**summary, "status": "completed", "dataset_version": snapshot["version"]}
    save_json(directory / "result.json", result)
    return result
