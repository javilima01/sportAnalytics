"""Resumable, budgeted experiments with an immutable benchmark and sealed test."""

import csv
import io
import json
import math
import sys
import time
from importlib.metadata import version
from pathlib import Path

from ..dataset import atomic_write
from .agent import ResearchAgent
from .config import Recipe
from .evaluation import acceptance
from .manifest import audit_dataset
from .providers import ProvidersUnavailable
from .runtime import file_hash, read_json, run_process, save_json, value_hash

ROOT = Path(__file__).resolve().parents[2]


class ConfirmationIncomplete(ValueError):
    """The recipe has not passed all required confirmation seeds."""


def initialize(cfg):
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.output_dir / "contract.json"
    value = cfg.model_dump(mode="json")
    if path.exists() and read_json(path) != value:
        raise ValueError("Campaign configuration changed. Use a new output directory/campaign ID.")
    save_json(path, value)


def freeze_dataset(cfg, *, allow_training_growth=False):
    from ..dataset import read_names

    if list(read_names(cfg.dataset_dir / "data.yaml").values()) != cfg.names:
        raise ValueError("Dataset taxonomy differs from the campaign.")
    snapshot = audit_dataset(cfg.dataset_dir, cfg.evaluation)
    path = cfg.output_dir / "snapshot.json"
    previous = read_json(path)
    records = [
        json.loads(line) for line in (cfg.dataset_dir / "manifest.jsonl").read_text().splitlines()
    ]
    versions = cfg.output_dir / "dataset_versions"
    yaml_hash = file_hash(cfg.dataset_dir / "data.yaml")
    if previous is not None and previous != snapshot:
        if (
            not allow_training_growth
            or not cfg.data_growth.enabled
            or (cfg.output_dir / "final_test.json").exists()
        ):
            raise ValueError("Frozen dataset changed. Use a new campaign.")
        old = read_json(versions / f"{previous['version']}.json")
        if not old or old["yaml_sha256"] != yaml_hash:
            raise ValueError("Frozen dataset metadata changed.")
        before = {r["image"]: r for r in old["records"]}
        after = {r["image"]: r for r in records}
        if any(after.get(key) != value for key, value in before.items()):
            raise ValueError("Existing frozen images or labels changed during training growth.")
        if any(r["split"] != "train" for key, r in after.items() if key not in before):
            raise ValueError("Validation and test data cannot grow after freezing.")
    save_json(
        versions / f"{snapshot['version']}.json",
        {"snapshot": snapshot, "yaml_sha256": yaml_hash, "records": records},
    )
    save_json(path, snapshot)
    return snapshot


def rank(record):
    metrics = record.get("metrics")
    if record["status"] != "completed" or not metrics:
        return (2, 0, 0, record["id"])
    if metrics["feasible"]:
        return (
            0,
            metrics["parameters"],
            metrics.get("p95_ms") or math.inf,
            metrics["checkpoint_bytes"],
            record["id"],
        )
    return (1, -metrics["progress"], -metrics["ball"]["ap50_95"], record["id"])


def write_results(directory, records):
    output = io.StringIO()
    fields = [
        "id",
        "phase",
        "recipe",
        "seed",
        "status",
        "seconds",
        "decision",
        "dataset_version",
        "parameters",
        "macro_ap50_95",
        "ball_ap50_95",
        "feasible",
        "error",
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for record in records:
        metrics = record.get("metrics", {})
        writer.writerow(
            {
                "id": record["id"],
                "phase": record["phase"],
                "recipe": record["recipe"]["id"],
                "seed": record["seed"],
                "status": record["status"],
                "seconds": record["seconds"],
                "decision": record.get("decision", "invalid"),
                "dataset_version": record.get("dataset_version"),
                "parameters": metrics.get("parameters"),
                "macro_ap50_95": metrics.get("macro", {}).get("ap50_95"),
                "ball_ap50_95": metrics.get("ball", {}).get("ap50_95"),
                "feasible": metrics.get("feasible"),
                "error": record.get("error", ""),
            }
        )
    atomic_write(Path(directory) / "results.csv", output.getvalue())


def source_fingerprint():
    paths = [ROOT / "main.py", ROOT / "requirements.txt", *sorted((ROOT / "src").rglob("*.py"))]
    result = {str(path.relative_to(ROOT)): file_hash(path) for path in paths}
    result["runtime"] = {
        "python": sys.version,
        **{name: version(name) for name in ("torch", "ultralytics", "numpy", "pydantic")},
    }
    return result


def load_state(cfg):
    state = read_json(cfg.output_dir / "state.json", {"trials": [], "incumbents": {}})
    # A crashed controller cannot know how long its child ran: charge its reserved deadline.
    for record in state["trials"]:
        if record["status"] == "running":
            record.update(
                status="interrupted",
                seconds=record["reserved_seconds"],
                error="Controller was interrupted; attempt is retained and not repeated.",
            )
    return state


def persist(cfg, state):
    save_json(cfg.output_dir / "state.json", state)
    write_results(cfg.output_dir, state["trials"])


def run_trial(cfg, state, recipe, phase, seed, timeout, executor=run_process):
    identity = f"trial-{len(state['trials']) + 1:04d}"
    folder = cfg.output_dir / identity
    folder.mkdir(parents=True, exist_ok=False)
    record = {
        "id": identity,
        "recipe": recipe.model_dump(),
        "phase": phase,
        "seed": seed,
        "status": "running",
        "seconds": 0,
        "reserved_seconds": timeout,
        "dataset_version": read_json(cfg.output_dir / "snapshot.json", {}).get("version"),
    }
    state["trials"].append(record)
    persist(cfg, state)
    print(
        f"[{phase}] {identity}: {recipe.id}, seed {seed}, limit {timeout / 60:g} min; {folder / 'run.log'}",
        flush=True,
    )
    reserve = min(cfg.budget.evaluation_reserve_seconds, timeout * 0.4)
    job = {
        "kind": "train",
        "campaign": cfg.model_dump(mode="json"),
        "recipe": recipe.model_dump(),
        "seed": seed,
        "folder": str(folder),
        "training_seconds": max(0.1, timeout - reserve),
    }
    save_json(folder / "job.json", job)
    save_json(
        folder / "provenance.json",
        {
            "source_hashes": source_fingerprint(),
            "python": sys.version,
            "checkpoint_sha256": file_hash(recipe.model),
            "dataset": read_json(cfg.output_dir / "snapshot.json"),
        },
    )
    started = time.monotonic()
    try:
        executor(
            [sys.executable, "-m", "src.research.worker", str(folder / "job.json")],
            timeout=timeout,
            log=folder / "run.log",
            cwd=ROOT,
        )
        metrics = read_json(folder / "metrics.json")
        if not isinstance(metrics, dict):
            raise ValueError("Worker produced no valid metrics.")
        metrics.update(acceptance(metrics, cfg.evaluation))
        if not isinstance(metrics["parameters"], int) or metrics["parameters"] <= 0:
            raise ValueError("Invalid parameter count.")
        if (
            not Path(metrics["checkpoint"]).is_file()
            or file_hash(metrics["checkpoint"]) != metrics["checkpoint_sha256"]
        ):
            raise ValueError("Checkpoint is missing or has changed.")
        record.update(status="completed", metrics=metrics)
        completed = [
            r
            for r in state["trials"]
            if r["phase"] == phase
            and r["status"] == "completed"
            and r.get("dataset_version") == record["dataset_version"]
        ]
        best = min(completed, key=rank)
        record["decision"] = "keep" if best["id"] == identity else "discard"
        state["incumbents"][phase] = best["id"]
    except TimeoutError as error:
        record.update(status="timeout", error=str(error), decision="invalid")
    except (RuntimeError, ValueError, KeyError, TypeError, OSError) as error:
        record.update(status="failed", error=str(error), decision="invalid")
    finally:
        if record["status"] == "running":
            record.update(status="interrupted", decision="invalid")
        record["seconds"] = time.monotonic() - started
        persist(cfg, state)
    print(
        f"[{phase}] {identity}: {record['status']} ({record.get('decision', 'invalid')})",
        flush=True,
    )
    return record


def run_campaign(
    cfg,
    phase="explore",
    executor=run_process,
    agent=None,
    retry_interrupted=False,
    phase_trial_limit=None,
):
    initialize(cfg)
    if (cfg.output_dir / "final_test.json").exists():
        raise ValueError("Final test has been opened; this campaign is closed to further trials.")
    if not (cfg.dataset_dir / "manifest.jsonl").exists():
        from .acquisition import acquire

        acquire(cfg, agent=agent)
    snapshot = freeze_dataset(cfg)
    fingerprint = source_fingerprint()
    previous = read_json(cfg.output_dir / "source_hashes.json")
    if previous is not None and previous != fingerprint:
        raise ValueError(
            "Experiment code changed; start a new campaign to keep comparisons reproducible."
        )
    save_json(cfg.output_dir / "source_hashes.json", fingerprint)
    state = load_state(cfg)
    if state.get("dataset_version") != snapshot["version"]:
        state.update(dataset_version=snapshot["version"], incumbents={})
    persist(cfg, state)
    agent = agent or ResearchAgent(cfg)
    if phase == "explore":
        queue = [(recipe, 0) for recipe in cfg.recipes]
        previous_exploration = [
            r for r in state["trials"] if r["phase"] == "explore" and r["status"] == "completed"
        ]
        if cfg.data_growth.enabled and previous_exploration:
            best_recipe = min(previous_exploration, key=rank)["recipe"]
            tested_models = {r["recipe"]["model"] for r in previous_exploration}
            queue.sort(
                key=lambda item: (
                    0
                    if item[0].model_dump() == best_recipe
                    else 1
                    if item[0].model not in tested_models
                    else 2
                )
            )
        minutes = cfg.budget.exploration_minutes
    else:
        parent_phase = (
            "promote" if phase == "confirm" and "promote" in state["incumbents"] else "explore"
        )
        parents = [
            r
            for r in state["trials"]
            if r["phase"] == parent_phase
            and r["status"] == "completed"
            and r.get("dataset_version") == snapshot["version"]
        ]
        if not parents:
            raise ValueError(f"No completed {parent_phase} candidate to promote.")
        recipe = Recipe.model_validate(min(parents, key=rank)["recipe"])
        if phase == "confirm":
            recipe.epochs = cfg.budget.confirmation_epochs
        queue = [(recipe, seed) for seed in ([0, 1, 2] if phase == "confirm" else [0])]
        minutes = (
            cfg.budget.confirmation_minutes if phase == "confirm" else cfg.budget.promotion_minutes
        )
    attempted = {
        (r["phase"], value_hash(r["recipe"]), r["seed"], r.get("dataset_version"))
        for r in state["trials"]
        if not (retry_interrupted and r["status"] == "interrupted")
    }
    queue = [
        (recipe, seed)
        for recipe, seed in queue
        if (phase, value_hash(recipe.model_dump()), seed, snapshot["version"]) not in attempted
    ]
    proposal_index = len(list((cfg.output_dir / "proposals").glob("*")))
    while len(state["trials"]) < cfg.budget.max_trials:
        if (
            phase_trial_limit is not None
            and sum(r["phase"] == phase for r in state["trials"]) >= phase_trial_limit
        ):
            break
        if (
            phase == "explore"
            and sum(r["phase"] == "explore" for r in state["trials"])
            >= cfg.budget.max_exploration_trials
        ):
            break
        remaining = cfg.budget.max_hours * 3600 - sum(r["seconds"] for r in state["trials"])
        if remaining < minutes * 60:
            break
        if not queue:
            if phase != "explore" or cfg.proposals != "codex":
                break
            proposal_index += 1
            proposal_dir = cfg.output_dir / "proposals" / f"proposal-{proposal_index:04d}"
            try:
                recipe = agent.propose(state["trials"], proposal_dir)
                if any(recipe.id == r["recipe"]["id"] for r in state["trials"]):
                    raise ValueError("Agent repeated a recipe ID.")
                queue = [(recipe, 0)]
            except ProvidersUnavailable:
                raise
            except (RuntimeError, ValueError, TimeoutError) as error:
                save_json(proposal_dir / "failure.json", {"error": str(error)})
                break
        recipe, seed = queue.pop(0)
        if not Path(recipe.model).is_file():
            folder = cfg.output_dir / "preparation" / recipe.id
            folder.mkdir(parents=True, exist_ok=True)
            save_json(
                folder / "job.json",
                {"kind": "prepare", "model": recipe.model, "folder": str(folder)},
            )
            executor(
                [sys.executable, "-m", "src.research.worker", str(folder / "job.json")],
                timeout=300,
                log=folder / "run.log",
                cwd=ROOT,
            )
            if not Path(recipe.model).is_file():
                raise FileNotFoundError(f"Could not prepare starting checkpoint: {recipe.model}")
        if source_fingerprint() != fingerprint:
            raise ValueError("Code changed during this campaign.")
        checkpoint_path = cfg.output_dir / "initial_checkpoints.json"
        checkpoints = read_json(checkpoint_path, {})
        digest = file_hash(recipe.model)
        if recipe.model in checkpoints and checkpoints[recipe.model] != digest:
            raise ValueError("Starting checkpoint changed during this campaign.")
        checkpoints[recipe.model] = digest
        save_json(checkpoint_path, checkpoints)
        freeze_dataset(cfg)
        run_trial(cfg, state, recipe, phase, seed, minutes * 60, executor=executor)
    return state


def finalize(cfg, executor=run_process):
    initialize(cfg)
    snapshot = freeze_dataset(cfg)
    if read_json(cfg.output_dir / "source_hashes.json") != source_fingerprint():
        raise ValueError("Experiment code changed; final testing requires the frozen evaluator.")
    marker = cfg.output_dir / "final_test.json"
    if marker.exists():
        result = read_json(marker)
        if result["status"] == "running":
            result["status"] = "interrupted"
            save_json(marker, result)
        return result
    state = load_state(cfg)
    candidates = {}
    for trial in state["trials"]:
        if (
            trial["phase"] == "confirm"
            and trial["status"] == "completed"
            and trial["metrics"]["feasible"]
            and trial.get("dataset_version") == snapshot["version"]
        ):
            candidates.setdefault(value_hash(trial["recipe"]), {})[trial["seed"]] = trial
    eligible = [seeds for seeds in candidates.values() if set(seeds) == {0, 1, 2}]
    if not eligible:
        raise ConfirmationIncomplete(
            "Final testing requires one recipe passing confirmation for seeds 0, 1, and 2."
        )
    trial = min(eligible, key=lambda seeds: rank(seeds[0]))[0]
    if file_hash(trial["metrics"]["checkpoint"]) != trial["metrics"]["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed.")
    folder = cfg.output_dir / "final"
    folder.mkdir(exist_ok=True)
    job = {
        "kind": "test",
        "campaign": cfg.model_dump(mode="json"),
        "folder": str(folder),
        "checkpoint": trial["metrics"]["checkpoint"],
        "imgsz": trial["recipe"]["imgsz"],
        "confidence": trial["metrics"]["confidence"],
    }
    save_json(folder / "job.json", job)
    result = {"status": "running", "trial": trial["id"], "independent_ground_truth": False}
    result.update(checkpoint=job["checkpoint"], imgsz=job["imgsz"], confidence=job["confidence"])
    save_json(marker, result)
    try:
        executor(
            [sys.executable, "-m", "src.research.worker", str(folder / "job.json")],
            timeout=cfg.budget.final_test_minutes * 60,
            log=folder / "run.log",
            cwd=ROOT,
        )
        metrics = read_json(folder / "metrics.json")
        if not isinstance(metrics, dict):
            raise ValueError("Worker produced no valid final-test metrics.")
        metrics.update(acceptance(metrics, cfg.evaluation))
        result.update(
            status="passed_against_agent_labels" if metrics["feasible"] else "failed",
            metrics=metrics,
        )
    except (RuntimeError, TimeoutError, ValueError, TypeError, KeyError, OSError) as error:
        result.update(status="failed", error=str(error))
    finally:
        if result["status"] == "running":
            result["status"] = "interrupted"
        save_json(marker, result)
    return result


def status(cfg):
    return {
        "campaign": cfg.campaign_id,
        "workflow": read_json(cfg.output_dir / "autonomous.json"),
        "providers": read_json(cfg.output_dir / "providers.json"),
        "acquisition": read_json(cfg.output_dir / "acquisition.json"),
        "experiments": read_json(cfg.output_dir / "state.json"),
        "final_test": read_json(cfg.output_dir / "final_test.json"),
    }
