"""Bounded, resumable training-data collection guided only by validation evidence."""

import json
import time

from pydantic import Field

from .acquisition import acquire, discover
from .config import Settings, Source
from .controller import freeze_dataset, load_state, rank, run_campaign
from .providers import ProviderWaitExhausted
from .runtime import read_json, save_json, tree_bytes


class DataSearch(Settings):
    reason: str = Field(min_length=1)
    queries: list[str] = Field(min_length=1, max_length=3)


def validation_feedback(cfg, trials):
    # Explicit allowlist: never send test records, predictions, or artifact paths.
    return {
        "thresholds": cfg.evaluation.thresholds,
        "training_images": read_json(cfg.output_dir / "snapshot.json")["counts"]["train"]["images"],
        "trials": [
            {
                "id": r["id"],
                "recipe": r["recipe"],
                "dataset_version": r.get("dataset_version"),
                "validation": {
                    key: r["metrics"].get(key)
                    for key in ("macro", "ball", "per_class", "failures", "parameters", "feasible")
                },
            }
            for r in trials
            if r["status"] == "completed" and r["phase"] == "explore"
        ],
    }


def grow_training(cfg, agent, directory, feedback):
    if (cfg.output_dir / "final_test.json").exists():
        raise ValueError("Final test has been opened; training data cannot grow.")
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "result.json").exists():
        freeze_dataset(cfg)
        return read_json(directory / "result.json")
    if not (directory / "before.json").exists():
        save_json(directory / "before.json", freeze_dataset(cfg))
        save_json(directory / "feedback.json", feedback)
    deadline = time.monotonic() + cfg.acquisition.minutes * 60
    plan = read_json(directory / "plan.json")
    if plan is None:
        prompt = (
            "Plan new YouTube searches for TRAINING data for football player/ball detection. "
            "Use the supplied validation failures to form data-collection hypotheses, e.g. "
            "distant balls or crowded gameplay. Do not claim aggregate metrics prove a cause. "
            "With no trials, expand match and camera diversity. Prioritize helping small "
            "students pass every quality gate. No human labels are available. Never request "
            "changes to validation/test data or accuracy thresholds. Return a reason and "
            "1–3 search queries. No tools or file edits. Evidence follows:\n"
            + json.dumps(
                {
                    "feedback": read_json(directory / "feedback.json"),
                    "prior_searches": [
                        read_json(p) for p in sorted(directory.parent.glob("*/plan.json"))
                    ],
                }
            )
        )
        plan = agent.request(
            prompt,
            DataSearch,
            directory / "planning",
            timeout=min(cfg.acquisition.agent_timeout_seconds, deadline - time.monotonic()),
        ).model_dump()
        save_json(directory / "plan.json", plan)
    selected = read_json(directory / "sources.json")
    if selected is None:
        existing = read_json(cfg.output_dir / "acquisition.json", {}).get("sources", [])
        sources = discover(
            cfg,
            agent,
            deadline,
            folder=directory / "discovery",
            queries=plan["queries"],
            training_only=True,
            existing=existing,
            limit=cfg.data_growth.sources_per_round,
        )
        selected = [source.model_dump() for source in sources]
        save_json(directory / "sources.json", selected)
    if selected:
        acquire(
            cfg,
            agent,
            training_sources=[Source.model_validate(s) for s in selected],
            deadline=deadline,
        )
    snapshot = freeze_dataset(cfg, allow_training_growth=True)
    before = read_json(directory / "before.json")
    result = {
        "reason": plan["reason"],
        "added_images": snapshot["counts"]["train"]["images"] - before["counts"]["train"]["images"],
        "dataset_version": snapshot["version"],
        "sources": len(selected),
    }
    save_json(directory / "result.json", result)
    return result


def explore_with_growth(cfg, workflow, record, wait, agent, executor, grower=grow_training):
    """Alternate collection with small trial batches; all trials share global budgets."""
    workflow.setdefault("data_rounds", 0)
    while True:
        if workflow.get("growth_pending"):
            directory = cfg.output_dir / "data_growth" / workflow["growth_pending"]
            feedback = validation_feedback(cfg, load_state(cfg)["trials"])
            try:
                result = wait(lambda: grower(cfg, agent, directory, feedback))
            except ProviderWaitExhausted:
                raise
            except (RuntimeError, TimeoutError, ValueError) as error:
                # A failed search/download consumes this data round, not the entire campaign.
                # Audit any partial append first; corruption still propagates and stops work.
                freeze_dataset(cfg, allow_training_growth=True)
                save_json(directory / "failure.json", {"error": str(error)})
                result = {"added_images": 0}
                record("acquire_train", f"Data round failed: {error}. Retaining accepted images.")
            workflow.pop("growth_pending")
            workflow.pop("assess_data", None)
            record(
                "acquire_train",
                f"Added {result['added_images']} training images; validation/test unchanged.",
            )

        snapshot = freeze_dataset(cfg)
        state = load_state(cfg)
        explored = [r for r in state["trials"] if r["phase"] == "explore"]
        current = [
            r
            for r in explored
            if r["status"] == "completed" and r.get("dataset_version") == snapshot["version"]
        ]
        needs_minimum = snapshot["counts"]["train"]["images"] < cfg.data_growth.min_train_images
        needs_quality = False
        if current:
            best = min(current, key=rank)
            needs_quality = not best["metrics"]["feasible"] or best["metrics"]["parameters"] > min(
                r["metrics"]["parameters"] for r in current
            )
        trial_room = (
            len(explored) < cfg.budget.max_exploration_trials
            and len(state["trials"]) < cfg.budget.max_trials
            and cfg.budget.max_hours * 3600 - sum(r["seconds"] for r in state["trials"])
            >= cfg.budget.exploration_minutes * 60
        )
        acquisition = read_json(cfg.output_dir / "acquisition.json", {})
        data_room = (
            workflow["data_rounds"] < cfg.data_growth.max_rounds
            and acquisition.get("bytes_downloaded", 0) < cfg.acquisition.download_gb * 1e9
            and tree_bytes(cfg.dataset_dir) < cfg.acquisition.storage_gb * 1e9
        )
        if needs_minimum or (workflow.get("assess_data") and needs_quality and trial_room):
            if data_room and trial_room:
                workflow["data_rounds"] += 1
                workflow["growth_pending"] = f"round-{workflow['data_rounds']:03d}"
                record(
                    "acquire_train",
                    "Searching and labeling new training matches to improve coverage/validation quality.",
                )
                continue
            if needs_minimum:
                record(
                    "acquire_train",
                    "Training-image minimum remains unmet within the available data/trial budgets.",
                    "stopped",
                )
                return False
        workflow.pop("assess_data", None)
        if not trial_room:
            break
        if "explore_target" not in workflow:
            workflow["explore_target"] = min(
                len(explored) + cfg.data_growth.trials_per_round, cfg.budget.max_exploration_trials
            )
        record("explore", "Running a short trial batch before reassessing training-data needs.")
        state = wait(
            lambda: run_campaign(
                cfg,
                executor=executor,
                agent=agent,
                retry_interrupted=True,
                phase_trial_limit=workflow["explore_target"],
            )
        )
        workflow.pop("explore_target")
        workflow["assess_data"] = True
        record("explore", "Trial batch saved; assessing validation failures and remaining budgets.")
        if len(state["trials"]) == len(explored):
            break
    if not current:
        record(
            "explore",
            "No completed trial on the current training dataset within the trial budget.",
            "stopped",
        )
        return False
    workflow["completed_phases"].append("explore")
    record(
        "explore", "Exploration complete. Training data are fixed for promotion and confirmation."
    )
    return True
