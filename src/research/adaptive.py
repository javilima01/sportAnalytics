"""Agent-directed, resumable exploration with bounded diagnostic actions."""

import json
from typing import Literal

from pydantic import Field, model_validator

from .config import Recipe, Settings
from .controller import (
    freeze_dataset,
    load_state,
    rank,
    remaining_seconds,
    run_campaign,
    source_fingerprint,
)
from .diagnostics import diagnostic_allowance, run_diagnostic
from .growth import grow_training, validation_feedback
from .runtime import read_json, save_json, tree_bytes, value_hash


class ResearchDecision(Settings):
    action: Literal["train", "collect", "diagnose", "finish", "stop"]
    reason: str = Field(min_length=1)
    recipe: Recipe | None = None
    diagnostic: Literal["inspect_data", "evaluate_train", "overfit_crops"] | None = None
    trial_id: str | None = None
    queries: list[str] = Field(default_factory=list, max_length=3)

    @model_validator(mode="after")
    def payload(self):
        if (self.recipe is not None) != (self.action == "train"):
            raise ValueError("Only train actions require a recipe.")
        if (self.diagnostic is not None) != (self.action == "diagnose"):
            raise ValueError("Only diagnose actions require a diagnostic.")
        if bool(self.queries) != (self.action == "collect") or any(
            not q.strip() for q in self.queries
        ):
            raise ValueError("Only collect actions require 1–3 nonempty search queries.")
        if self.trial_id is not None and self.action != "diagnose":
            raise ValueError("trial_id applies only to a diagnostic.")
        return self


def recipe_key(recipe):
    return value_hash({k: v for k, v in recipe.items() if k not in ("id", "hypothesis")})


def can_train(cfg, state):
    return (
        sum(r["phase"] == "explore" for r in state["trials"]) < cfg.budget.max_exploration_trials
        and len(state["trials"]) < cfg.budget.max_trials
        and remaining_seconds(cfg, state) >= cfg.budget.exploration_minutes * 60
    )


def can_collect(cfg, workflow, state):
    acquisition = read_json(cfg.output_dir / "acquisition.json", {})
    return (
        cfg.data_growth.enabled
        and can_train(cfg, state)
        and workflow["data_rounds"] < cfg.data_growth.max_rounds
        and acquisition.get("bytes_downloaded", 0) < cfg.acquisition.download_gb * 1e9
        and tree_bytes(cfg.dataset_dir) < cfg.acquisition.storage_gb * 1e9
    )


def evidence(cfg, workflow, state, snapshot):
    history = []
    for r in state["trials"]:
        if r["phase"] != "explore":
            continue
        history.append(
            {
                "id": r["id"],
                "recipe": r["recipe"],
                "status": r["status"],
                "dataset_version": r.get("dataset_version"),
                "validation": {
                    k: r.get("metrics", {}).get(k)
                    for k in ("macro", "ball", "per_class", "feasible", "failures", "parameters")
                },
            }
        )
    diagnostics, images = [], []
    for path in sorted((cfg.output_dir / "diagnostics").glob("*/outcome.json")):
        provenance = read_json(path.parent / "provenance.json", {})
        diagnostics.append({"id": path.parent.name, **provenance, **read_json(path)})
        # Only the latest inspection on the current version supplies visual evidence.
        if provenance.get("dataset_version") == snapshot["version"]:
            crops = [path.parent / f"{s}-ball-crops.jpg" for s in ("train", "val")]
            if any(p.exists() for p in crops):
                images = [p for p in crops if p.exists()]
    # Code hashes belong in audit records, not in the model's limited context.
    for d in diagnostics:
        d.pop("source_hashes", None)
    outcomes = [read_json(p) for p in sorted((cfg.output_dir / "decisions").glob("*/outcome.json"))]
    payload = {
        "objective": "Smallest student passing every macro AND ball gate",
        "thresholds": cfg.evaluation.thresholds,
        "available_recipes": [r.model_dump() for r in cfg.recipes],
        "dataset_version": snapshot["version"],
        "counts": {s: snapshot["counts"][s] for s in ("train", "val")},
        "trials": history,
        "diagnostics": diagnostics,
        "previous_decisions": outcomes,
        "remaining": {
            "training_allowed": can_train(cfg, state),
            "collection_allowed": can_collect(cfg, workflow, state),
            "diagnostic_seconds_per_next_action": diagnostic_allowance(cfg),
            "decisions": cfg.diagnostics.max_decisions - workflow["decision_count"],
            "exploration_trials": cfg.budget.max_exploration_trials - len(history),
        },
    }
    return payload, images


def choose_action(cfg, agent, folder, payload, images):
    saved = folder / "decision.json"
    if saved.exists():
        return ResearchDecision.model_validate(read_json(saved))
    prompt = (
        "You direct football detection research. Choose the NEXT action from evidence, not a fixed queue. "
        "Return the structured decision. Do not execute tools or edit files; the controller executes your action. "
        "train: propose a complete recipe, including existing presets or changed resolution, augmentation, "
        "learning rate, optimizer, batch or epochs. Use only configured starting model paths and a unique id. "
        "No repeated recipe settings already completed/failed on this dataset version; changing only id does not count. "
        "Start with small students. Prefer one interpretable change; diagnose repeated zero ball scores before "
        "spending remaining trials on larger models or more similar frames. You need not exhaust the preset queue. "
        "diagnose: inspect_data measures object sizes/augmentation and supplies labeled train/val crop montages; "
        "evaluate_train checks a bounded sample of training predictions against validation; overfit_crops tests "
        "whether the training path can memorize enlarged TRAINING crops. A diagnostic can specify trial_id from "
        "completed current-version trials; otherwise the best available trial is used. inspect_data can run without one. "
        "Crop memorization is NOT generalization or a deployable model. Diagnose only unresolved questions; reuse "
        "recorded results. Annotation crops may reveal errors, but do not silently rewrite frozen labels. "
        "collect: provide 1–3 targeted video search queries to address evidence-backed training-data needs. "
        "finish: end exploration and promote the best current candidate; stop: end research with an explanation. "
        "Honor remaining budgets; if no useful action is affordable, finish or stop. All actions are bounded. "
        "The held-out test is sealed: never request its data. Do not lower gates or increase budgets. "
        "Among passing candidates fewer parameters win; additional accuracy does not justify a larger model. "
        "The teacher and students are separate. These are agent-generated labels, not independent ground truth. "
        "Use null for irrelevant recipe/diagnostic/trial_id fields and [] for irrelevant queries. Evidence:\n"
        + json.dumps(payload)
    )
    save_json(folder / "evidence.json", payload)
    decision = agent.request(
        prompt,
        ResearchDecision,
        folder / "agent",
        images=images,
        timeout=cfg.acquisition.agent_timeout_seconds,
    )
    save_json(saved, decision.model_dump(mode="json"))
    return decision


def explore_adaptively(
    cfg, workflow, record, wait, agent, executor, grower=grow_training, diagnoser=run_diagnostic
):
    workflow.setdefault("data_rounds", 0)
    workflow.setdefault("decision_count", 0)
    fingerprint = source_fingerprint()
    previous = read_json(cfg.output_dir / "source_hashes.json")
    if previous is not None and previous != fingerprint:
        raise ValueError("Experiment code changed; start a new campaign.")
    save_json(cfg.output_dir / "source_hashes.json", fingerprint)

    def finish(reason, stopped=False):
        if stopped:
            record("explore", reason, "stopped")
            return False
        workflow["completed_phases"].append("explore")
        record("explore", reason)
        return True

    while True:
        if source_fingerprint() != fingerprint:
            raise ValueError("Code changed during this campaign.")
        if workflow.get("growth_pending"):
            folder = cfg.output_dir / "data_growth" / workflow["growth_pending"]
            feedback = validation_feedback(cfg, load_state(cfg)["trials"])
            try:
                result = wait(lambda: grower(cfg, agent, folder, feedback))
            except (RuntimeError, TimeoutError, ValueError) as error:
                # Provider wait exhaustion is handled by the autonomous controller.
                from .providers import ProviderWaitExhausted

                if isinstance(error, ProviderWaitExhausted):
                    raise
                freeze_dataset(cfg, allow_training_growth=True)
                result = {"added_images": 0, "error": str(error)}
                save_json(folder / "failure.json", result)
            workflow.pop("growth_pending")
            if workflow.get("decision_pending"):
                save_json(
                    cfg.output_dir
                    / "decisions"
                    / workflow.pop("decision_pending")
                    / "outcome.json",
                    {"action": "collect", "result": result},
                )
            record("acquire_train", f"Added {result['added_images']} training images.")

        snapshot = freeze_dataset(cfg)
        state = load_state(cfg)
        current = [
            r
            for r in state["trials"]
            if r["phase"] == "explore"
            and r["status"] == "completed"
            and r.get("dataset_version") == snapshot["version"]
        ]
        needs_minimum = (
            cfg.data_growth.enabled
            and snapshot["counts"]["train"]["images"] < cfg.data_growth.min_train_images
        )
        if needs_minimum:
            if not can_collect(cfg, workflow, state):
                return finish("Training-image minimum unmet within available budgets.", True)
            workflow["data_rounds"] += 1
            workflow["growth_pending"] = f"round-{workflow['data_rounds']:03d}"
            record(
                "acquire_train", "Collecting the minimum training set before diagnostics/trials."
            )
            continue

        initial = cfg.output_dir / "diagnostics/initial"
        if (
            workflow["decision_count"] == 0
            and not (initial / "outcome.json").exists()
            and (
                diagnostic_allowance(cfg) > 0
                or any(
                    r["directory"] == str(initial)
                    for r in read_json(cfg.output_dir / "diagnostics/state.json", [])
                )
            )
        ):
            record(
                "diagnose",
                "Inspecting labels, effective ball sizes and augmentation before experiments.",
            )
            diagnoser(cfg, "inspect_data", None, initial, executor=executor)
            continue
        if workflow["decision_count"] >= cfg.diagnostics.max_decisions and not workflow.get(
            "decision_pending"
        ):
            return finish("Research decision limit reached.", not bool(current))
        if not workflow.get("decision_pending"):
            workflow["decision_count"] += 1
            workflow["decision_pending"] = f"decision-{workflow['decision_count']:03d}"
            record(
                "decide",
                "Agent is choosing the next diagnostic, training recipe or data collection.",
            )
        folder = cfg.output_dir / "decisions" / workflow["decision_pending"]
        outcome = folder / "outcome.json"
        if outcome.exists():
            recovered = read_json(outcome)
            workflow.pop("decision_pending")
            record("decide", "Recovered a completed decision.")
            if recovered["action"] in ("finish", "stop"):
                return finish(recovered["reason"], recovered["action"] == "stop")
            continue
        payload, images = evidence(cfg, workflow, state, snapshot)
        try:
            decision = wait(lambda: choose_action(cfg, agent, folder, payload, images))
            result = {"action": decision.action, "reason": decision.reason}
            record("decide", f"{decision.action}: {decision.reason}")
            if decision.action == "train":
                effect = read_json(folder / "effect.json")
                if effect and len(state["trials"]) > effect["first_trial_index"]:
                    result["trials"] = [
                        r["id"] for r in state["trials"][effect["first_trial_index"] :]
                    ]
                else:
                    if not can_train(cfg, state):
                        raise ValueError("No exploration trial budget remains.")
                    recipe = decision.recipe
                    if recipe.model not in {r.model for r in cfg.recipes}:
                        raise ValueError(
                            "Recipe checkpoint must be from the configured search space."
                        )
                    for trial in state["trials"]:
                        if trial["recipe"]["id"] == recipe.id:
                            raise ValueError("Recipe id already used; choose a unique id.")
                        if (
                            trial["status"] != "interrupted"
                            and trial.get("dataset_version") == snapshot["version"]
                            and recipe_key(trial["recipe"]) == recipe_key(recipe.model_dump())
                        ):
                            raise ValueError(
                                "These recipe settings were already attempted on this dataset version."
                            )
                    save_json(folder / "effect.json", {"first_trial_index": len(state["trials"])})
                    updated = run_campaign(
                        cfg, executor=executor, agent=agent, recipes_override=[recipe]
                    )
                    result["trials"] = [r["id"] for r in updated["trials"][len(state["trials"]) :]]
            elif decision.action == "collect":
                if not can_collect(cfg, workflow, state):
                    raise ValueError("Training-data collection budget is unavailable.")
                workflow["data_rounds"] += 1
                workflow["growth_pending"] = f"round-{workflow['data_rounds']:03d}"
                save_json(
                    cfg.output_dir / "data_growth" / workflow["growth_pending"] / "plan.json",
                    {"reason": decision.reason, "queries": decision.queries},
                )
                record("acquire_train", "Executing the agent's targeted search plan.")
                continue
            elif decision.action == "diagnose":
                trial = (
                    next((r for r in current if r["id"] == decision.trial_id), None)
                    if decision.trial_id
                    else (min(current, key=rank) if current else None)
                )
                if decision.trial_id and trial is None:
                    raise ValueError(
                        "Diagnostic trial must be completed on the current dataset version."
                    )
                if trial is None and decision.diagnostic != "inspect_data":
                    raise ValueError("This diagnostic requires a trained checkpoint.")
                result["result"] = diagnoser(
                    cfg,
                    decision.diagnostic,
                    trial,
                    cfg.output_dir / "diagnostics" / workflow["decision_pending"],
                    executor=executor,
                )
            elif decision.action == "finish" and not current:
                raise ValueError(
                    "No completed candidate on the current dataset version; train or stop."
                )
        except (RuntimeError, TimeoutError, ValueError) as error:
            from .providers import ProviderWaitExhausted

            if isinstance(error, ProviderWaitExhausted):
                raise
            result = {"action": "rejected", "reason": str(error)}
            record("decide", f"Decision could not be executed: {error}")
        save_json(outcome, result)
        workflow.pop("decision_pending")
        record("decide", "Decision and outcome saved.")
        if result["action"] in ("finish", "stop"):
            return finish(result["reason"], result["action"] == "stop")
