"""Agent-directed, resumable exploration with bounded diagnostic actions."""

import json
from typing import Literal

from pydantic import Field, model_validator

from .allowances import (
    BudgetUpdate,
    budget_ceilings,
    budget_values,
    change_budget,
    effective_campaign,
    trial_minutes,
)
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
from .label_review import review_labels
from .runtime import file_hash, read_json, save_json, tree_bytes, value_hash


class ResearchDecision(Settings):
    action: Literal[
        "train", "collect", "review_labels", "adjust_budget", "diagnose", "finish", "stop"
    ]
    reason: str = Field(min_length=1)
    recipe: Recipe | None = None
    diagnostic: (
        Literal["inspect_data", "evaluate_train", "evaluate_train_full", "overfit_crops"] | None
    ) = None
    trial_id: str | None = None
    queries: list[str] = Field(default_factory=list, max_length=3)
    split: Literal["train", "val", "test"] | None = None
    images: list[str] = Field(default_factory=list, max_length=64)
    minutes: float | None = Field(None, gt=0, le=1440)
    budget_update: BudgetUpdate | None = None

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
        if self.split is not None and self.action not in ("collect", "review_labels"):
            raise ValueError("split applies only to collection or label review.")
        if self.action == "review_labels" and self.split is None:
            raise ValueError("Label review requires a split.")
        if self.images and self.action != "review_labels":
            raise ValueError("Image selection applies only to label review.")
        if self.minutes is not None and self.action != "train":
            raise ValueError("minutes applies only to a training action.")
        if (self.budget_update is not None) != (self.action == "adjust_budget"):
            raise ValueError("Only adjust_budget actions require a budget_update.")
        return self


def recipe_key(recipe):
    return value_hash({k: v for k, v in recipe.items() if k not in ("id", "hypothesis")})


def can_train(cfg, state, minutes=None):
    return (
        sum(r["phase"] == "explore" for r in state["trials"]) < cfg.budget.max_exploration_trials
        and len(state["trials"]) < cfg.budget.max_trials
        and remaining_seconds(cfg, state) >= trial_minutes(cfg, minutes) * 60
    )


def can_collect(cfg, workflow, state):
    acquisition = read_json(cfg.output_dir / "acquisition.json", {})
    return (
        cfg.data_growth.enabled
        and workflow["data_rounds"] < cfg.data_growth.max_rounds
        and acquisition.get("bytes_downloaded", 0) < cfg.acquisition.download_gb * 1e9
        and tree_bytes(cfg.dataset_dir) < cfg.acquisition.storage_gb * 1e9
    )


def prior_trials(cfg):
    if cfg.prior_campaign is None:
        return []
    return [
        {**r, "id": "prior/" + r["id"]}
        for r in read_json(cfg.prior_campaign / "state.json", {}).get("trials", [])
    ]


def trial_context(record):
    return {
        "id": record["id"],
        "phase": record["phase"],
        "recipe": record["recipe"],
        "status": record["status"],
        "dataset_version": record.get("dataset_version"),
        "error": record.get("error"),
        "metrics": {
            k: record.get("metrics", {}).get(k)
            for k in (
                "macro",
                "ball",
                "feasible",
                "training",
                "epochs_completed",
                "runtime_seconds",
            )
        },
    }


def evidence(cfg, workflow, state, snapshot, base_cfg=None):
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
                "benchmark_version": r.get("benchmark_version"),
                "training": r.get("metrics", {}).get(
                    "training",
                    {
                        "completed_epochs": r.get("metrics", {}).get("epochs_completed"),
                        "runtime_seconds": r.get("metrics", {}).get("runtime_seconds"),
                    },
                ),
                "localization": r.get("metrics", {}).get("localization"),
                "minutes_allowed": r.get("reserved_seconds", 0) / 60,
                "validation": {
                    k: r.get("metrics", {}).get(k)
                    for k in ("macro", "ball", "per_class", "feasible", "failures", "parameters")
                },
            }
        )
    diagnostics, images = [], []
    for path in sorted(
        (cfg.output_dir / "diagnostics").glob("*/outcome.json"),
        key=lambda p: (p.parent.name != "initial", p.parent.name),
    ):
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
        "benchmark_version": snapshot.get("benchmark_version"),
        "counts": {s: snapshot["counts"][s] for s in ("train", "val")},
        "test_coverage": snapshot["counts"]["test"],
        "autonomy": cfg.autonomy.model_dump(),
        "working_budgets": budget_values(cfg),
        "budget_ceilings": budget_ceilings(base_cfg or cfg),
        "warm_start_models": [
            {
                "trial_id": r["id"],
                "model": r["metrics"]["checkpoint"],
                "dataset_version": r.get("dataset_version"),
            }
            for r in [*state["trials"], *prior_trials(cfg)]
            if r["status"] == "completed"
        ]
        if cfg.autonomy.enabled
        else [],
        "trials": history,
        "later_phase_trials": [
            trial_context(r) for r in state["trials"] if r["phase"] != "explore"
        ],
        "prior_campaign_trials": [trial_context(r) for r in prior_trials(cfg)],
        "diagnostics": diagnostics,
        "previous_decisions": outcomes,
        "remaining": {
            "training_allowed": can_train(cfg, state),
            "collection_allowed": can_collect(cfg, workflow, state),
            "diagnostic_seconds_per_next_action": diagnostic_allowance(cfg),
            "decisions": cfg.diagnostics.max_decisions - workflow["decision_count"],
            "exploration_trials": cfg.budget.max_exploration_trials - len(history),
            "experiment_hours": remaining_seconds(cfg, state) / 3600,
            "data_rounds": cfg.data_growth.max_rounds - workflow["data_rounds"],
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
        "Set minutes to request a longer or shorter training run within the time ceiling. You may also choose a "
        "registered warm_start_models checkpoint as recipe.model to continue learning from previous weights "
        "with a fresh optimizer/schedule. Repeating identical settings needs a different time allowance or dataset. "
        "Prior-campaign trials are historical context, not acceptance evidence under the new code/data. "
        "Use their lessons and warm-start checkpoints instead of blindly repeating their failed baselines. "
        "Later-phase failures are returned to you for corrective experiments or budget adjustments. "
        "Start with small students, but a larger-model control can diagnose unresolved capacity limits; "
        "do not rule out capacity from a tiny training sample or a crop memorization test. "
        "diagnose: inspect_data measures object sizes/augmentation and supplies labeled train/val crop montages; "
        "evaluate_train checks a targeted, potentially optimistic sample; evaluate_train_full checks ALL training "
        "images within the diagnostic deadline. Use the full check before declaring training fit sufficient. "
        "Compare metrics at the SAME confidence. Training evidence includes actual epochs, loss and LR history. "
        "A time-limited or early-stopped run is not proof of convergence. Localization evidence shows near misses "
        "below IoU 0.50; zero AP does not imply zero improvement. overfit_crops tests "
        "whether the training path can memorize enlarged TRAINING crops. A diagnostic can specify trial_id from "
        "any completed trial, including prior/ IDs; otherwise the best current trial is used. Diagnostics use current "
        "training labels; recorded validation metrics may belong to older versions. inspect_data can run without a trial. "
        "Crop memorization is NOT generalization or a deployable model. Diagnose only unresolved questions; reuse "
        "recorded results. No human will label or curate this dataset. You own automatic data improvements. "
        "collect: choose split=train, val or test and provide 1–3 targeted video search queries. Expand evaluation "
        "coverage across independent matches when it is too small (aim for at least 3 matches and 50 ball labels "
        "per evaluation split before drawing strong conclusions). Test coverage counts are available, but student "
        "test predictions/metrics remain sealed until final testing. Use coverage/diversity, not test performance, "
        "to design test acquisition. review_labels: choose a split and optional image filenames (empty=next bounded "
        "batch). An independent labeling call inspects original pixels and existing labels without student "
        "predictions; corrected labels and original labels are journaled. Uncertain images are retained. "
        "All split/label changes create a dataset version; benchmark changes invalidate previous acceptance "
        "claims. Train on the new version before promotion. Never move held-out matches into training. "
        "adjust_budget: request explicit increases in budget_update within budget_ceilings when justified by "
        "evidence, including more trials, hours, collection rounds, download/storage allowance or diagnostics. "
        "These changes persist across restarts without editing YAML. Extend decision limits before using the last decision. "
        "finish: end exploration and promote the best current candidate; stop: end research with an explanation. "
        "Check autonomy flags before requesting extensions or dataset review/growth. Before stopping, consider "
        "full training fit, automatic label review, new evaluation coverage, longer optimization or a controlled "
        "capacity probe. Do not stop merely because the initial budget was small; extend it if useful work remains "
        "within the user ceilings. Stop with an evidence-based explanation when no useful authorized action remains. "
        "Do not lower quality gates or change the user ceilings. "
        "Among passing candidates fewer parameters win; additional accuracy does not justify a larger model. "
        "The teacher and students are separate. These are agent-generated labels, not independent ground truth. "
        "Use null for irrelevant optional fields and [] for irrelevant lists. Evidence:\n"
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
    base_cfg = cfg
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
        cfg = effective_campaign(base_cfg)
        if source_fingerprint() != fingerprint:
            raise ValueError("Code changed during this campaign.")
        if workflow.get("review_pending"):
            folder = cfg.output_dir / "label_reviews" / workflow["review_pending"]
            plan = read_json(folder / "plan.json")
            try:
                result = wait(lambda: review_labels(cfg, agent, folder, plan))
            except (RuntimeError, TimeoutError, ValueError) as error:
                from .providers import ProviderWaitExhausted

                if isinstance(error, ProviderWaitExhausted) or (folder / "revision.json").exists():
                    raise
                freeze_dataset(cfg)
                result = {"status": "failed", "reason": str(error)}
                save_json(folder / "result.json", result)
            workflow.pop("review_pending")
            save_json(
                cfg.output_dir / "decisions" / workflow.pop("decision_pending") / "outcome.json",
                {"action": "review_labels", "result": result},
            )
            record(
                "review_labels",
                f"Reviewed {result.get('reviewed_images', 0)} {plan['split']} images; "
                f"changed {result.get('changed_labels', 0)} label files.",
            )
        if workflow.get("growth_pending"):
            folder = cfg.output_dir / "data_growth" / workflow["growth_pending"]
            split = read_json(folder / "plan.json", {}).get("split", "train")
            feedback = validation_feedback(cfg, load_state(cfg)["trials"])
            try:
                result = wait(lambda: grower(cfg, agent, folder, feedback))
            except (RuntimeError, TimeoutError, ValueError) as error:
                # Provider wait exhaustion is handled by the autonomous controller.
                from .providers import ProviderWaitExhausted

                if isinstance(error, ProviderWaitExhausted):
                    raise
                freeze_dataset(cfg, allow_growth_splits={split})
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
            record("acquire_data", f"Added {result['added_images']} {split} images.")

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
            if not can_collect(cfg, workflow, state) and not cfg.autonomy.enabled:
                return finish("Training-image minimum unmet within available budgets.", True)
            if can_collect(cfg, workflow, state):
                workflow["data_rounds"] += 1
                workflow["growth_pending"] = f"round-{workflow['data_rounds']:03d}"
                record(
                    "acquire_train",
                    "Collecting the minimum training set before diagnostics/trials.",
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
        payload, images = evidence(cfg, workflow, state, snapshot, base_cfg)
        try:
            decision = wait(lambda: choose_action(cfg, agent, folder, payload, images))
            result = {"action": decision.action, "reason": decision.reason}
            record("decide", f"{decision.action}: {decision.reason}")
            if decision.action == "train":
                minutes = trial_minutes(cfg, decision.minutes)
                effect = read_json(folder / "effect.json")
                if effect and len(state["trials"]) > effect["first_trial_index"]:
                    result["trials"] = [
                        r["id"] for r in state["trials"][effect["first_trial_index"] :]
                    ]
                else:
                    if not can_train(cfg, state, minutes):
                        raise ValueError("No exploration trial budget remains.")
                    recipe = decision.recipe
                    allowed_models = {r.model for r in cfg.recipes}
                    if cfg.autonomy.enabled:
                        for prior in [*state["trials"], *prior_trials(cfg)]:
                            if (
                                prior["status"] == "completed"
                                and recipe.model == prior["metrics"]["checkpoint"]
                            ):
                                if file_hash(recipe.model) != prior["metrics"]["checkpoint_sha256"]:
                                    raise ValueError("Warm-start checkpoint changed.")
                                allowed_models.add(recipe.model)
                    if recipe.model not in allowed_models:
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
                            and trial.get("reserved_seconds") == minutes * 60
                        ):
                            raise ValueError(
                                "These recipe settings were already attempted on this dataset version."
                            )
                    save_json(folder / "effect.json", {"first_trial_index": len(state["trials"])})
                    updated = run_campaign(
                        base_cfg,
                        executor=executor,
                        agent=agent,
                        recipes_override=[recipe],
                        minutes_override=minutes,
                    )
                    result["trials"] = [r["id"] for r in updated["trials"][len(state["trials"]) :]]
            elif decision.action == "collect":
                split = decision.split or "train"
                if split != "train" and not (
                    cfg.autonomy.enabled and cfg.autonomy.allow_benchmark_growth
                ):
                    raise ValueError("Autonomous validation/test growth is disabled.")
                if not can_collect(cfg, workflow, state):
                    raise ValueError("Training-data collection budget is unavailable.")
                workflow["data_rounds"] += 1
                workflow["growth_pending"] = f"round-{workflow['data_rounds']:03d}"
                save_json(
                    cfg.output_dir / "data_growth" / workflow["growth_pending"] / "plan.json",
                    {"reason": decision.reason, "queries": decision.queries, "split": split},
                )
                record("acquire_train", "Executing the agent's targeted search plan.")
                continue
            elif decision.action == "review_labels":
                if not (
                    cfg.autonomy.enabled
                    and cfg.autonomy.allow_label_review
                    and cfg.data_growth.enabled
                    and workflow["data_rounds"] < cfg.data_growth.max_rounds
                ):
                    raise ValueError(
                        "Label review requires autonomy and an available data-round budget."
                    )
                workflow["data_rounds"] += 1
                workflow["review_pending"] = f"round-{workflow['data_rounds']:03d}"
                save_json(
                    cfg.output_dir / "label_reviews" / workflow["review_pending"] / "plan.json",
                    {"reason": decision.reason, "split": decision.split, "images": decision.images},
                )
                record("review_labels", f"Reviewing {decision.split} labels from original images.")
                continue
            elif decision.action == "adjust_budget":
                result["result"] = change_budget(
                    base_cfg, workflow["decision_pending"], decision.reason, decision.budget_update
                )
            elif decision.action == "diagnose":
                diagnostic_trials = [
                    r for r in [*state["trials"], *prior_trials(cfg)] if r["status"] == "completed"
                ]
                trial = (
                    next((r for r in diagnostic_trials if r["id"] == decision.trial_id), None)
                    if decision.trial_id
                    else (min(current, key=rank) if current else None)
                )
                if decision.trial_id and trial is None:
                    raise ValueError("Diagnostic trial must be a registered completed trial.")
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
