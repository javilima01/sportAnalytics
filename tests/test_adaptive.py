"""Adaptive decisions must alter real controller flow without escaping its budgets."""

import json
import time
from pathlib import Path

import pytest
from pydantic import ValidationError
from test_research import append_growth_image, fake_worker, good_metrics
from test_research import campaign as campaign

from src.research.adaptive import ResearchDecision
from src.research.autonomous import run_autonomous
from src.research.config import Campaign, Recipe, base_models
from src.research.controller import freeze_dataset, remaining_seconds, run_campaign
from src.research.diagnostics import diagnostic_allowance, run_diagnostic, selected_images
from src.research.runtime import read_json, save_json


def enable(cfg, **settings):
    cfg.proposals = "codex"
    for key, value in settings.items():
        setattr(cfg.diagnostics, key, value)
    (cfg.output_dir / "contract.json").unlink()


class ScriptedAgent:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.evidence = []

    def check_ready(self, folder):
        pass

    def request(self, prompt, schema, directory, **kwargs):
        assert schema is ResearchDecision
        self.evidence.append(json.loads(prompt.split("Evidence:\n", 1)[1]))
        action = next(self.actions)
        if isinstance(action, BaseException):
            raise action
        return ResearchDecision.model_validate({"reason": "test hypothesis", **action})


def worker(command, **kwargs):
    job = read_json(command[-1])
    if job["kind"] == "diagnostic":
        save_json(
            Path(job["folder"]) / "result.json",
            {"diagnostic": job["diagnostic"], "marker": "diagnostic evidence"},
        )
    else:
        fake_worker(command, **kwargs)


def test_agent_diagnoses_then_runs_adjustment_ahead_of_preset_queue(campaign):
    enable(campaign)
    campaign.recipes += [campaign.recipes[0].model_copy(update={"id": "unused", "imgsz": 960})]
    baseline = campaign.recipes[0].model_dump()
    adjusted = {**baseline, "id": "agent-resolution", "imgsz": 1280, "mosaic": 0}
    agent = ScriptedAgent(
        [
            {"action": "train", "recipe": baseline},
            {"action": "diagnose", "diagnostic": "evaluate_train", "trial_id": "trial-0001"},
            {"action": "train", "recipe": adjusted},
            {"action": "finish"},
        ]
    )
    result = run_autonomous(campaign, agent=agent, executor=worker)
    assert result["status"] == "completed"
    trials = read_json(campaign.output_dir / "state.json")["trials"]
    assert [r["recipe"]["id"] for r in trials[:2]] == ["baseline", "agent-resolution"]
    assert all(r["recipe"]["id"] != "unused" for r in trials)
    assert len(agent.evidence) == 4
    assert agent.evidence[0]["diagnostics"][0]["evidence"]["marker"] == "diagnostic evidence"
    assert len(agent.evidence[2]["diagnostics"]) == 2
    assert all(set(e["counts"]) == {"train", "val"} for e in agent.evidence)
    assert all("checkpoint" not in str(e["trials"]) for e in agent.evidence)
    # Same terminal workflow is returned without repeating any agent/worker action.
    assert (
        run_autonomous(campaign, agent=agent, executor=lambda *a, **k: pytest.fail("repeated"))
        == result
    )


def test_agent_can_collect_then_retest_on_new_training_version(campaign):
    enable(campaign)
    campaign.data_growth.enabled = True
    campaign.data_growth.min_train_images = 1
    campaign.data_growth.max_rounds = 1
    recipe = campaign.recipes[0].model_dump()
    agent = ScriptedAgent(
        [
            {"action": "train", "recipe": recipe},
            {"action": "collect", "queries": ["football clearly visible match ball"]},
            {"action": "train", "recipe": {**recipe, "id": "more-data"}},
            {"action": "finish"},
        ]
    )

    def grow(cfg, agent, directory, feedback):
        assert read_json(directory / "plan.json")["queries"] == [
            "football clearly visible match ball"
        ]
        snapshot = append_growth_image(cfg)
        return {"added_images": 1, "dataset_version": snapshot["version"]}

    result = run_autonomous(campaign, agent=agent, executor=worker, grower=grow)
    trials = read_json(campaign.output_dir / "state.json")["trials"]
    assert result["status"] == "completed" and result["data_rounds"] == 1
    assert trials[0]["dataset_version"] != trials[1]["dataset_version"]
    assert agent.evidence[2]["previous_decisions"][-1]["action"] == "collect"


@pytest.mark.parametrize("invalid", ["duplicate", "checkpoint", "diagnostic_trial", "collection"])
def test_invalid_actions_return_feedback_without_executing(campaign, invalid):
    enable(campaign)
    recipe = campaign.recipes[0].model_dump()
    bad = {
        "duplicate": {"action": "train", "recipe": {**recipe, "id": "alias"}},
        "checkpoint": {
            "action": "train",
            "recipe": {**recipe, "id": "outsider", "model": "/unknown.pt"},
        },
        "diagnostic_trial": {
            "action": "diagnose",
            "diagnostic": "evaluate_train",
            "trial_id": "test",
        },
        "collection": {"action": "collect", "queries": ["football"]},
    }[invalid]
    agent = ScriptedAgent([{"action": "train", "recipe": recipe}, bad, {"action": "stop"}])
    result = run_autonomous(campaign, agent=agent, executor=worker)
    assert result["status"] == "stopped"
    assert len(read_json(campaign.output_dir / "state.json")["trials"]) == 1
    assert agent.evidence[2]["previous_decisions"][-1]["action"] == "rejected"


def test_quota_retry_reuses_reserved_decision(campaign):
    from src.research.providers import ProvidersUnavailable

    enable(campaign, max_decisions=1)
    agent = ScriptedAgent([ProvidersUnavailable(time.time()), {"action": "stop"}])
    result = run_autonomous(campaign, agent=agent, executor=worker)
    assert result["decision_count"] == 1
    assert len(list((campaign.output_dir / "decisions").iterdir())) == 1


def test_interrupted_trial_does_not_reexecute_saved_action(campaign):
    enable(campaign)
    recipe = campaign.recipes[0].model_dump()
    agent = ScriptedAgent([{"action": "train", "recipe": recipe}, {"action": "stop"}])

    def interrupt(command, **kwargs):
        if read_json(command[-1])["kind"] == "train":
            raise KeyboardInterrupt
        worker(command, **kwargs)

    with pytest.raises(KeyboardInterrupt):
        run_autonomous(campaign, agent=agent, executor=interrupt)
    result = run_autonomous(
        campaign, agent=agent, executor=lambda *a, **k: pytest.fail("repeated action")
    )
    trials = read_json(campaign.output_dir / "state.json")["trials"]
    assert result["status"] == "stopped" and len(trials) == 1
    assert trials[0]["status"] == "interrupted"


def test_adaptive_growth_resumes_partial_append_without_new_round(campaign):
    enable(campaign)
    campaign.data_growth.enabled = True
    campaign.data_growth.min_train_images = 1
    campaign.data_growth.max_rounds = 1
    agent = ScriptedAgent(
        [
            {"action": "collect", "queries": ["football close ball"]},
            {"action": "stop"},
        ]
    )

    def interrupt(cfg, agent, directory, feedback):
        append_growth_image(cfg, commit=False)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_autonomous(campaign, agent=agent, executor=worker, grower=interrupt)
    workflow = read_json(campaign.output_dir / "autonomous.json")
    assert workflow["growth_pending"] == "round-001"
    assert workflow["decision_pending"] == "decision-001"

    def resume(cfg, agent, directory, feedback):
        assert directory.name == "round-001"
        freeze_dataset(cfg, allow_training_growth=True)
        return {"added_images": 1}

    result = run_autonomous(campaign, agent=agent, executor=worker, grower=resume)
    assert result["data_rounds"] == 1 and result["decision_count"] == 2
    assert read_json(campaign.output_dir / "snapshot.json")["counts"]["train"]["images"] == 2


def test_resume_honors_terminal_decision_saved_before_workflow_update(campaign):
    enable(campaign)
    save_json(
        campaign.output_dir / "autonomous.json",
        {
            "status": "interrupted",
            "stage": "decide",
            "events": [],
            "acquisition_rounds": 0,
            "completed_phases": [],
            "decision_count": 1,
            "decision_pending": "decision-001",
        },
    )
    save_json(
        campaign.output_dir / "decisions/decision-001/outcome.json",
        {
            "action": "stop",
            "reason": "Recorded decision to stop",
        },
    )
    result = run_autonomous(campaign, agent=ScriptedAgent([]), executor=worker)
    assert result["status"] == "stopped" and result["reason"] == "Recorded decision to stop"
    assert result["decision_count"] == 1


def test_hard_killed_initial_diagnostic_recovers_at_exhausted_action_cap(campaign):
    enable(campaign, max_actions=1)
    save_json(
        campaign.output_dir / "diagnostics/state.json",
        [
            {
                "directory": str(campaign.output_dir / "diagnostics/initial"),
                "kind": "inspect_data",
                "status": "running",
                "seconds": 0,
                "reserved_seconds": 300,
            }
        ],
    )
    agent = ScriptedAgent([{"action": "stop"}])
    result = run_autonomous(
        campaign, agent=agent, executor=lambda *a, **k: pytest.fail("reran worker")
    )
    assert result["status"] == "stopped"
    assert agent.evidence[0]["diagnostics"][0]["status"] == "interrupted"


def test_diagnostic_interrupt_is_charged_and_cannot_repeat(campaign):
    freeze_dataset(campaign)
    folder = campaign.output_dir / "diagnostics/check"

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_diagnostic(campaign, "inspect_data", None, folder, executor=interrupt)
    result = run_diagnostic(
        campaign, "inspect_data", None, folder, executor=lambda *a, **k: pytest.fail("repeated")
    )
    assert result["status"] == "interrupted"
    ledger = read_json(campaign.output_dir / "diagnostics/state.json")
    assert ledger[0]["seconds"] == campaign.diagnostics.minutes * 60
    assert (
        remaining_seconds(campaign, {"trials": []})
        == campaign.budget.max_hours * 3600 - ledger[0]["seconds"]
    )


def test_diagnostic_budget_cannot_be_reset_by_another_directory(campaign):
    freeze_dataset(campaign)
    campaign.diagnostics.max_actions = 1
    run_diagnostic(
        campaign, "inspect_data", None, campaign.output_dir / "diagnostics/one", executor=worker
    )
    assert diagnostic_allowance(campaign) == 0
    with pytest.raises(ValueError, match="budget exhausted"):
        run_diagnostic(
            campaign, "inspect_data", None, campaign.output_dir / "diagnostics/two", executor=worker
        )
    with pytest.raises(ValueError, match="only inspect"):
        selected_images(campaign, "test")
    save_json(campaign.output_dir / "final_test.json", {})
    with pytest.raises(ValueError, match="after final"):
        run_diagnostic(
            campaign, "inspect_data", None, campaign.output_dir / "diagnostics/one", executor=worker
        )


def test_diagnostics_reduce_available_trial_runtime(campaign):
    campaign.budget.max_hours = 0.3  # 1080 seconds; a trial needs 900.
    (campaign.output_dir / "contract.json").unlink()
    save_json(
        campaign.output_dir / "diagnostics/state.json",
        [{"directory": "old", "status": "interrupted", "seconds": 300, "reserved_seconds": 300}],
    )
    state = run_campaign(campaign, executor=lambda *a, **k: pytest.fail("exceeded global budget"))
    assert not state["trials"]


def test_decision_payloads_cannot_smuggle_unrelated_actions():
    with pytest.raises(ValidationError):
        ResearchDecision(action="stop", reason="done", queries=["download"])
    with pytest.raises(ValidationError):
        ResearchDecision(action="diagnose", reason="check", diagnostic="read_test")


def test_recipe_exposes_augmentation_and_optimization_knobs():
    recipe = Recipe(
        id="knobs",
        hypothesis="full hyperparameter control",
        erasing=0.0,
        hsv_v=0.2,
        copy_paste=0.5,
        mixup=0.1,
        shear=2.0,
        perspective=0.0005,
        translate=0.2,
        fliplr=0.0,
        close_mosaic=10,
        patience=50,
        lrf=0.1,
        weight_decay=0.001,
        warmup_epochs=1.5,
        dropout=0.1,
    )
    assert recipe.erasing == 0.0 and recipe.patience == 50 and recipe.hsv_v == 0.2
    with pytest.raises(ValidationError):
        Recipe(id="bad", hypothesis="x", erasing=1.5)
    with pytest.raises(ValidationError):
        Recipe(id="bad", hypothesis="x", perspective=0.01)
    with pytest.raises(ValidationError):
        Recipe(id="bad", hypothesis="x", patience=0)


def test_base_models_include_neighbor_architectures(tmp_path):
    checkpoint = tmp_path / "yolov8n.pt"
    checkpoint.write_bytes(b"fixture checkpoint")
    variant = tmp_path / "yolov8-p2.yaml"
    variant.write_text("nc: 2")
    cfg = Campaign(recipes=[Recipe(id="a", hypothesis="h", model=str(checkpoint))])
    models = base_models(cfg)
    assert str(checkpoint) in models and str(variant) in models


def warm_start_worker(command, **kwargs):
    job = read_json(command[-1])
    if job["kind"] == "diagnostic":
        save_json(Path(job["folder"]) / "result.json", {"marker": "diagnostic evidence"})
        return
    checkpoint = Path(job["folder"]) / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"trained weights")
    save_json(Path(job["folder"]) / "metrics.json", good_metrics(checkpoint))


def test_warm_start_repeat_of_converged_settings_is_rejected(campaign):
    enable(campaign)
    recipe = campaign.recipes[0].model_dump()
    warm = str(campaign.output_dir / "trial-0001" / "weights" / "best.pt")
    agent = ScriptedAgent(
        [
            {"action": "train", "recipe": recipe},
            {"action": "train", "recipe": {**recipe, "id": "warm", "model": warm}},
            {
                "action": "train",
                "recipe": {**recipe, "id": "warm-erasing-off", "model": warm, "erasing": 0.0},
            },
            {"action": "stop"},
        ]
    )
    result = run_autonomous(campaign, agent=agent, executor=warm_start_worker)
    assert result["status"] == "stopped"
    trials = read_json(campaign.output_dir / "state.json")["trials"]
    assert [r["recipe"]["id"] for r in trials] == ["baseline", "warm-erasing-off"]
    rejected = agent.evidence[2]["previous_decisions"][-1]
    assert rejected["action"] == "rejected" and "converged settings" in rejected["reason"]


def test_warm_start_continues_time_limited_settings(campaign):
    enable(campaign)
    recipe = campaign.recipes[0].model_dump()
    warm = str(campaign.output_dir / "trial-0001" / "weights" / "best.pt")
    agent = ScriptedAgent(
        [
            {"action": "train", "recipe": recipe},
            {"action": "train", "recipe": {**recipe, "id": "continued", "model": warm}},
            {"action": "stop"},
        ]
    )

    def worker(command, **kwargs):
        job = read_json(command[-1])
        if job["kind"] == "diagnostic":
            save_json(Path(job["folder"]) / "result.json", {"marker": "diagnostic evidence"})
            return
        checkpoint = Path(job["folder"]) / "weights" / "best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"trained weights")
        metrics = good_metrics(checkpoint)
        metrics["training"] = {"stop_reason": "time_budget"}
        save_json(Path(job["folder"]) / "metrics.json", metrics)

    result = run_autonomous(campaign, agent=agent, executor=worker)
    assert result["status"] == "stopped"
    trials = read_json(campaign.output_dir / "state.json")["trials"]
    assert [r["recipe"]["id"] for r in trials] == ["baseline", "continued"]


def test_evidence_signals_surface_plateau_and_generalization_gap(campaign):
    enable(campaign)
    version = freeze_dataset(campaign)["version"]
    seeded = campaign.output_dir / "diagnostics" / "seeded"
    seeded.mkdir(parents=True)
    save_json(seeded / "provenance.json", {"dataset_version": version, "trial_id": "trial-0001"})
    save_json(
        seeded / "outcome.json",
        {
            "kind": "evaluate_train_full",
            "status": "completed",
            "evidence": {"ball": {"ap50": 0.9}, "validation_ball": {"ap50": 0.5}},
        },
    )
    recipe = campaign.recipes[0].model_dump()
    agent = ScriptedAgent(
        [
            {"action": "train", "recipe": recipe},
            {"action": "train", "recipe": {**recipe, "id": "res-672", "imgsz": 672}},
            {"action": "train", "recipe": {**recipe, "id": "res-704", "imgsz": 704}},
            {"action": "stop"},
        ]
    )
    result = run_autonomous(campaign, agent=agent, executor=warm_start_worker)
    assert result["status"] == "stopped"
    signals = agent.evidence[-1]["signals"]
    assert signals["completed_trials_on_version"] == 3
    assert signals["best_current"]["trial_id"] == "trial-0001"
    assert signals["ball_ap50_span_on_version"] == [0.96, 0.96]
    assert "plateau" in signals and "repeated_settings" not in signals
    assert signals["train_validation_gap"]["gap"] == pytest.approx(0.4)
    assert "generalization_limited" in signals
    assert any(m.endswith(".pt") for m in agent.evidence[-1]["available_models"])


@pytest.mark.integration
def test_real_diagnostic_worker_inspects_training_without_test_images(campaign):
    freeze_dataset(campaign)
    # Poison the test image: inspection must not even try decoding it.
    (campaign.dataset_dir / "images/test/test.jpg").write_bytes(b"unreadable sealed test")
    result = run_diagnostic(
        campaign, "inspect_data", None, campaign.output_dir / "diagnostics/real"
    )
    log = campaign.output_dir / "diagnostics/real/run.log"
    assert result["status"] == "completed", log.read_text()
    assert result["evidence"]["splits"]["train"]["class_counts"] == [1, 1]
    assert (log.parent / "train-ball-crops.jpg").is_file()
    assert (log.parent / "val-ball-crops.jpg").is_file()
