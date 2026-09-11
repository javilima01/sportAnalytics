"""Autonomous budget, benchmark and annotation changes retain reproducible boundaries."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from test_adaptive import ScriptedAgent, enable, worker
from test_research import campaign as campaign

from src.dataset import atomic_write, write_labels
from src.research.adaptive import ResearchDecision, evidence
from src.research.agent import Annotation, Discovery, Review
from src.research.allowances import BudgetUpdate, change_budget, effective_campaign
from src.research.autonomous import run_autonomous
from src.research.controller import freeze_dataset, run_campaign
from src.research.growth import grow_training
from src.research.label_review import review_labels
from src.research.runtime import file_hash, read_json, save_json


def fresh_checkpoint_worker(command, **kwargs):
    """Like the real worker, each trial produces its own new checkpoint file."""
    job = read_json(command[-1])
    if job["kind"] == "diagnostic":
        save_json(Path(job["folder"]) / "result.json", {"marker": "diagnostic evidence"})
        return
    checkpoint = Path(job["folder"]) / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"trained weights")
    from test_research import good_metrics

    save_json(Path(job["folder"]) / "metrics.json", good_metrics(checkpoint))


def append_image(cfg, split, name="added", match="new-match", commit=True):
    image = cfg.dataset_dir / "images" / split / f"{name}.jpg"
    cv2.imwrite(str(image), np.full((64, 64, 3), 241, np.uint8))
    label = cfg.dataset_dir / "labels" / split / f"{name}.txt"
    write_labels(label, [(0, (2, 2, 20, 40)), (1, (45, 45, 53, 53))], (64, 64))
    manifest = cfg.dataset_dir / "manifest.jsonl"
    record = {
        "image": str(image.relative_to(cfg.dataset_dir)),
        "label": str(label.relative_to(cfg.dataset_dir)),
        "split": split,
        "match_id": match,
        "image_sha256": file_hash(image),
        "label_sha256": file_hash(label),
        "review_status": "agent_labeled",
    }
    atomic_write(manifest, manifest.read_text() + json.dumps(record) + "\n")
    if commit:
        return freeze_dataset(cfg, allow_growth_splits={split})


def test_agent_extends_allowances_and_repeats_recipe_with_more_time(campaign):
    enable(campaign)
    campaign.budget.max_trials = campaign.budget.max_exploration_trials = 1
    recipe = campaign.recipes[0].model_dump()
    agent = ScriptedAgent(
        [
            {"action": "train", "recipe": recipe},
            {
                "action": "adjust_budget",
                "budget_update": {
                    "max_trials": 4,
                    "max_exploration_trials": 3,
                    "max_hours": 48,
                    "data_rounds": 20,
                },
            },
            {"action": "train", "recipe": {**recipe, "id": "longer"}, "minutes": 45},
            {"action": "stop"},
        ]
    )
    result = run_autonomous(campaign, agent=agent, executor=fresh_checkpoint_worker)
    state = read_json(campaign.output_dir / "state.json")
    assert result["status"] == "stopped"
    assert [r["reserved_seconds"] for r in state["trials"]] == [900, 2700]
    assert campaign.budget.max_trials == 1
    assert effective_campaign(campaign).budget.max_trials == 4
    assert agent.evidence[2]["working_budgets"]["max_hours"] == 48
    assert read_json(campaign.output_dir / "contract.json")["budget"]["max_trials"] == 1
    assert len(read_json(campaign.output_dir / "budget_changes.json")) == 1
    assert run_autonomous(campaign, agent=ScriptedAgent([]), executor=worker) == result


def test_budget_journal_is_idempotent_and_ceilings_cannot_be_extended(campaign):
    update = BudgetUpdate(max_hours=24)
    first = change_budget(campaign, "decision-1", "longer learning", update)
    assert change_budget(campaign, "decision-1", "retry", update) == first
    with pytest.raises(ValueError, match="ceiling"):
        change_budget(campaign, "decision-2", "too much", BudgetUpdate(max_hours=1000))
    with pytest.raises(ValueError, match="ceiling"):
        change_budget(campaign, "decision-2", "reset consumed time", BudgetUpdate(max_hours=1))
    assert effective_campaign(campaign).budget.max_hours == 24
    with pytest.raises(ValueError):
        ResearchDecision(
            action="adjust_budget", reason="weaken gates", budget_update={"precision": 0.1}
        )


@pytest.mark.parametrize("split", ["val", "test"])
def test_agent_can_grow_benchmark_and_retrain_without_promoting_old_version(campaign, split):
    enable(campaign)
    campaign.data_growth.enabled = True
    campaign.data_growth.min_train_images = 1
    recipe = campaign.recipes[0].model_dump()
    agent = ScriptedAgent(
        [
            {"action": "train", "recipe": recipe},
            {"action": "collect", "split": split, "queries": ["independent football matches"]},
            {"action": "finish"},  # Current-version training is required after a benchmark change.
            {"action": "train", "recipe": {**recipe, "id": "new-benchmark"}},
            {"action": "stop"},
        ]
    )

    def grow(cfg, agent, directory, feedback):
        assert read_json(directory / "plan.json")["split"] == split
        snapshot = append_image(cfg, split)
        return {"added_images": 1, "split": split, "dataset_version": snapshot["version"]}

    run_autonomous(campaign, agent=agent, executor=worker, grower=grow)
    trials = read_json(campaign.output_dir / "state.json")["trials"]
    assert len(trials) == 2
    assert trials[0]["benchmark_version"] != trials[1]["benchmark_version"]
    assert agent.evidence[3]["previous_decisions"][-1]["action"] == "rejected"
    assert not (campaign.output_dir / "final_test.json").exists()


@pytest.mark.parametrize("split", ["val", "test"])
def test_interrupted_benchmark_append_resumes_same_round(campaign, split):
    enable(campaign)
    campaign.data_growth.enabled = True
    campaign.data_growth.min_train_images = 1
    agent = ScriptedAgent(
        [{"action": "collect", "split": split, "queries": ["football"]}, {"action": "stop"}]
    )

    def interrupt(cfg, agent, directory, feedback):
        append_image(cfg, split, commit=False)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_autonomous(campaign, agent=agent, executor=worker, grower=interrupt)

    def resume(cfg, agent, directory, feedback):
        assert directory.name == "round-001"
        freeze_dataset(cfg, allow_growth_splits={split})
        return {"added_images": 1}

    result = run_autonomous(campaign, agent=agent, executor=worker, grower=resume)
    assert result["data_rounds"] == 1
    assert read_json(campaign.output_dir / "snapshot.json")["counts"][split]["images"] == 2


def test_growth_recovers_known_sources_without_initial_discovery(campaign, monkeypatch):
    import src.research.acquisition as acquisition
    from src.research.config import Source

    campaign.data_growth.enabled = True
    freeze_dataset(campaign)
    directory = campaign.output_dir / "data_growth/round-001"
    save_json(
        directory / "plan.json",
        {"reason": "coverage", "split": "test", "queries": ["new football match"]},
    )
    calls = []

    def process(command, **kwargs):
        calls.append(command)
        if "yt_dlp" in command:
            save_json(kwargs["log"], {"entries": [{"id": "fresh", "title": "New match"}]})
        else:
            job = read_json(command[-1])
            assert job["source"]["split"] == "test"
            folder = Path(job["folder"])
            path = folder / "fresh_00000000.jpg"
            cv2.imwrite(str(path), np.full((64, 64, 3), 241, np.uint8))
            save_json(
                folder / "frames.json",
                [{"image": str(path), "frame_index": 0, "time_seconds": 0, "proposals": []}],
            )

    class Agent:
        def request(self, prompt, schema, folder, **kwargs):
            assert schema is Discovery and "split=test" in prompt
            for key in ("match-0", "match-1", "match-2"):
                assert key in prompt
            return Discovery(
                explanation="independent",
                sources=[Source(id="fresh", url="unused", match_id="fresh-match", split="test")],
            )

        def label(self, *args):
            return accepted_review()

    monkeypatch.setattr(acquisition, "run_process", process)
    result = grow_training(campaign, Agent(), directory, {})
    assert result["split"] == "test" and result["added_images"] == 1
    assert len(calls) == 2  # One targeted search, one extraction; no initial all-split search.
    assert len(read_json(campaign.output_dir / "acquisition.json")["sources"]) == 1


def accepted_review():
    return Review(
        status="accepted",
        reason="Corrected from original pixels",
        boxes=[
            Annotation(class_id=0, x1=0.1, y1=0.1, x2=0.4, y2=0.8),
            Annotation(class_id=1, x1=0.7, y1=0.7, x2=0.8, y2=0.8),
        ],
    )


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_automatic_review_preserves_original_labels_and_versions(campaign, split):
    campaign.data_growth.enabled = True
    old = freeze_dataset(campaign)
    original = (campaign.dataset_dir / f"labels/{split}/{split}.txt").read_text()
    directory = campaign.output_dir / "label_reviews/round-001"

    class Agent:
        def label(self, path, proposals, folder, timeout):
            assert path.parent.name == split and len(proposals) == 2
            assert all("confidence" not in p for p in proposals)
            return accepted_review()

    result = review_labels(campaign, Agent(), directory, {"split": split})
    new = freeze_dataset(campaign)
    assert result["changed_labels"] == 1 and old["version"] != new["version"]
    assert (old["benchmark_version"] == new["benchmark_version"]) == (split == "train")
    journal = read_json(directory / "revision.json")
    assert journal["changes"][0]["before"] == original
    version = read_json(campaign.output_dir / f"dataset_versions/{new['version']}.json")
    assert version["label_revision"] == str(directory / "revision.json")
    assert review_labels(campaign, None, directory, {"split": split}) == result


def test_review_transaction_recovers_after_labels_written_before_manifest(campaign, monkeypatch):
    import src.research.label_review as module

    campaign.data_growth.enabled = True
    freeze_dataset(campaign)
    directory = campaign.output_dir / "label_reviews/round-001"
    write = module.atomic_write

    def interrupt(path, content):
        if Path(path).name == "manifest.jsonl":
            raise KeyboardInterrupt
        write(path, content)

    monkeypatch.setattr(module, "atomic_write", interrupt)

    class Agent:
        def label(self, *args):
            return accepted_review()

    with pytest.raises(KeyboardInterrupt):
        review_labels(campaign, Agent(), directory, {"split": "val"})
    assert (directory / "revision.json").exists()
    monkeypatch.setattr(module, "atomic_write", write)
    result = review_labels(campaign, None, directory, {"split": "val"})
    assert result["changed_labels"] == 1
    freeze_dataset(campaign)


def test_bad_review_does_not_destroy_last_required_class(campaign):
    campaign.data_growth.enabled = True
    before = freeze_dataset(campaign)

    class Agent:
        def label(self, *args):
            return Review(status="accepted", reason="No ball", boxes=accepted_review().boxes[:1])

    result = review_labels(
        campaign, Agent(), campaign.output_dir / "label_reviews/round-001", {"split": "val"}
    )
    assert result["status"] == "needs_more_data" and result["changed_labels"] == 0
    assert freeze_dataset(campaign) == before


def test_diagnostic_latest_images_and_training_evidence(campaign):
    snapshot = freeze_dataset(campaign)
    for name in ("initial", "decision-002"):
        folder = campaign.output_dir / "diagnostics" / name
        save_json(folder / "outcome.json", {"status": "completed"})
        save_json(folder / "provenance.json", {"dataset_version": snapshot["version"]})
        (folder / "val-ball-crops.jpg").write_bytes(b"fixture")
    payload, images = evidence(
        campaign, {"data_rounds": 0, "decision_count": 0}, {"trials": []}, snapshot
    )
    assert images[0].parent.name == "decision-002"
    assert payload["test_coverage"] == snapshot["counts"]["test"]
    assert "test-ball-crops" not in str(images)


def test_localization_reports_near_misses_without_relaxing_acceptance():
    from src.research.evaluation import localization_report, match_detections

    frames = [
        {
            "image": "ball.jpg",
            "targets": [[1, 10, 10, 20, 20]],
            "predictions": [[1, 0.2, 10, 10, 20, 32], [0, 0.9, 10, 10, 20, 20]],
        }
    ]
    report = localization_report(frames, 1)
    assert report["overlap_at_least_040"] == 1 and report["overlap_at_least_050"] == 0
    assert not match_detections(frames[0]["predictions"], frames[0]["targets"], [0.5])[1].any()


def test_time_schedule_reaches_final_lr_before_short_deadline():
    from src.research.worker import schedule_fraction

    assert schedule_fraction(0, 100, 0, 1800) == 0
    assert schedule_fraction(40, 100, 1620, 1800) == 1
    assert schedule_fraction(99, 100, 500, 1800) == 1
    assert schedule_fraction(20, 100, 810, 1800) == 0.5


def test_long_trial_ceiling_rejection_and_registered_warm_start(campaign):
    enable(campaign)
    recipe = campaign.recipes[0].model_dump()
    agent = ScriptedAgent(
        [
            {"action": "train", "recipe": recipe, "minutes": 241},
            {"action": "train", "recipe": recipe},
            {"action": "train", "recipe": {**recipe, "id": "warm"}, "minutes": 45},
            {"action": "stop"},
        ]
    )
    run_autonomous(campaign, agent=agent, executor=fresh_checkpoint_worker)
    assert agent.evidence[1]["previous_decisions"][-1]["action"] == "rejected"
    assert len(read_json(campaign.output_dir / "state.json")["trials"]) == 2


def test_prior_results_available_but_not_current_candidates(campaign):
    run_campaign(campaign, executor=worker)
    previous = campaign.output_dir
    cfg = campaign.model_copy(
        update={"output_dir": previous.parent / "new-campaign", "prior_campaign": previous}
    )
    cfg.proposals = "codex"
    agent = ScriptedAgent(
        [
            {
                "action": "diagnose",
                "diagnostic": "evaluate_train_full",
                "trial_id": "prior/trial-0001",
            },
            {"action": "finish"},
            {"action": "stop"},
        ]
    )
    run_autonomous(cfg, agent=agent, executor=worker)
    assert agent.evidence[0]["prior_campaign_trials"][0]["id"] == "prior/trial-0001"
    assert agent.evidence[2]["previous_decisions"][-1]["action"] == "rejected"
    assert not (cfg.output_dir / "final_test.json").exists()
