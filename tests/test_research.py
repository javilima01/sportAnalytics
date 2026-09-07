"""Controller invariants and automated annotation, without network or paid calls."""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from src.dataset import atomic_write, write_labels
from src.research.acquisition import acquire, validate_sources
from src.research.agent import Annotation, Review, strict_schema
from src.research.config import Campaign, Evaluation, Recipe, Source, load_campaign
from src.research.controller import finalize, initialize, run_campaign, run_trial
from src.research.evaluation import acceptance, match_detections, summarize
from src.research.manifest import audit_dataset
from src.research.runtime import campaign_lock, file_hash, read_json, run_process, save_json


@pytest.fixture
def campaign(tmp_path):
    checkpoint = tmp_path / "initial.pt"
    checkpoint.write_bytes(b"fixture checkpoint")
    cfg = Campaign(
        output_dir=tmp_path / "runs",
        dataset_dir=tmp_path / "dataset",
        device="cpu",
        workers=0,
        proposals="queue",
        recipes=[Recipe(id="baseline", hypothesis="baseline", model=str(checkpoint))],
    )
    cfg.acquisition.teacher = str(checkpoint)
    initialize(cfg)
    records = []
    for index, split in enumerate(("train", "val", "test")):
        image = cfg.dataset_dir / "images" / split / f"{split}.jpg"
        image.parent.mkdir(parents=True)
        cv2.imwrite(str(image), np.full((64, 64, 3), index * 60, np.uint8))
        label = cfg.dataset_dir / "labels" / split / f"{split}.txt"
        write_labels(label, [(0, (5, 5, 30, 55)), (1, (40, 40, 48, 48))], (64, 64))
        records.append(
            {
                "image": str(image.relative_to(cfg.dataset_dir)),
                "label": str(label.relative_to(cfg.dataset_dir)),
                "split": split,
                "match_id": f"match-{index}",
                "image_sha256": file_hash(image),
                "label_sha256": file_hash(label),
                "review_status": "agent_labeled",
            }
        )
    atomic_write(
        cfg.dataset_dir / "data.yaml",
        yaml.safe_dump(
            {
                "path": str(cfg.dataset_dir),
                "names": cfg.names,
                **{s: f"images/{s}" for s in ("train", "val", "test")},
            }
        ),
    )
    atomic_write(cfg.dataset_dir / "manifest.jsonl", "".join(json.dumps(r) + "\n" for r in records))
    return cfg


def good_metrics(checkpoint):
    scores = dict.fromkeys(("ap50_95", "ap50", "precision", "recall", "f1"), 0.96)
    return {
        "macro": scores.copy(),
        "ball": scores.copy(),
        "parameters": 1000,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_hash(checkpoint),
        "confidence": 0.25,
        "p95_ms": None,
    }


def fake_worker(command, **kwargs):
    job = read_json(command[-1])
    checkpoint = Path(job.get("checkpoint", job.get("recipe", {}).get("model", "")))
    save_json(Path(job["folder"]) / "metrics.json", good_metrics(checkpoint))


class ReadyAgent:
    def check_ready(self, directory):
        pass


def test_autonomous_runs_every_stage_and_returns_final_checkpoint(campaign):
    from src.research.autonomous import run_autonomous

    result = run_autonomous(campaign, agent=ReadyAgent(), executor=fake_worker)
    assert result["status"] == "completed"
    assert result["completed_phases"] == ["explore", "promote", "confirm"]
    assert result["result"]["status"] == "passed_against_agent_labels"
    assert Path(result["result"]["checkpoint"]).is_file()
    assert len(read_json(campaign.output_dir / "state.json")["trials"]) == 5
    assert run_autonomous(campaign, executor=lambda *a, **k: pytest.fail("repeated")) == result


def test_autonomous_collects_and_retries_without_user_input(campaign):
    from src.research.autonomous import run_autonomous

    prepared = campaign.dataset_dir.with_name("prepared")
    campaign.dataset_dir.rename(prepared)
    calls = []

    def collect(cfg, agent):
        calls.append(cfg)
        if len(calls) == 1:
            raise TimeoutError("transient download error")
        prepared.rename(cfg.dataset_dir)

    result = run_autonomous(campaign, agent=ReadyAgent(), executor=fake_worker, collector=collect)
    assert result["status"] == "completed"
    assert result["acquisition_rounds"] == len(calls) == 2


def test_autonomous_stops_at_acquisition_limit(campaign):
    from src.research.autonomous import run_autonomous

    campaign.dataset_dir.rename(campaign.dataset_dir.with_name("unused"))
    calls = []

    def collect(cfg, agent):
        calls.append(cfg)

    result = run_autonomous(campaign, agent=ReadyAgent(), executor=fake_worker, collector=collect)
    assert result["status"] == "stopped"
    assert len(calls) == campaign.acquisition.max_rounds
    assert not (campaign.output_dir / "state.json").exists()
    run_autonomous(campaign, agent=ReadyAgent(), executor=fake_worker, collector=collect)
    assert len(calls) == campaign.acquisition.max_rounds


def test_autonomous_resumes_interrupted_phase_with_a_recorded_new_attempt(campaign):
    from src.research.autonomous import run_autonomous

    calls = []

    def interrupt(command, **kwargs):
        calls.append(command)
        if len(calls) == 2:
            raise KeyboardInterrupt
        fake_worker(command, **kwargs)

    with pytest.raises(KeyboardInterrupt):
        run_autonomous(campaign, agent=ReadyAgent(), executor=interrupt)
    workflow = read_json(campaign.output_dir / "autonomous.json")
    assert workflow["status"] == "interrupted"
    assert workflow["completed_phases"] == ["explore"]
    result = run_autonomous(campaign, agent=ReadyAgent(), executor=fake_worker)
    assert result["status"] == "completed"
    trials = read_json(campaign.output_dir / "state.json")["trials"]
    assert len(trials) == 6
    assert trials[1]["status"] == "interrupted"
    assert trials[2]["phase"] == "promote" and trials[2]["status"] == "completed"


def test_autonomous_quality_failure_keeps_test_unopened(campaign):
    from src.research.autonomous import run_autonomous

    def weak_worker(command, **kwargs):
        job = read_json(command[-1])
        metrics = good_metrics(Path(job["recipe"]["model"]))
        metrics["ball"]["recall"] = 0.2
        save_json(Path(job["folder"]) / "metrics.json", metrics)

    result = run_autonomous(campaign, agent=ReadyAgent(), executor=weak_worker)
    assert result["status"] == "stopped" and result["stage"] == "confirm"
    assert "final test remains unopened" in result["reason"]
    assert not (campaign.output_dir / "final_test.json").exists()


def test_autonomous_does_not_collect_over_corrupted_data(campaign):
    from src.research.autonomous import run_autonomous

    (campaign.dataset_dir / "labels/val/val.txt").write_text("0 .5 .5 .3 .3\n")
    with pytest.raises(ValueError, match="changed"):
        run_autonomous(
            campaign,
            agent=ReadyAgent(),
            executor=fake_worker,
            collector=lambda *a, **k: pytest.fail("recollected"),
        )
    assert read_json(campaign.output_dir / "autonomous.json")["status"] == "error"


def test_init_starts_autonomous_workflow_and_preserves_existing_config(tmp_path, monkeypatch):
    import src.research.autonomous as autonomous
    from main import parse_args, run_research

    path = tmp_path / "research.yaml"
    calls = []

    def run(cfg):
        calls.append(cfg)
        return {"status": "completed"}

    monkeypatch.setattr(autonomous, "run_autonomous", run)
    run_research(parse_args(["research", "init", "--config", str(path)]))
    original = path.read_text()
    run_research(parse_args(["research", "init", "--config", str(path)]))
    assert path.read_text() == original
    assert len(calls) == 2


def test_agent_readiness_failure_happens_before_collection(campaign):
    from src.research.autonomous import run_autonomous

    class Agent:
        def check_ready(self, directory):
            raise RuntimeError("login required")

    with pytest.raises(RuntimeError, match="login required"):
        run_autonomous(
            campaign,
            agent=Agent(),
            executor=fake_worker,
            collector=lambda *a, **k: pytest.fail("started downloading"),
        )
    assert read_json(campaign.output_dir / "autonomous.json")["stage"] == "preflight"


def test_autonomous_waits_through_quota_without_spending_acquisition_rounds(campaign, monkeypatch):
    import src.research.autonomous as autonomous
    from src.research.providers import ProvidersUnavailable

    prepared = campaign.dataset_dir.with_name("prepared")
    campaign.dataset_dir.rename(prepared)
    now = [1000.0]
    monkeypatch.setattr(autonomous.time, "time", lambda: now[0])
    monkeypatch.setattr(
        autonomous.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds)
    )
    calls = []

    def collect(cfg, agent):
        calls.append(cfg)
        if len(calls) == 1:
            raise ProvidersUnavailable(now[0] + 65)
        prepared.rename(cfg.dataset_dir)

    result = autonomous.run_autonomous(
        campaign, agent=ReadyAgent(), executor=fake_worker, collector=collect
    )
    assert result["status"] == "completed"
    assert result["acquisition_rounds"] == 1
    assert result["provider_wait_seconds"] == 65
    assert any(event["status"] == "waiting" for event in result["events"])


def test_autonomous_stops_at_provider_wait_budget(campaign, monkeypatch):
    import src.research.autonomous as autonomous
    from src.research.providers import ProvidersUnavailable

    campaign.dataset_dir.rename(campaign.dataset_dir.with_name("unused"))
    campaign.fallback.max_wait_hours = 1 / 3600
    (campaign.output_dir / "contract.json").unlink()
    now = [1000.0]
    monkeypatch.setattr(autonomous.time, "time", lambda: now[0])
    monkeypatch.setattr(
        autonomous.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds)
    )

    def collect(cfg, agent):
        raise ProvidersUnavailable(now[0] + 60)

    result = autonomous.run_autonomous(
        campaign, agent=ReadyAgent(), executor=fake_worker, collector=collect
    )
    assert result["status"] == "stopped"
    assert "waiting budget exhausted" in result["reason"]
    assert result["provider_wait_seconds"] == 1


def test_label_quota_does_not_consume_image_retry(campaign):
    from src.research.providers import ProvidersUnavailable

    campaign.acquisition.sources = [
        Source(id=s, url=s, match_id=s, split=s) for s in ("train", "val", "test")
    ]
    folder = campaign.dataset_dir / ".staging/train"
    folder.mkdir(parents=True)
    image = folder / "new_frame.jpg"
    cv2.imwrite(str(image), np.full((64, 64, 3), 17, np.uint8))
    save_json(
        folder / "frames.json",
        [{"image": str(image), "frame_index": 0, "time_seconds": 0, "proposals": []}],
    )

    class Agent:
        def label(self, *args):
            raise ProvidersUnavailable(time.time() + 60)

    with pytest.raises(ProvidersUnavailable):
        acquire(campaign, agent=Agent())
    state = read_json(campaign.output_dir / "acquisition.json")
    assert state["frames"]["new_frame"] == {"attempts": 0, "status": "pending"}
    assert len((campaign.dataset_dir / "manifest.jsonl").read_text().splitlines()) == 3


def test_proposal_quota_does_not_end_exploration_or_repeat_baseline(campaign):
    from src.research.providers import ProvidersUnavailable

    campaign.proposals = "codex"
    campaign.budget.max_exploration_trials = 2
    (campaign.output_dir / "contract.json").unlink()

    class Agent:
        calls = 0

        def propose(self, history, directory):
            self.calls += 1
            if self.calls == 1:
                raise ProvidersUnavailable(time.time() + 60)
            return campaign.recipes[0].model_copy(update={"id": "new", "imgsz": 960})

    agent = Agent()
    with pytest.raises(ProvidersUnavailable):
        run_campaign(campaign, executor=fake_worker, agent=agent)
    assert len(read_json(campaign.output_dir / "state.json")["trials"]) == 1
    state = run_campaign(campaign, executor=fake_worker, agent=agent)
    assert len(state["trials"]) == 2


def test_campaign_resume_confirmation_and_sealed_test(campaign):
    state = run_campaign(campaign, executor=fake_worker)
    assert len(state["trials"]) == 1
    assert state["trials"][0]["status"] == "completed"
    assert len(run_campaign(campaign, executor=fake_worker)["trials"]) == 1
    with pytest.raises(ValueError, match="confirmation"):
        finalize(campaign, executor=fake_worker)
    run_campaign(campaign, phase="promote", executor=fake_worker)
    state = run_campaign(campaign, phase="confirm", executor=fake_worker)
    assert [r["seed"] for r in state["trials"] if r["phase"] == "confirm"] == [0, 1, 2]
    seen = []

    def test_worker(command, **kwargs):
        job = read_json(command[-1])
        assert job["kind"] == "test" and job["confidence"] == 0.25
        seen.append(job)
        fake_worker(command, **kwargs)

    result = finalize(campaign, executor=test_worker)
    assert result["status"] == "passed_against_agent_labels"
    assert finalize(campaign, executor=test_worker) == result
    assert len(seen) == 1
    with pytest.raises(ValueError, match="closed"):
        run_campaign(campaign, executor=fake_worker)
    assert (campaign.output_dir / "results.csv").is_file()


def test_timeout_and_invalid_metrics_preserve_incumbent(campaign):
    state = run_campaign(campaign, executor=fake_worker)

    def timeout(*args, **kwargs):
        raise TimeoutError("deadline")

    run_trial(campaign, state, campaign.recipes[0], "explore", 1, 1, executor=timeout)
    assert state["trials"][-1]["status"] == "timeout"
    assert state["incumbents"]["explore"] == "trial-0001"

    def bad_worker(command, **kwargs):
        save_json(Path(read_json(command[-1])["folder"]) / "metrics.json", {})

    run_trial(campaign, state, campaign.recipes[0], "explore", 2, 1, executor=bad_worker)
    assert state["trials"][-1]["status"] == "failed"
    assert state["incumbents"]["explore"] == "trial-0001"


@pytest.mark.parametrize("group", ["macro", "ball"])
@pytest.mark.parametrize("metric", ["ap50_95", "ap50", "precision", "recall", "f1"])
def test_smallest_passing_student_survives_promotion_and_final_test(campaign, group, metric):
    from src.research.autonomous import run_autonomous

    campaign.recipes = [
        campaign.recipes[0].model_copy(update={"id": size}) for size in ("tiny", "small", "large")
    ]
    (campaign.output_dir / "contract.json").unlink()

    def measured_worker(command, **kwargs):
        job = read_json(command[-1])
        if job["kind"] == "test":
            return fake_worker(command, **kwargs)
        size = job["recipe"]["id"]
        metrics = good_metrics(Path(job["recipe"]["model"]))
        metrics["parameters"] = {"tiny": 100, "small": 1000, "large": 10000}[size]
        if size == "tiny":
            # Even equality on one gate disqualifies the smallest candidate.
            metrics[group][metric] = campaign.evaluation.thresholds[metric]
        elif size == "large":
            # Extra accuracy cannot outweigh a smaller student that already passes.
            for scores in (metrics["macro"], metrics["ball"]):
                scores.update(dict.fromkeys(scores, 0.99))
        save_json(Path(job["folder"]) / "metrics.json", metrics)

    result = run_autonomous(campaign, agent=ReadyAgent(), executor=measured_worker)
    assert result["result"]["status"] == "passed_against_agent_labels"
    state = read_json(campaign.output_dir / "state.json")
    assert state["incumbents"]["explore"] == "trial-0002"
    assert all(r["recipe"]["id"] == "small" for r in state["trials"] if r["phase"] != "explore")


def test_budget_does_not_launch_partial_trial(campaign):
    campaign.budget.max_hours = 0.001
    (campaign.output_dir / "contract.json").unlink()
    state = run_campaign(campaign, executor=lambda *a, **k: pytest.fail("launched"))
    assert state["trials"] == []


def test_interrupted_trial_is_charged_and_not_repeated(campaign):
    state = run_campaign(campaign, executor=fake_worker)
    state["trials"][0].update(status="running", seconds=0)
    save_json(campaign.output_dir / "state.json", state)
    state = run_campaign(campaign, executor=lambda *a, **k: pytest.fail("repeated"))
    assert state["trials"][0]["status"] == "interrupted"
    assert state["trials"][0]["seconds"] == 900


def test_proposal_limit_leaves_room_for_confirmation(campaign):
    campaign.proposals = "codex"
    campaign.budget.max_exploration_trials = 2
    (campaign.output_dir / "contract.json").unlink()

    class Agent:
        def propose(self, history, directory):
            assert all(r["phase"] == "explore" for r in history)
            return campaign.recipes[0].model_copy(update={"id": "new-resolution", "imgsz": 960})

    state = run_campaign(campaign, executor=fake_worker, agent=Agent())
    assert len(state["trials"]) == 2
    state = run_campaign(campaign, phase="confirm", executor=fake_worker)
    assert len(state["trials"]) == 5


def test_changed_initial_checkpoint_is_rejected(campaign):
    run_campaign(campaign, executor=fake_worker)
    Path(campaign.recipes[0].model).write_bytes(b"different initialization")
    with pytest.raises(ValueError, match="Starting checkpoint changed"):
        run_campaign(campaign, phase="promote", executor=fake_worker)


@pytest.mark.integration
def test_real_video_extraction_worker(tmp_path):
    from ultralytics import YOLO

    from src.research.acquisition import ROOT

    cfg = Campaign(device="cpu", workers=0)
    cfg.acquisition.teacher = str(tmp_path / "teacher.pt")
    cfg.acquisition.teacher_imgsz = 64
    cfg.acquisition.max_frames_per_source = 2
    YOLO("yolov8n.yaml").save(cfg.acquisition.teacher)
    video = tmp_path / "video.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 64))
    assert writer.isOpened()
    for i in range(4):
        writer.write(np.full((64, 64, 3), i * 60, np.uint8))
    writer.release()
    source = Source(id="local", url=str(video), match_id="match", split="train")
    job = tmp_path / "job.json"
    save_json(
        job,
        {
            "kind": "extract",
            "campaign": cfg.model_dump(mode="json"),
            "source": source.model_dump(),
            "folder": str(tmp_path),
        },
    )
    run_process(
        [sys.executable, "-m", "src.research.worker", str(job)],
        timeout=60,
        log=tmp_path / "extract.log",
        cwd=ROOT,
    )
    frames = read_json(tmp_path / "frames.json")
    assert [f["frame_index"] for f in frames] == [0, 3]
    assert all(Path(f["image"]).is_file() and isinstance(f["proposals"], list) for f in frames)


def test_discovery_rejects_invented_video_ids(tmp_path, monkeypatch):
    import src.research.acquisition as acquisition
    from src.research.agent import Discovery

    cfg = Campaign(output_dir=tmp_path)

    def search(command, **kwargs):
        save_json(kwargs["log"], {"entries": [{"id": "real", "title": "match"}]})

    class Agent:
        def request(self, *args, **kwargs):
            return Discovery(
                explanation="selection",
                sources=[Source(id="invented", url="bad", match_id="match", split="train")],
            )

    monkeypatch.setattr(acquisition, "run_process", search)
    with pytest.raises(ValueError, match="outside the search results"):
        acquisition.discover(cfg, Agent(), time.monotonic() + 60)


def test_frozen_labels_and_split_leakage(campaign):
    run_campaign(campaign, executor=fake_worker)
    label = campaign.dataset_dir / "labels/val/val.txt"
    label.write_text("0 .5 .5 .5 .5\n")
    with pytest.raises(ValueError, match="changed"):
        audit_dataset(campaign.dataset_dir, campaign.evaluation)


def test_manifest_match_and_path_checks(campaign):
    path = campaign.dataset_dir / "manifest.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[1]["match_id"] = records[0]["match_id"]
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    with pytest.raises(ValueError, match="leakage"):
        audit_dataset(campaign.dataset_dir, campaign.evaluation)
    records[1]["match_id"] = "distinct"
    records[1]["split"] = "train"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    with pytest.raises(ValueError, match="paths disagree"):
        audit_dataset(campaign.dataset_dir, campaign.evaluation)


def test_missing_ball_is_not_zero_score(campaign):
    path = campaign.dataset_dir / "manifest.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    label = campaign.dataset_dir / records[1]["label"]
    label.write_text("0 .5 .5 .5 .5\n")
    records[1]["label_sha256"] = file_hash(label)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    with pytest.raises(ValueError, match="unsupported"):
        audit_dataset(campaign.dataset_dir, campaign.evaluation)


def test_matching_duplicates_and_shared_confidence():
    targets = [[0, 0, 0, 10, 10], [1, 20, 20, 30, 30]]
    predictions = [[0, 0.9, 0, 0, 10, 10], [0, 0.1, 0, 0, 10, 10], [1, 0.8, 20, 20, 30, 30]]
    _, correct = match_detections(predictions, targets, [0.5, 0.95])
    assert correct.tolist() == [[True, True], [True, True], [False, False]]
    metrics = summarize(
        [{"targets": targets, "predictions": predictions}], {0: "player", 1: "ball"}, Evaluation()
    )
    assert metrics["confidence"] > 0.1
    assert metrics["macro"]["precision"] == 1
    assert metrics["ball"]["recall"] == 1
    assert acceptance(metrics, Evaluation())["feasible"]
    fixed = summarize(
        [{"targets": targets, "predictions": predictions}],
        {0: "player", 1: "ball"},
        Evaluation(),
        confidence=0.95,
    )
    assert fixed["macro"]["recall"] == 0
    assert fixed["confidence"] == 0.95


def test_acceptance_strict_and_finite(campaign):
    metrics = good_metrics(Path(campaign.recipes[0].model))
    metrics["ball"]["recall"] = 0.9
    assert not acceptance(metrics, campaign.evaluation)["feasible"]
    metrics["ball"]["recall"] = float("nan")
    with pytest.raises(ValueError, match="Invalid metric"):
        acceptance(metrics, campaign.evaluation)


def test_process_deadline_and_lock(tmp_path):
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        run_process(
            [sys.executable, "-c", "import time; time.sleep(20)"], timeout=0.2, log=tmp_path / "log"
        )
    assert time.monotonic() - start < 4
    with campaign_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another controller"):
            with campaign_lock(tmp_path):
                pytest.fail("lock acquired twice")


def test_storage_budget_checks_fast_processes(tmp_path):
    with pytest.raises(RuntimeError, match="Storage budget exceeded"):
        run_process(
            [
                sys.executable,
                "-c",
                "from pathlib import Path; Path('large').write_bytes(b'x' * 10000)",
            ],
            timeout=5,
            log=tmp_path / "log",
            cwd=tmp_path,
            watch_dir=tmp_path,
            max_bytes=100,
        )


def test_research_cli_init_and_status(tmp_path, capsys):
    from main import parse_args, run_research

    path = tmp_path / "research.yaml"
    run_research(parse_args(["research", "init", "--config-only", "--config", str(path)]))
    capsys.readouterr()
    cfg = load_campaign(path)
    assert cfg.dataset_dir == tmp_path / "datasets/football-pilot"
    with pytest.raises(FileExistsError):
        run_research(parse_args(["research", "init", "--config-only", "--config", str(path)]))
    run_research(parse_args(["research", "status", "--config", str(path)]))
    assert json.loads(capsys.readouterr().out)["experiments"] is None


def test_config_and_annotation_reject_invalid_values(tmp_path):
    with pytest.raises(ValidationError):
        Annotation(class_id=0, x1=0.5, y1=0.1, x2=0.2, y2=0.9)
    with pytest.raises(ValidationError):
        Evaluation(max_p95_ms=30)
    path = tmp_path / "config.yaml"
    path.write_text("output_dir: dataset/runs\ndataset_dir: dataset\n")
    with pytest.raises(ValueError, match="separate"):
        load_campaign(path)
    schema = strict_schema(Review)
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["additionalProperties"] is False


def test_source_plan_requires_distinct_matches():
    sources = [Source(id=s, url=s, match_id="same", split=s) for s in ("train", "val", "test")]
    with pytest.raises(ValueError, match="share a split"):
        validate_sources(sources, 6)


def test_acquisition_uses_agent_boxes_and_resumes(tmp_path):
    cfg = Campaign(output_dir=tmp_path / "runs", dataset_dir=tmp_path / "dataset")
    cfg.acquisition.teacher = str(tmp_path / "teacher.pt")
    Path(cfg.acquisition.teacher).write_bytes(b"teacher")
    cfg.acquisition.sources = [
        Source(id=s, url=s, match_id=s, split=s) for s in ("train", "val", "test")
    ]
    initialize(cfg)
    for index, source in enumerate(cfg.acquisition.sources):
        folder = cfg.dataset_dir / ".staging" / source.id
        folder.mkdir(parents=True)
        image = folder / f"{source.id}_00000000.jpg"
        cv2.imwrite(str(image), np.full((64, 64, 3), index * 60, np.uint8))
        save_json(
            folder / "frames.json",
            [{"image": str(image), "frame_index": 0, "time_seconds": 0, "proposals": []}],
        )

    class Agent:
        calls = 0

        def label(self, *args):
            self.calls += 1
            return Review(
                status="accepted",
                reason="Located player and ball",
                boxes=[
                    Annotation(class_id=0, x1=0.1, y1=0.1, x2=0.4, y2=0.9),
                    Annotation(class_id=1, x1=0.7, y1=0.7, x2=0.8, y2=0.8),
                ],
            )

    agent = Agent()
    assert acquire(cfg, agent)["accepted_images"] == 3
    assert acquire(cfg, agent)["accepted_images"] == 3
    assert agent.calls == 3
    assert audit_dataset(cfg.dataset_dir, cfg.evaluation)["counts"]["val"]["instances"] == [1, 1]


@pytest.mark.integration
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_real_research_worker_training_and_evaluation(campaign, device):
    import torch
    from ultralytics import YOLO

    from src.research.worker import test_job

    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Apple MPS is unavailable.")
    campaign.device = device
    YOLO("yolov8n.yaml").save(campaign.recipes[0].model)
    campaign.recipes[0].epochs = 1
    campaign.recipes[0].imgsz = 64
    campaign.recipes[0].batch = 2
    campaign.budget.exploration_minutes = 2
    (campaign.output_dir / "contract.json").unlink()
    state = run_campaign(campaign)
    trial = state["trials"][0]
    log = campaign.output_dir / trial["id"] / "run.log"
    assert trial["status"] == "completed", log.read_text()
    assert trial["metrics"]["epochs_completed"] == 1
    assert trial["metrics"]["parameters"] > 0
    folder = campaign.output_dir / "test-smoke"
    folder.mkdir()
    test_job(
        {
            "campaign": campaign.model_dump(mode="json"),
            "folder": str(folder),
            "checkpoint": trial["metrics"]["checkpoint"],
            "imgsz": 64,
            "confidence": trial["metrics"]["confidence"],
        }
    )
    assert read_json(folder / "metrics.json")["confidence"] == trial["metrics"]["confidence"]


@pytest.mark.network
def test_live_codex_image_annotation(tmp_path):
    import os

    from ultralytics.utils import ASSETS

    from src.research.agent import CodexAgent

    if os.environ.get("RUN_CODEX_SMOKE") != "1":
        pytest.skip("Set RUN_CODEX_SMOKE=1 to use the signed-in Codex CLI.")
    cfg = Campaign(taxonomy="Player means every visible person; ball means a visible sports ball.")
    review = CodexAgent(cfg).label(ASSETS / "bus.jpg", [], tmp_path / "annotation", 120)
    assert review.status == "accepted"
    assert any(box.class_id == 0 for box in review.boxes)
    assert (tmp_path / "annotation/response.json").is_file()


@pytest.mark.network
def test_live_opencode_image_annotation(tmp_path):
    import os

    from ultralytics.utils import ASSETS

    from src.research.agent import OpenCodeAgent

    if os.environ.get("RUN_OPENCODE_SMOKE") != "1":
        pytest.skip("Set RUN_OPENCODE_SMOKE=1 to call the configured free OpenCode model.")
    cfg = Campaign(taxonomy="Player means every visible person; ball means a visible sports ball.")
    review = OpenCodeAgent(cfg).label(ASSETS / "bus.jpg", [], tmp_path / "annotation", 120)
    assert review.status == "accepted"
    assert any(box.class_id == 0 for box in review.boxes)
    assert (tmp_path / "annotation/response.json").is_file()
