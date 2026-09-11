from unittest.mock import Mock

import pytest
from pydantic import ValidationError

import main
from src.config import setup_logger
from src.models import TrainConfig
from src.trainer import YOLOFineTuner


@pytest.mark.parametrize("segment", ["-1:2", "nan:2", "1:inf", "3:2", "bad"])
def test_invalid_segment(segment):
    with pytest.raises(SystemExit) as error:
        main.parse_args(
            ["generate", "--model", "best.pt", "--video", "match.mp4", "--segments=" + segment]
        )
    assert error.value.code == 2


def test_default_device_is_portable():
    assert main.parse_args(["train", "--data", "data.yaml"]).device is None
    assert TrainConfig(data="data.yaml").device is None


def test_visualize_rejects_edit_with_model():
    with pytest.raises(SystemExit) as error:
        main.parse_args(["visualize", "--dataset", "data", "--model", "model.pt", "--edit"])
    assert error.value.code == 2


def test_cli_visualize_prediction_mode(monkeypatch, tmp_path):
    viewer = Mock()
    factory = Mock(return_value=viewer)
    monkeypatch.setattr("src.visualizer.Visualizer", factory)
    args = main.parse_args(
        [
            "visualize",
            "--dataset",
            "data",
            "--model",
            "model.pt",
            "--conf",
            "0.4",
            "--imgsz",
            "640",
            "--device",
            "cpu",
            "--save_dir",
            str(tmp_path / "out"),
        ]
    )
    main.run_visualize(args)
    kwargs = factory.call_args.kwargs
    assert kwargs["model_path"] == "model.pt"
    assert kwargs["conf"] == 0.4
    assert kwargs["imgsz"] == 640
    assert kwargs["device"] == "cpu"
    viewer.visualize.assert_called_once_with(save_dir=str(tmp_path / "out"))


@pytest.mark.parametrize("options", [{"epochs": 0}, {"batch": 0}, {"imgsz": 0}, {"workers": -1}])
def test_training_validation(options):
    with pytest.raises(ValidationError):
        TrainConfig(data="data.yaml", **options)


def test_logger_does_not_duplicate_handlers(tmp_path):
    name = "test.logger"
    logger = setup_logger(name, log_file=str(tmp_path / "run.log"))
    try:
        setup_logger(name, log_file=str(tmp_path / "run.log"))
        assert len(logger.handlers) == 2
        assert logger.propagate is False
    finally:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


def test_training_uses_actual_run_directory(tmp_path, monkeypatch):
    data = tmp_path / "data.yaml"
    data.touch()
    model = Mock(task="detect")
    model.trainer.save_dir = tmp_path / "run2"
    model.train.return_value.results_dict = {"loss": 0.1}
    model.val.return_value.results_dict = {"map": 0.2}
    monkeypatch.setattr("src.trainer.YOLO", Mock(return_value=model))
    writer = Mock()
    monkeypatch.setattr("src.trainer.SummaryWriter", writer)
    trainer = YOLOFineTuner(TrainConfig(data=str(data)))
    trainer.train()
    trainer.validate()
    trainer.close()
    writer.assert_called_once_with(log_dir=str(tmp_path / "run2/tensorboard"))
    writer.return_value.close.assert_called_once()
    assert "tensorboard_dir" not in model.train.call_args.kwargs
    assert model.val.call_args.kwargs["split"] == "val"


def test_cli_closes_trainer_on_failure(monkeypatch):
    trainer = Mock()
    trainer.train.side_effect = RuntimeError("failed")
    monkeypatch.setattr("src.trainer.YOLOFineTuner", Mock(return_value=trainer))
    with pytest.raises(RuntimeError):
        main.run_train(main.parse_args(["train", "--data", "data.yaml"]))
    trainer.close.assert_called_once()
    trainer.validate.assert_not_called()


def test_collect_model_paths_expands_and_deduplicates(tmp_path):
    folder = tmp_path / "models"
    folder.mkdir()
    for name in ("a.pt", "b.pt", "notes.txt"):
        (folder / name).touch()
    paths = main.collect_model_paths([str(folder / "b.pt"), str(folder)], None)
    assert [path.name for path in paths] == ["b.pt", "a.pt"]


def test_collect_model_paths_rejects_empty_and_missing(tmp_path):
    with pytest.raises(ValueError):
        main.collect_model_paths([], None)
    with pytest.raises(NotADirectoryError):
        main.collect_model_paths([], str(tmp_path / "missing-dir"))
    with pytest.raises(FileNotFoundError):
        main.collect_model_paths([str(tmp_path / "missing.pt")], None)


def test_metric_row_normalizes_ultralytics_keys():
    metrics = Mock(
        results_dict={"metrics/mAP50(B)": 0.5, "metrics/mAP50-95(B)": 0.3},
        speed={"inference": 4.2},
    )
    assert main.metric_row(metrics) == {"mAP50": 0.5, "mAP50-95": 0.3, "inference_ms": 4.2}


def test_results_table_marks_best_and_lists_failures():
    summary = {
        "small.pt": {"precision": 0.8, "recall": 0.7, "inference_ms": 10.0},
        "large.pt": {"precision": 0.9, "recall": 0.6, "inference_ms": 12.0},
    }
    table = main.format_results_table(summary, {"broken.pt": "boom"})
    assert "0.9000*" in table
    assert "0.8000*" not in table
    assert "0.7000*" in table
    assert "10.0*" in table
    assert "12.0*" not in table
    assert "FAILED broken.pt: boom" in table


def test_cli_validate_prints_table_and_exits_on_failure(tmp_path, monkeypatch, capsys):
    data = tmp_path / "data.yaml"
    data.touch()
    good, bad = tmp_path / "good.pt", tmp_path / "bad.pt"
    good.touch()
    bad.touch()

    class FakeTrainer:
        def __init__(self, cfg):
            self.cfg = cfg

        def validate(self, split="val"):
            assert split == "test"
            if self.cfg.model.endswith("bad.pt"):
                raise RuntimeError("broken checkpoint")
            return Mock(results_dict={"metrics/mAP50(B)": 0.61}, speed={"inference": 5.5})

        def close(self):
            pass

    monkeypatch.setattr("src.trainer.YOLOFineTuner", FakeTrainer)
    args = main.parse_args(
        ["validate", "--data", str(data), "--model", str(good), str(bad), "--split", "test"]
    )
    with pytest.raises(SystemExit) as error:
        main.run_validate(args)
    assert error.value.code == 1
    output = capsys.readouterr().out
    assert str(good) in output
    assert "mAP50" in output
    assert "inference_ms" in output
    assert f"FAILED {bad}: broken checkpoint" in output
