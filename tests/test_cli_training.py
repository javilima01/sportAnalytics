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
