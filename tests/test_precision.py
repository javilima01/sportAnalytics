import types

import pytest
import torch
from ultralytics.engine import trainer as trainer_module

import main
from src.models import TrainConfig
from src.precision import enable_mixed_precision, resolve_device
from src.research.config import Campaign


class FakeModel:
    def __init__(self):
        self.callbacks = {}

    def add_callback(self, event, callback):
        self.callbacks[event] = callback


def test_resolve_device_explicit_and_indexed():
    assert resolve_device("mps").type == "mps"
    assert resolve_device("0") == torch.device("cuda:0")


def test_fp32_is_a_no_op(monkeypatch):
    monkeypatch.setattr(trainer_module, "autocast", trainer_module.autocast)
    original = trainer_module.autocast
    model = FakeModel()
    enable_mixed_precision(model, "fp32", torch.device("mps"))
    assert trainer_module.autocast is original
    assert model.callbacks == {}


def test_unknown_precision_rejected():
    with pytest.raises(ValueError):
        enable_mixed_precision(FakeModel(), "int8", torch.device("mps"))


def test_cpu_rejected():
    with pytest.raises(ValueError):
        enable_mixed_precision(FakeModel(), "bf16", torch.device("cpu"))


def test_bf16_patches_autocast_without_scaler(monkeypatch):
    monkeypatch.setattr(trainer_module, "autocast", trainer_module.autocast)
    seen = {}
    monkeypatch.setattr(
        torch.amp,
        "autocast",
        lambda device, dtype=None, enabled=True: (
            seen.update(device=device, dtype=dtype, enabled=enabled) or "ctx"
        ),
    )
    monkeypatch.setattr(
        torch.amp, "GradScaler", lambda *a, **k: pytest.fail("bf16 needs no scaler")
    )
    model = FakeModel()
    enable_mixed_precision(model, "bf16", torch.device("mps"))
    assert trainer_module.autocast(False) == "ctx"
    assert seen == {"device": "mps", "dtype": torch.bfloat16, "enabled": True}
    trainer = types.SimpleNamespace(scaler="untouched")
    model.callbacks["on_train_start"](trainer)
    assert trainer.scaler == "untouched"


def test_fp16_installs_mps_scaler(monkeypatch):
    monkeypatch.setattr(trainer_module, "autocast", trainer_module.autocast)
    monkeypatch.setattr(torch.amp, "autocast", lambda *a, **k: "ctx")
    installed = {}
    monkeypatch.setattr(
        torch.amp,
        "GradScaler",
        lambda device, enabled: installed.update(device=device, enabled=enabled) or "scaler",
    )
    model = FakeModel()
    enable_mixed_precision(model, "fp16", torch.device("mps"))
    trainer = types.SimpleNamespace(scaler="default")
    model.callbacks["on_train_start"](trainer)
    assert trainer.scaler == "scaler"
    assert installed == {"device": "mps", "enabled": True}


def test_config_defaults_and_training_args():
    cfg = TrainConfig(data="data.yaml")
    assert cfg.precision == "fp32"
    assert cfg.half is False
    assert "precision" not in cfg.training_args()
    assert "half" not in cfg.training_args()
    campaign = Campaign()
    assert campaign.precision == "fp32"
    assert campaign.half is False


def test_cli_accepts_precision_and_half():
    assert main.parse_args(["train", "--data", "d.yaml", "--precision", "bf16"]).precision == "bf16"
    assert main.parse_args(["validate", "--data", "d.yaml", "--half"]).half is True
    assert (
        main.parse_args(["generate", "--model", "m.pt", "--video", "v.mp4", "--half"]).half is True
    )
    assert main.parse_args(["visualize", "--dataset", "d", "--half"]).half is True
