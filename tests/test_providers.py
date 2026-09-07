"""Provider routing, quota classification and free-model safeguards without network calls."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.research.agent import OpenCodeAgent, ResearchAgent, Review
from src.research.config import Campaign
from src.research.providers import (
    ProvidersUnavailable,
    QuotaExceeded,
    executable_path,
    execute_events,
    model_metadata,
)
from src.research.runtime import read_json


@pytest.fixture
def cfg(tmp_path):
    return Campaign(output_dir=tmp_path / "campaign")


class Client:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def request(self, prompt, response_type, directory, **kwargs):
        self.calls.append((prompt, directory, kwargs))
        directory.mkdir(parents=True, exist_ok=True)
        value = next(self.replies)
        if isinstance(value, Exception):
            raise value
        return value


def accepted():
    return Review(status="accepted", reason="Verified empty image", boxes=[])


def test_fallback_persists_and_returns_to_codex(cfg, tmp_path, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("src.research.agent.time.time", lambda: now[0])
    router = ResearchAgent(cfg)
    codex = Client([QuotaExceeded("codex")])
    muse = Client([accepted()])
    router.clients = {"codex": codex, "opencode": muse}
    image = tmp_path / "image.jpg"
    result = router.request("inspect", Review, tmp_path / "first", images=[image], timeout=10)
    assert result.status == "accepted"
    assert muse.calls[0][2]["images"] == [image]
    assert muse.calls[0][2]["timeout"] <= 10
    assert read_json(tmp_path / "first/provider.json")["model"] == cfg.fallback.model
    assert read_json(cfg.output_dir / "providers.json")["active"] == "opencode"

    # Starting another controller does not reset the quota cooldown.
    router = ResearchAgent(cfg)
    codex = Client([accepted()])
    muse = Client([accepted()])
    router.clients = {"codex": codex, "opencode": muse}
    router.request("again", Review, tmp_path / "second")
    assert not codex.calls and len(muse.calls) == 1
    now[0] += cfg.fallback.codex_retry_seconds + 1
    router.request("back", Review, tmp_path / "third")
    assert len(codex.calls) == 1
    assert read_json(tmp_path / "third/provider.json")["provider"] == "codex"
    assert read_json(cfg.output_dir / "providers.json")["active"] == "codex"


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("authentication failed"),
        TimeoutError("network timeout"),
        ValueError("bad JSON"),
    ],
)
def test_ordinary_errors_do_not_switch_providers(cfg, tmp_path, error):
    router = ResearchAgent(cfg)
    muse = Client([])
    router.clients = {"codex": Client([error]), "opencode": muse}
    with pytest.raises(type(error), match=str(error)):
        router.request("request", Review, tmp_path / "request")
    assert not muse.calls


def test_both_limited_raise_resumable_wait(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr("src.research.agent.time.time", lambda: 1000)
    router = ResearchAgent(cfg)
    router.clients = {p: Client([QuotaExceeded(p)]) for p in ("codex", "opencode")}
    with pytest.raises(ProvidersUnavailable) as result:
        router.request("request", Review, tmp_path / "request")
    assert result.value.retry_at == 1060
    assert router.state["cooldowns"] == {"codex": 1300, "opencode": 1060}


def test_disabled_fallback_waits_for_codex(cfg, tmp_path):
    cfg.fallback.enabled = False
    router = ResearchAgent(cfg)
    muse = Client([])
    router.clients = {"codex": Client([QuotaExceeded("codex")]), "opencode": muse}
    with pytest.raises(ProvidersUnavailable):
        router.request("request", Review, tmp_path / "request")
    assert not muse.calls


@pytest.mark.parametrize(
    "event",
    [
        {"type": "turn.failed", "error": {"message": "You've hit your usage limit."}},
        {"type": "error", "error": {"codexErrorInfo": "UsageLimitExceeded"}},
        {"type": "error", "error": {"name": "APIError", "data": {"statusCode": 429}}},
    ],
)
def test_quota_classification_from_transport_events(tmp_path, monkeypatch, event):
    def execute(command, log, **kwargs):
        Path(log).write_text(json.dumps(event) + "\n")
        raise RuntimeError("exit 1")

    monkeypatch.setattr("src.research.providers.run_process", execute)
    with pytest.raises(QuotaExceeded):
        execute_events(["cli"], provider="codex", log=tmp_path / "log", timeout=1)


def test_model_text_cannot_trigger_quota_switch(tmp_path, monkeypatch):
    def execute(command, log, **kwargs):
        Path(log).write_text(json.dumps({"type": "text", "part": {"text": "usage limit reached"}}))

    monkeypatch.setattr("src.research.providers.run_process", execute)
    events = execute_events(["cli"], provider="opencode", log=tmp_path / "log", timeout=1)
    assert events[0]["type"] == "text"


def test_recovered_codex_error_does_not_discard_completed_response(tmp_path, monkeypatch):
    def execute(command, log, **kwargs):
        events = [
            {"type": "error", "message": "rate limit: reconnecting"},
            {"type": "turn.completed", "usage": {"input_tokens": 1}},
        ]
        Path(log).write_text("".join(json.dumps(event) + "\n" for event in events))

    monkeypatch.setattr("src.research.providers.run_process", execute)
    events = execute_events(["cli"], provider="codex", log=tmp_path / "log", timeout=1)
    assert events[-1]["type"] == "turn.completed"


def test_shell_path_resolution_handles_nvm_without_interpolating_input(tmp_path, monkeypatch):
    executable = tmp_path / "opencode"
    executable.touch(mode=0o755)
    monkeypatch.setattr("src.research.providers.shutil.which", lambda name: None)

    def shell(command, **kwargs):
        assert command[-1] == "opencode"
        assert command[-3] == 'command -v -- "$1"'
        return SimpleNamespace(stdout=f"startup message\n{executable}\n")

    monkeypatch.setattr("src.research.providers.subprocess.run", shell)
    assert executable_path("opencode") == str(executable)
    with pytest.raises(RuntimeError, match="Executable not found"):
        executable_path("$(unexpected-command)")


@pytest.mark.parametrize("cost,vision", [(1, True), (0, False)])
def test_opencode_rejects_paid_or_nonvision_model(cfg, tmp_path, monkeypatch, cost, vision):
    monkeypatch.setattr("src.research.agent.executable_path", lambda name: "/bin/opencode")
    metadata = {
        "providerID": "opencode",
        "id": cfg.fallback.model.split("/")[1],
        "cost": {"input": cost, "output": cost},
        "capabilities": {"input": {"image": vision}},
    }

    def execute(command, log, **kwargs):
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        Path(log).write_text(cfg.fallback.model + "\n" + json.dumps(metadata, indent=2))

    monkeypatch.setattr("src.research.agent.run_process", execute)
    client = OpenCodeAgent(cfg)
    with pytest.raises(ValueError, match="free|images"):
        client.check_ready(tmp_path)
    assert client.metadata is None  # Failed validation must not authorize a later request.


def test_opencode_parses_json_and_uses_only_selected_model(cfg, tmp_path, monkeypatch):
    client = OpenCodeAgent(cfg)
    client.executable = "/bin/opencode"
    client.metadata = {"ready": True}

    def execute(command, **kwargs):
        assert command[command.index("--model") + 1] == cfg.fallback.model
        assert "--pure" in command
        environment = json.loads(kwargs["env"]["OPENCODE_CONFIG_CONTENT"])
        assert environment["small_model"] == cfg.fallback.model
        assert environment["permission"] == {"*": "deny"}
        assert environment["share"] == "disabled"
        return [
            {"type": "text", "part": {"text": "```json\n" + accepted().model_dump_json() + "\n```"}}
        ]

    monkeypatch.setattr("src.research.agent.execute_events", execute)
    assert client.request("inspect", Review, tmp_path) == accepted()
    assert read_json(tmp_path / "response.json")["status"] == "accepted"


def test_catalog_lookup_uses_exact_free_model(cfg):
    paid = {"providerID": "opencode", "id": "muse-spark-1.3"}
    free = {"providerID": "opencode", "id": "muse-spark-1.3-contributor-free"}
    catalog = "paid\n" + json.dumps(paid, indent=2) + "\nfree\n" + json.dumps(free, indent=2)
    assert model_metadata(catalog, cfg.fallback.model) == free
