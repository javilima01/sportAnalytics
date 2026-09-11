"""Provider routing, quota classification and free-model safeguards without network calls."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.research.agent import CodexAgent, OpenCodeAgent, ResearchAgent, Review
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
    router.routes = [("codex", codex), ("opencode", muse)]
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
    router.routes = [("codex", codex), ("opencode", muse)]
    router.request("again", Review, tmp_path / "second")
    assert not codex.calls and len(muse.calls) == 1
    now[0] += cfg.fallback.codex_retry_seconds + 1
    router.request("back", Review, tmp_path / "third")
    assert len(codex.calls) == 1
    assert read_json(tmp_path / "third/provider.json")["provider"] == "codex"
    assert read_json(cfg.output_dir / "providers.json")["active"] == "codex"
    assert read_json(tmp_path / "third/provider.json")["reasoning_effort"] == "high"
    assert read_json(tmp_path / "first/provider.json")["reasoning_effort"] is None


@pytest.mark.parametrize("effort", ["high", "medium", None])
def test_codex_passes_explicit_model_and_reasoning(cfg, tmp_path, monkeypatch, effort):
    cfg.codex_reasoning_effort = effort
    monkeypatch.setattr("src.research.agent.executable_path", lambda name: "/bin/codex")

    def execute(command, **kwargs):
        assert command[command.index("--model") + 1] == "gpt-6-astra"
        assert "--ignore-user-config" in command
        assert command[command.index("--sandbox") + 1] == "read-only"
        if effort is None:
            assert "--config" not in command
        else:
            assert command[command.index("--config") + 1] == f'model_reasoning_effort="{effort}"'
        Path(command[command.index("--output-last-message") + 1]).write_text(
            accepted().model_dump_json()
        )
        return []

    monkeypatch.setattr("src.research.agent.execute_events", execute)
    assert CodexAgent(cfg).request("inspect", Review, tmp_path / "reply").status == "accepted"


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
    router.routes = [("codex", Client([error])), ("opencode", muse)]
    with pytest.raises(type(error), match=str(error)):
        router.request("request", Review, tmp_path / "request")
    assert not muse.calls


def test_both_limited_raise_resumable_wait(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr("src.research.agent.time.time", lambda: 1000)
    router = ResearchAgent(cfg)
    router.routes = [(p, Client([QuotaExceeded(p)])) for p in ("codex", "opencode")]
    with pytest.raises(ProvidersUnavailable) as result:
        router.request("request", Review, tmp_path / "request")
    assert result.value.retry_at == 1060
    assert router.state["cooldowns"] == {
        "codex:gpt-6-astra": 1300,
        "opencode:opencode/muse-spark-1.3-contributor-free": 1060,
    }


def test_disabled_fallback_waits_for_codex(cfg, tmp_path):
    cfg.fallback.enabled = False
    router = ResearchAgent(cfg)
    assert [provider for provider, _ in router.routes] == ["codex"]
    router.routes = [("codex", Client([QuotaExceeded("codex")]))]
    with pytest.raises(ProvidersUnavailable):
        router.request("request", Review, tmp_path / "request")


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


@pytest.mark.parametrize("shell_failure", [False, True])
def test_codex_discovery_without_editor_path(tmp_path, monkeypatch, shell_failure):
    monkeypatch.setattr("src.research.providers.shutil.which", lambda name: None)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("src.research.providers.platform.system", lambda: "Darwin")
    monkeypatch.setattr("src.research.providers.platform.machine", lambda: "arm64")

    def shell(command, **kwargs):
        if shell_failure:
            raise subprocess.TimeoutExpired(command, 10)
        return SimpleNamespace(stdout="")

    monkeypatch.setattr("src.research.providers.subprocess.run", shell)
    paths = []
    for version, architecture, executable in [
        ("26.9.1", "aarch64", True),
        ("26.10.1", "aarch64", True),
        ("26.11.1", "aarch64", False),
        ("26.12.1", "x86_64", True),
    ]:
        path = (
            tmp_path
            / ".vscode/extensions"
            / f"openai.chatgpt-{version}-darwin-arm64"
            / f"bin/macos-{architecture}/codex"
        )
        path.parent.mkdir(parents=True)
        path.touch(mode=0o755 if executable else 0o644)
        paths.append(path)
    assert executable_path("codex") == str(paths[1])
    # Do not silently replace an explicitly configured missing executable or another provider.
    for name in (str(tmp_path / "missing/codex"), "opencode"):
        with pytest.raises(RuntimeError, match="Executable not found"):
            executable_path(name)


def test_explicit_cli_on_path_takes_precedence(monkeypatch):
    monkeypatch.setattr("src.research.providers.shutil.which", lambda name: "/opt/bin/codex")
    monkeypatch.setattr(
        "src.research.providers.bundled_codex", lambda: pytest.fail("should use PATH")
    )
    assert executable_path("codex") == "/opt/bin/codex"


@pytest.mark.parametrize("cost,vision", [(0, True), (1, True), (0, False), (1, False)])
def test_opencode_checks_model_cost_and_vision(cfg, tmp_path, monkeypatch, capsys, cost, vision):
    monkeypatch.setattr("src.research.agent.executable_path", lambda name: "/bin/opencode")
    metadata = {
        "providerID": "opencode",
        "id": cfg.fallback.model.split("/")[1],
        "cost": {"input": cost, "output": cost},
        "capabilities": {"input": {"image": vision}},
    }

    def execute(command, log, **kwargs):
        assert command[1:3] == ["models", "opencode"]
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        Path(log).write_text(cfg.fallback.model + "\n" + json.dumps(metadata, indent=2))

    monkeypatch.setattr("src.research.agent.run_process", execute)
    client = OpenCodeAgent(cfg)
    if not vision:
        with pytest.raises(ValueError, match="images"):
            client.check_ready(tmp_path)
        assert client.metadata is None  # Failed validation must not authorize a later request.
        return
    client.check_ready(tmp_path)
    assert client.metadata == metadata
    assert ("not listed as free" in capsys.readouterr().out) == bool(cost)


def test_opencode_parses_json_and_uses_only_selected_model(cfg, tmp_path, monkeypatch):
    cfg.fallback.reasoning_effort = "medium"
    client = OpenCodeAgent(cfg)
    client.executable = "/bin/opencode"
    client.metadata = {"ready": True}

    def execute(command, **kwargs):
        assert command[command.index("--model") + 1] == cfg.fallback.model
        assert command[command.index("--variant") + 1] == "medium"
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


def test_providers_swap_between_primary_and_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr("src.research.agent.time.time", lambda: 1000.0)
    cfg = Campaign(
        output_dir=tmp_path / "campaign",
        codex_executable="opencode",
        codex_model="opencode-go/deepseek-v4.1-flash",
        codex_reasoning_effort="high",
        fallback={
            "executable": "codex",
            "model": "gpt-6-astra",
            "reasoning_effort": "medium",
        },
    )
    router = ResearchAgent(cfg)
    assert [provider for provider, _ in router.routes] == ["opencode", "codex"]
    primary = Client([QuotaExceeded("opencode")])
    backup = Client([accepted()])
    router.routes = [("opencode", primary), ("codex", backup)]
    result = router.request("inspect", Review, tmp_path / "swap")
    assert result.status == "accepted"
    assert len(primary.calls) == 1 and len(backup.calls) == 1
    record = read_json(tmp_path / "swap/provider.json")
    assert record["provider"] == "codex"
    assert record["model"] == "gpt-6-astra"
    assert record["reasoning_effort"] == "medium"
    assert read_json(cfg.output_dir / "providers.json")["active"] == "codex"


def test_same_provider_fallback_model_is_attempted(tmp_path, monkeypatch):
    monkeypatch.setattr("src.research.agent.time.time", lambda: 1000.0)
    cfg = Campaign(
        output_dir=tmp_path / "campaign",
        codex_executable="opencode",
        codex_model="opencode-go/deepseek-v4.1-flash",
        fallback={"executable": "opencode", "model": "opencode/muse-spark-1.3-contributor-free"},
    )
    router = ResearchAgent(cfg)
    primary = Client([QuotaExceeded("opencode")])
    free = Client([accepted()])
    router.routes = [("opencode", primary), ("opencode", free)]
    assert router.request("inspect", Review, tmp_path / "same").status == "accepted"
    assert len(primary.calls) == 1 and len(free.calls) == 1


def test_explicit_provider_overrides_executable_name(tmp_path):
    cfg = Campaign(
        output_dir=tmp_path / "campaign",
        provider="opencode",
        codex_executable="codex",
        codex_model="opencode-go/deepseek-v4.1-flash",
    )
    assert [endpoint.provider for endpoint in ResearchAgent(cfg).endpoints] == [
        "opencode",
        "opencode",
    ]


def test_opencode_primary_requires_explicit_model(tmp_path):
    with pytest.raises(ValidationError, match="codex_model"):
        Campaign(output_dir=tmp_path / "campaign", codex_executable="opencode")


def test_fallback_codex_drops_the_opencode_default_model(tmp_path):
    cfg = Campaign(output_dir=tmp_path / "campaign", fallback={"executable": "codex"})
    assert cfg.fallback.model is None
