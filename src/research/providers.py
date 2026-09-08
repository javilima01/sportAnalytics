"""CLI transport helpers and explicit provider-quota errors."""

import json
import math
import os
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path

from .runtime import run_process


class QuotaExceeded(RuntimeError):
    def __init__(self, provider, retry_at=None):
        self.provider = provider
        self.retry_at = retry_at
        super().__init__(f"{provider} usage limit reached.")


class ProvidersUnavailable(RuntimeError):
    def __init__(self, retry_at):
        self.retry_at = retry_at
        super().__init__("Agent providers are at their usage limits; progress is saved.")


class ProviderWaitExhausted(RuntimeError):
    pass


def bundled_codex():
    """Find the newest native macOS editor bundle without relying on the IDE's PATH."""
    architecture = {"arm64": "aarch64", "x86_64": "x86_64"}.get(platform.machine())
    if platform.system() != "Darwin" or architecture is None:
        return None
    candidates = []
    for editor in (".vscode", ".vscode-insiders", ".cursor"):
        extensions = Path.home() / editor / "extensions"
        for path in extensions.glob(f"openai.chatgpt-*/bin/macos-{architecture}/codex"):
            version = re.fullmatch(r"openai\.chatgpt-(\d+(?:\.\d+)*)(?:-.*)?", path.parents[2].name)
            if version and path.is_file() and os.access(path, os.X_OK):
                candidates.append((tuple(map(int, version[1].split("."))), str(path)))
    return max(candidates)[1] if candidates else None


def executable_path(name):
    """Prefer PATH, then interactive shell startup, then the installed Codex editor bundle."""
    found = shutil.which(name)
    if found:
        return found
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
        raise RuntimeError(f"Executable not found: {name}")
    shell = os.environ.get("SHELL", "/bin/zsh")
    try:
        result = subprocess.run(
            [shell, "-lic", 'command -v -- "$1"', "research", name],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        lines = result.stdout.splitlines()
    except (OSError, subprocess.TimeoutExpired):
        lines = []
    for line in reversed(lines):
        path = Path(line.strip())
        if path.is_absolute() and path.is_file() and os.access(path, os.X_OK):
            return str(path)
    if name == "codex" and (found := bundled_codex()):
        return found
    raise RuntimeError(f"Executable not found: {name}. Set its full path in research.yaml.")


def json_events(log):
    events = []
    for line in Path(log).read_text().splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def quota_reset(payload):
    values = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in ("resetsAt", "reset_at", "resets_at", "retry_after_seconds"):
                if isinstance(value, (int, float)) and math.isfinite(value):
                    values.append(time.time() + value if key == "retry_after_seconds" else value)
            elif isinstance(value, (dict, list)):
                nested = quota_reset(value)
                if nested is not None:
                    values.append(nested)
    elif isinstance(payload, list):
        values = [v for item in payload if (v := quota_reset(item)) is not None]
    return max(values) if values else None


def execute_events(command, *, provider, log, **kwargs):
    failure = None
    try:
        run_process(command, log=log, **kwargs)
    except (RuntimeError, TimeoutError) as error:
        failure = error
    if failure and not Path(log).exists():
        raise failure
    events = json_events(log)
    terminal = [event for event in events if event.get("type") in ("turn.completed", "turn.failed")]
    if (
        provider == "codex"
        and not failure
        and terminal
        and terminal[-1]["type"] == "turn.completed"
    ):
        # Codex can emit reconnect errors and subsequently complete the same request.
        return events
    # Inspect transport errors only. Model text mentioning limits is not a quota signal.
    errors = [
        event.get("error", event)
        for event in events
        if event.get("type") in ("error", "turn.failed")
    ]
    for error in errors:
        text = json.dumps(error).lower()
        markers = (
            "usage limit",
            "usage_limit",
            "usagelimitexceeded",
            "freeusagelimiterror",
            "insufficient_quota",
            "rate limit",
            "rate_limit",
            "ratelimiterror",
            "quota exceeded",
            "quota_exceeded",
            "too_many_requests",
        )
        if any(marker in text for marker in markers) or re.search(
            r'"(?:statusCode|httpStatusCode)"\s*:\s*429', json.dumps(error)
        ):
            raise QuotaExceeded(provider, quota_reset(error))
    if failure:
        raise failure
    if errors:
        raise RuntimeError(f"{provider} returned an error; see {log}.")
    return events


def model_metadata(text, model):
    """Decode the CLI's model-name/pretty-JSON catalog without loading credentials."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"(?m)^\{", text):
        record, _ = decoder.raw_decode(text[match.start() :])
        if f"{record.get('providerID')}/{record.get('id')}" == model:
            return record
    raise ValueError(f"OpenCode does not list the configured model: {model}")
