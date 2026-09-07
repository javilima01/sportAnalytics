"""Atomic records, campaign locking, and process-group deadlines."""

import hashlib
import json
import os
import signal
import subprocess
import time
from contextlib import contextmanager, suppress
from pathlib import Path

from ..dataset import atomic_write


def save_json(path, value):
    atomic_write(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@contextmanager
def campaign_lock(directory):
    import fcntl

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "controller.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another controller owns this campaign.") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def tree_bytes(directory):
    total = 0
    for path in Path(directory).rglob("*"):
        with suppress(FileNotFoundError):
            if path.is_file():
                total += path.stat().st_size
    return total


def run_process(
    command, *, timeout, log, cwd=None, stdin=None, watch_dir=None, max_bytes=None, env=None
):
    """Kill the whole child process group on deadline, storage overflow, or interruption."""
    if timeout <= 0:
        raise TimeoutError("No time remains for this operation.")
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with log.open("w") as output:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env=None if env is None else {**os.environ, **env},
        )
        try:
            if stdin is not None:
                process.stdin.write(stdin)
                process.stdin.close()
            while process.poll() is None:
                if time.monotonic() - start >= timeout:
                    raise TimeoutError(f"Process exceeded {timeout:.1f} seconds; see {log}.")
                if (
                    watch_dir is not None
                    and max_bytes is not None
                    and tree_bytes(watch_dir) > max_bytes
                ):
                    raise RuntimeError(f"Storage budget exceeded; see {log}.")
                time.sleep(0.1)
            if process.returncode:
                raise RuntimeError(f"Process exited {process.returncode}; see {log}.")
            if (
                watch_dir is not None
                and max_bytes is not None
                and tree_bytes(watch_dir) > max_bytes
            ):
                raise RuntimeError(f"Storage budget exceeded; see {log}.")
        finally:
            if process.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            # A child may outlive a terminated parent, including by ignoring SIGTERM.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
    return time.monotonic() - start
