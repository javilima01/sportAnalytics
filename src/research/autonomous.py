"""One-command orchestration from an empty dataset to a final model test."""

import time
from datetime import datetime, timezone

from .acquisition import acquire
from .agent import ResearchAgent
from .controller import ConfirmationIncomplete, finalize, freeze_dataset, initialize, run_campaign
from .growth import explore_with_growth, grow_training
from .manifest import InsufficientCoverage
from .providers import ProvidersUnavailable, ProviderWaitExhausted
from .runtime import read_json, run_process, save_json


def run_autonomous(
    cfg, *, agent=None, executor=run_process, collector=acquire, grower=grow_training
):
    """Resume stages under the caller's campaign lock; never expand configured budgets."""
    initialize(cfg)
    path = cfg.output_dir / "autonomous.json"
    workflow = read_json(
        path,
        {
            "status": "pending",
            "stage": "preflight",
            "acquisition_rounds": 0,
            "completed_phases": [],
            "events": [],
        },
    )
    if workflow["status"] in ("completed", "stopped"):
        return workflow
    workflow.setdefault("provider_wait_seconds", 0)

    def record(stage, reason, status="running"):
        workflow.update(stage=stage, status=status, reason=reason)
        workflow["events"].append(
            {
                "time": datetime.now(timezone.utc).isoformat(),
                "stage": stage,
                "status": status,
                "reason": reason,
            }
        )
        save_json(path, workflow)
        print(f"[research/{stage}] {reason}", flush=True)

    def with_provider_wait(operation):
        while True:
            try:
                return operation()
            except ProvidersUnavailable as error:
                stage = workflow["stage"]
                record(
                    stage,
                    "The configured agent providers are unavailable. Waiting before retrying; "
                    "accepted images and completed trials are preserved.",
                    "waiting",
                )
                while time.time() < error.retry_at:
                    remaining = (
                        cfg.fallback.max_wait_hours * 3600 - workflow["provider_wait_seconds"]
                    )
                    if remaining <= 0:
                        raise ProviderWaitExhausted(
                            "Provider waiting budget exhausted. Results are preserved."
                        ) from error
                    delay = min(30, error.retry_at - time.time(), remaining)
                    if delay <= 0:
                        break
                    # Reserve each sleep before waiting so a crash cannot reset this allowance.
                    workflow["provider_wait_seconds"] += delay
                    save_json(path, workflow)
                    time.sleep(delay)
                record(stage, "Retrying agent work after its quota cooldown.")

    try:
        # A recorded final test is terminal even if the orchestration process died afterwards.
        if (cfg.output_dir / "final_test.json").exists():
            result = finalize(cfg, executor=executor)
            workflow["result"] = result
            record("finalize", f"Final test: {result['status']}.", "completed")
            return workflow

        record("preflight", "Checking Codex CLI login before acquisition or training.")
        agent = agent or ResearchAgent(cfg)
        agent.check_ready(cfg.output_dir / "preflight")

        # Only inadequate coverage permits more acquisition. Corruption/leakage remains an error.
        while True:
            if workflow.get("growth_pending"):
                break  # Finish the reserved append-only transaction before checking its new snapshot.
            if (cfg.output_dir / "snapshot.json").exists():
                freeze_dataset(cfg)
                break
            coverage = "No accepted dataset is available yet."
            if (cfg.dataset_dir / "manifest.jsonl").exists():
                try:
                    freeze_dataset(cfg)
                    break
                except InsufficientCoverage as error:
                    coverage = str(error)
            acquisition = read_json(cfg.output_dir / "acquisition.json", {})
            if acquisition.get("stop_reason"):
                record("acquire", f"{acquisition['stop_reason']} {coverage}", "stopped")
                return workflow
            if workflow["acquisition_rounds"] >= cfg.acquisition.max_rounds:
                record("acquire", f"Acquisition round limit reached. {coverage}", "stopped")
                return workflow
            workflow["acquisition_rounds"] += 1
            record(
                "acquire",
                f"{coverage} Starting acquisition round "
                f"{workflow['acquisition_rounds']}/{cfg.acquisition.max_rounds}.",
            )
            try:
                with_provider_wait(lambda: collector(cfg, agent=agent))
            except ProviderWaitExhausted:
                raise
            except (RuntimeError, TimeoutError, ValueError) as error:
                record("acquire", f"Acquisition attempt failed: {error}")

        if cfg.data_growth.enabled and "explore" not in workflow["completed_phases"]:
            if not explore_with_growth(
                cfg, workflow, record, with_provider_wait, agent, executor, grower
            ):
                return workflow

        for phase in ("explore", "promote", "confirm"):
            if phase in workflow["completed_phases"]:
                continue
            reasons = {
                "explore": "Dataset passed its audit. Running baselines and result-driven proposals.",
                "promote": "Giving the best exploration recipe a longer training budget.",
                "confirm": "Training the selected recipe with seeds 0, 1, and 2.",
            }
            record(phase, reasons[phase])
            state = with_provider_wait(
                lambda: run_campaign(
                    cfg, phase=phase, executor=executor, agent=agent, retry_interrupted=True
                )
            )
            completed = [
                r
                for r in state["trials"]
                if r["phase"] == phase
                and r["status"] == "completed"
                and r.get("dataset_version") == state.get("dataset_version")
            ]
            if not completed:
                record(
                    phase,
                    "No completed candidate: the available trial budget was exhausted "
                    "or every attempt failed. Inspect results.csv and trial logs.",
                    "stopped",
                )
                return workflow
            workflow["completed_phases"].append(phase)
            record(phase, f"Finished {phase}: {len(completed)} completed trial(s).")

        record("finalize", "Checking confirmation quality before opening the final test.")
        # finalize checks all three seeds and creates no test record if they are ineligible.
        try:
            result = finalize(cfg, executor=executor)
        except ConfirmationIncomplete as error:
            record(
                "confirm",
                str(error) + " Quality or remaining budget was insufficient; "
                "the final test remains unopened.",
                "stopped",
            )
            return workflow
        workflow["result"] = result
        record("finalize", f"Final test: {result['status']}.", "completed")
    except ProviderWaitExhausted as error:
        record(workflow["stage"], str(error), "stopped")
    except KeyboardInterrupt:
        record(workflow["stage"], "Interrupted. Run research init again to resume.", "interrupted")
        raise
    except (OSError, RuntimeError, ValueError, TimeoutError) as error:
        record(workflow["stage"], str(error), "error")
        raise
    return workflow
