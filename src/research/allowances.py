"""Audited working-budget changes without rewriting the user's campaign contract."""

from pydantic import Field

from .config import Campaign, Settings
from .runtime import read_json, save_json, value_hash


class BudgetUpdate(Settings):
    exploration_minutes: float | None = Field(None, gt=0)
    promotion_minutes: float | None = Field(None, gt=0)
    confirmation_minutes: float | None = Field(None, gt=0)
    max_hours: float | None = Field(None, gt=0)
    max_trials: int | None = Field(None, ge=1)
    max_exploration_trials: int | None = Field(None, ge=1)
    data_rounds: int | None = Field(None, ge=0)
    acquisition_minutes: float | None = Field(None, gt=0)
    download_gb: float | None = Field(None, gt=0)
    storage_gb: float | None = Field(None, gt=0)
    diagnostic_actions: int | None = Field(None, ge=0)
    diagnostic_minutes: float | None = Field(None, gt=0)
    diagnostic_total_minutes: float | None = Field(None, gt=0)
    decisions: int | None = Field(None, ge=1)


# update name -> (working settings section, field, autonomy ceiling)
FIELDS = {
    "exploration_minutes": ("budget", "exploration_minutes", "max_trial_minutes"),
    "promotion_minutes": ("budget", "promotion_minutes", "max_trial_minutes"),
    "confirmation_minutes": ("budget", "confirmation_minutes", "max_trial_minutes"),
    "max_hours": ("budget", "max_hours", "max_hours"),
    "max_trials": ("budget", "max_trials", "max_trials"),
    "max_exploration_trials": ("budget", "max_exploration_trials", "max_exploration_trials"),
    "data_rounds": ("data_growth", "max_rounds", "max_data_rounds"),
    "acquisition_minutes": ("acquisition", "minutes", "max_acquisition_minutes"),
    "download_gb": ("acquisition", "download_gb", "max_download_gb"),
    "storage_gb": ("acquisition", "storage_gb", "max_storage_gb"),
    "diagnostic_actions": ("diagnostics", "max_actions", "max_diagnostic_actions"),
    "diagnostic_minutes": ("diagnostics", "minutes", "max_diagnostic_minutes"),
    "diagnostic_total_minutes": ("diagnostics", "max_minutes", "max_diagnostic_total_minutes"),
    "decisions": ("diagnostics", "max_decisions", "max_decisions"),
}


def budget_values(cfg):
    return {key: getattr(getattr(cfg, group), field) for key, (group, field, _) in FIELDS.items()}


def budget_ceilings(cfg):
    # An explicitly larger initial user allowance is already authorized.
    return {
        key: max(budget_values(cfg)[key], getattr(cfg.autonomy, ceiling))
        for key, (_, _, ceiling) in FIELDS.items()
    }


def apply_update(base, current, update):
    if not base.autonomy.enabled:
        raise ValueError("Autonomous budget changes are disabled.")
    patch = BudgetUpdate.model_validate(update).model_dump(exclude_none=True)
    if not patch:
        raise ValueError("Specify at least one budget increase.")
    values, ceilings = budget_values(current), budget_ceilings(base)
    data = current.model_dump()
    for key, value in patch.items():
        if not values[key] <= value <= ceilings[key]:
            raise ValueError(
                f"{key} must stay between {values[key]} and its ceiling {ceilings[key]}."
            )
        group, field, _ = FIELDS[key]
        data[group][field] = value
    return Campaign.model_validate(data)


def effective_campaign(base):
    current = base.model_copy(deep=True)
    for record in read_json(base.output_dir / "budget_changes.json", []):
        if record["contract_hash"] != value_hash(base.model_dump(mode="json")):
            raise ValueError("Budget changes belong to a different campaign contract.")
        current = apply_update(base, current, record["update"])
    return current


def change_budget(base, decision_id, reason, update):
    path = base.output_dir / "budget_changes.json"
    records = read_json(path, [])
    patch = update.model_dump(exclude_none=True)
    previous = next((r for r in records if r["decision_id"] == decision_id), None)
    if previous:
        if previous["update"] != patch:
            raise ValueError("A saved budget decision changed.")
        return previous
    current = effective_campaign(base)
    changed = apply_update(base, current, patch)
    if budget_values(changed) == budget_values(current):
        raise ValueError("The requested budgets are already available.")
    record = {
        "decision_id": decision_id,
        "reason": reason,
        "contract_hash": value_hash(base.model_dump(mode="json")),
        "update": patch,
        "before": budget_values(current),
        "after": budget_values(changed),
    }
    save_json(path, [*records, record])
    return record


def trial_minutes(cfg, requested=None):
    minutes = cfg.budget.exploration_minutes if requested is None else requested
    ceiling = max(cfg.budget.exploration_minutes, cfg.autonomy.max_trial_minutes)
    if not cfg.autonomy.enabled:
        ceiling = cfg.budget.exploration_minutes
    if not 0 < minutes <= ceiling:
        raise ValueError(f"Training time must be positive and at most {ceiling:g} minutes.")
    return minutes
