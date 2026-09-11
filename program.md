# Instructions for the research agent

Read `EXPERIMENT.md`. Use the controller; own video acquisition and visual labeling
as well as experiment proposals. No human annotation is required.

Activate `.venv` and run **`python main.py research init`**. This starts or resumes
the entire workflow using the repository's `research.yaml`; if missing, defaults
are created. The controller checks Codex login before downloading or training.
Use `init --config custom.yaml --config-only` only to prepare settings without
starting work. Review budgets before the first launch.

The orchestrator chooses stages from saved state. Codex selects real videos from
search results, corrects image labels and proposes recipes from prior validation
metrics. The controller checks data coverage, enforces budgets, ranks candidates,
promotes the best recipe, trains seeds 0/1/2, and opens the final test only when all
three pass. It records decisions in `autonomous.json` and metrics in `results.csv`.
No user input is needed between stages.

With `diagnostics.enabled` and `proposals: codex`, direct exploration through the
persisted decision loop. Inspect its initial label/size/augmentation evidence,
then choose training, diagnostics, collection for any split, automatic label review,
budget adjustment, finishing exploration, or stopping.
Diagnose repeated zero ball scores: inspect train/val ball crops and object sizes,
compare full-training fit with validation at the same confidence, or request the crop
learning check. `evaluate_train` is a targeted sample and can be optimistic; use
`evaluate_train_full` before declaring full-data fit sufficient. Examine actual epochs,
learning rates, losses, and near-miss box overlaps before ruling out optimization,
resolution or capacity. Treat crop memorization as a learning diagnostic, never as evidence
of deployment accuracy. Read `decisions/` and `diagnostics/` for prior evidence;
avoid repeating answered questions. All diagnostic runtime counts against its
separate cap and the total compute allowance. Keep the test sealed.

With `data_growth.enabled`, use `collect` with split `train`, `val` or `test` to grow
coverage across independent matches. Use `review_labels` to correct annotations from
original images; no human curation is available. Review calls receive no student
predictions. Existing images and match assignments remain fixed; label corrections
preserve the original labels in a recoverable journal. Benchmark changes invalidate
old acceptance claims. Never reuse held-out matches for training or use student test
results to guide acquisition. Aim for substantially more than a handful of ball
labels from one validation match. All versions share campaign allowances. Inspect
`data_growth/` for search hypotheses and outcomes, and `dataset_versions/` for
manifests. Promotion/confirmation use the current version, and growth stops before
those stages; unsuccessful promotion/confirmation can return to exploration.
Use `adjust_budget` to increase working allowances within the explicit `autonomy`
ceilings. Every increase is audited in `budget_changes.json`; do not reset consumed
work or edit the contract. `train.minutes` can request a longer run. Registered
warm-start checkpoints continue learned weights with a fresh optimizer/schedule;
repeating settings identical to a completed trial on the current dataset version
is rejected unless every such trial stopped on its time budget. Recipes also
control augmentation (mosaic, close_mosaic, mixup, copy_paste, erasing, hsv,
geometry, flips) and optimization (lr0, lrf, weight_decay, warmup, patience,
dropout); `model` may be any `.pt`/`.yaml` in the model directories. The evidence
payload includes deterministic signals (best current trial, validation span,
plateau, repeated settings, train–validation gap) — trust them over re-deriving.
Read prior-campaign evidence instead of blindly repeating its failed baselines.

Keep the objective fixed: minimize student parameters subject to every macro and
ball threshold in `EXPERIMENT.md`. Extra accuracy does not outweigh smaller size
once both candidates pass. The configured annotation teacher is independent of the
N/S/M/L/X student search; never force the deployment model to match the teacher.
Teacher proposals still require agent correction. Do not relax quality gates.

Use Codex as the primary agent and the configured free Muse model through OpenCode
on quota exhaustion. The router handles cooldowns, switching back, and provenance;
do not switch models manually or replace the free model with a paid one. If both
providers are limited, the workflow waits within `fallback.max_wait_hours` and
retains completed work. Inspect `providers.json` and the workflow status for details.

Keep the foreground process running and the computer awake. After interruption,
run `init` again. Accepted labels and completed trials are retained. An interrupted
trial may get a new recorded attempt within the remaining budget. A terminal
stop requires a new campaign if further research is wanted. Before stopping,
consider useful budget extensions, full-data diagnostics, automatic label review,
evaluation coverage, longer optimization, and controlled capacity probes.

Report the selected checkpoint, recipe, macro/ball scores, support, parameters and
latency if measured. State that labels were generated by the agent without independent
ground truth. Training stopped by a time limit does not prove convergence.

Use `--config PATH` on each command for a non-default configuration. Use
`python main.py research status` to inspect progress. Do not bypass budgets,
erase interrupted attempts, use test predictions to tune models, or claim
football accuracy from software smoke tests. Expand data diversity and training
time through the audited actions when evidence supports it.
