# Automatic football detection research

Find the smallest **tested** YOLO model that detects players and footballs at the
quality thresholds below. Codex owns video discovery, downloads, image labeling
and experiment proposals. No human annotation is required.

The implemented controller in `src/research/` uses Ultralytics/PyTorch for training
and the signed-in Codex CLI for visual annotation and structured proposals. It uses
the propose/train/evaluate/keep pattern; upstream autoresearch is not a dependency.

## Start

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py research init
```

`research.yaml` is included at the repository root. **`init` starts the entire
autonomous workflow**, using the existing configuration or creating defaults when
the file is missing. It checks Codex login, collects/labels data, explores recipes,
promotes a candidate, trains confirmation seeds, and performs the final test when
quality permits. No additional stage commands or human labeling are required.

Leave the terminal running and the computer awake. Ctrl-C interrupts the workflow;
run the same command again to resume. This is a local foreground process, not a
background service. Inspect progress from another terminal with
`python main.py research status`. `python main.py research auto` also resumes an
existing configuration. A finished or budget/quality-stopped workflow reports its
saved outcome instead of starting another campaign.

Edit budgets and search settings before first launch; the first campaign operation
freezes the configuration. To generate a different configuration without starting,
use `python main.py research init --config custom.yaml --config-only`; this option
refuses to overwrite an existing file. Pass `--config PATH` on later commands to
use it. Paths resolve relative to the configuration file.

### How decisions are made

The Python orchestrator in `src/research/autonomous.py` selects the next stage from
saved state and measured results. Codex makes the choices within that stage through
the prompts in `src/research/agent.py` and `src/research/acquisition.py`. `program.md`
documents the protocol; it is not a script that runs itself.

| Condition | Automatic action |
| --- | --- |
| No adequate dataset | Search/select videos, download, sample and let Codex label images; retry within acquisition limits |
| Dataset passes integrity and coverage checks | Freeze it and run the baseline recipes |
| Exploration has trials available | Give Codex prior trial metrics and quality targets; validate and execute its next recipe |
| Exploration finishes with a candidate | Promote the best candidate with a 30-minute budget |
| Promotion finishes with a candidate | Train the selected recipe with seeds 0/1/2 and longer budgets |
| All three confirmation seeds pass | Evaluate seed 0 once on test and report the artifact |
| Budgets exhausted or confirmation fails | Stop with a saved reason and preserve results; leave test unopened unless already finalized |

The agent's experiment hypotheses, prompts and responses are saved alongside trial
results. Python computes rankings and enforces the fixed gates. A feasible model
always ranks ahead of an infeasible one; among feasible models, fewer parameters
are preferred. Autonomy does not guarantee that the pilot data and budget can
produce a model meeting every quality target.

Accepted images and completed trials are not repeated. Autonomous resume can
launch a fresh attempt for an interrupted trial, retaining and charging the old
attempt against the total budget. The individual `acquire`, `run --phase ...`, and
`finalize` commands remain available for debugging, but are unnecessary for normal use.
A process lock prevents concurrent controllers for the same campaign. `status`
is safe during a run. Follow detailed output in each operation's `run.log` or
`agent.log`; `results.csv` and `state.json` track trials. `autonomous.json` records
the current stage, completed phases, acquisition rounds, decisions and final outcome.

### Automatic provider fallback

Codex is preferred. If its CLI reports a usage/quota limit, the pending request is
sent to **`opencode/muse-spark-1.3-contributor-free`** through OpenCode. The model ID
matches the [OpenCode Zen catalog](https://opencode.ai/docs/zen/); the local catalog
and a live image-annotation smoke test confirmed image input support. Executables
added by interactive terminal startup (such as nvm installations) are discovered
automatically, so the IDE's older PATH does not need to be edited.

`fallback.codex_retry_seconds` defaults to 300: at a subsequent request boundary
after that cooldown, try Codex first and switch back on success. An earlier reset
timestamp from a structured quota error can shorten the cooldown. This is periodic
recovery, not an instantaneous background switch. `fallback.opencode_retry_seconds`
defaults to 60. Authentication failures, malformed output and ordinary network
errors remain visible; they do not masquerade as quota exhaustion.

Only the configured free OpenCode model is selected, including auxiliary requests.
Startup checks its catalog pricing and image capability. No paid model is substituted.
Free availability can change; the catalog check is not a guarantee of future service
availability. The Contributor Free service may use submitted prompts and completions
for model improvement; see the provider's [privacy notes](https://opencode.ai/docs/zen/#privacy).

If both providers hit limits, `init`/`auto` persist progress and wait, then retry.
Accepted images and completed training trials remain intact, and quota events do
not consume an image's annotation retries. Waiting is separately capped by
`fallback.max_wait_hours` (24 hours cumulative by default), with each sleep reserved
in the persisted workflow state. On exhaustion the workflow stops with a reason.
Quota recovery resumes acquisition in its existing round; the resumed invocation
gets a fresh acquisition time window. Download, source and image allowances still
persist. Individual debugging commands surface quota unavailability rather than wait.

`providers.json` records cooldowns and provider changes, and each successful request
has a `provider.json` identifying its provider/model and raw attempt directory.
These records also survive restarting the controller. Prompts and label validation
use the same taxonomy/schema across both providers. Record mixed provider provenance
when interpreting label quality. Set `fallback.enabled: false` for Codex-only use.

## Hardware and budgets

Inspected September 7, 2026: MacBook Pro, Apple M5 Pro, 18 CPU cores, 64 GB unified
memory, macOS 26.6.2, PyTorch 2.14.0 with MPS available. Defaults are `device: mps`,
two data workers, batch four, one trial at a time. These are starting settings;
software smoke tests do not establish football accuracy or sustained throughput.

| Stage | Default limit |
| --- | --- |
| Acquisition round | 30 minutes; at most three automatic rounds per campaign |
| Source downloads / dataset storage | 10 GB cumulative / 20 GB |
| Annotation | 120 seconds per image; two attempts |
| Missing starting checkpoint preparation | Five minutes per checkpoint |
| Exploration | 15 minutes per trial; eight attempts, seed 0 |
| Promotion | 30 minutes, seed 0 |
| Confirmation | 120 minutes per run, up to 200 epochs, seeds 0/1/2 |
| All model trials | 20 attempts; 10 hours actual runtime |
| Final test | 30 minutes; one attempt |
| Waiting for provider quota recovery | 24 hours cumulative; separate from model-trial budgets |

Trial deadlines include startup, training, evaluation and saving. The worker
requests an early training stop with an evaluation reserve (normally 120 seconds);
the controller terminates its process group at the hard deadline. Incomplete
evaluation is an invalid attempt, never a zero score. After a controller crash,
the unfinished attempt is charged its full reserved time.

Acquisition, Codex proposals, checkpoint preparation and final testing are separate
from the aggregate model-trial budget. Acquisition time resets per invocation;
download bytes and retry counts persist. Downloads fetch the whole video before
sampling the chosen segment. Disk limits are polled and can slightly overshoot;
they measure local files, not exact network traffic. Staging frames stay for resume.

Longer confirmation is configurable before starting. If learning is still improving
at its deadline, use a follow-up campaign with adequate training time. A time-limited
run does not prove convergence.

## Agent-owned dataset

1. Search YouTube when `acquisition.sources` is empty. Choose identifiable distinct
   matches, wide gameplay views and short useful segments. Expand coverage across
   camera distances, ball sizes, occlusion, lighting and weather (snow, rain, sunshine).
2. Assign entire matches to train, validation or test before sampling. Different
   edits/views of the same match must share `match_id` and split. Omit ambiguous
   compilations. Match identity depends on the agent's metadata assessment.
3. Download with the existing yt-dlp/Deno implementation, uniformly sample frames
   and generate YOLO proposals. Defaults are six sources and twelve frames per
   source: a pipeline pilot, not a sufficient final dataset.
4. Send each original image plus proposals to Codex. The agent adds missed objects,
   corrects boxes/classes and removes false detections. It returns an explicit
   accept/reject decision and reason. Proposals are suggestions, never final labels.
5. Save accepted YOLO labels with prompts, structured responses, validated decisions,
   image/label hashes, source URLs, match IDs and timestamps. Empty labels require
   the agent to report no target objects. Reject ambiguous/unresolved frames.
6. Audit and freeze the dataset before training. Reject match overlap, exact
   duplicates across splits, changed labels, mismatched paths, missing/unmanifested
   files and unsupported classes. Near-duplicate detection and independent match
   identity verification are not implemented.

Default taxonomy: class 0 includes active players and goalkeepers; class 1 is the
match football. Referees, staff and spectators are excluded. Change names, taxonomy
and ball class ID before starting if needed. Every split must support every class.

The default annotation teacher is **YOLOv8-L at 1280 pixels**, configured separately
as `acquisition.teacher` and `acquisition.teacher_imgsz`. Use `models/yolov8x.pt`
before starting if an extra-large teacher is preferred. Missing standard checkpoints
download on first use. L and X are supported pretrained detection variants
([Ultralytics reference](https://docs.ultralytics.com/models/yolov8/)). A larger
teacher is a proposal source, not a guarantee of football label quality: the agent
still corrects its detections. Teacher size is not part of the student ranking,
and teacher predictions are not used as a distillation loss during training.

Optional explicit sources can be URLs or local files:

```yaml
acquisition:
  sources:
    - id: match-a
      url: /absolute/path/match-a.mp4
      match_id: match-a
      split: train
      start_minutes: 5
      end_minutes: 7
    # Supply distinct matches for val and test too.
```

Pilot minimums are one match and one ball instance per evaluation split. For a
meaningful later benchmark, raise `evaluation.min_matches` and
`evaluation.min_ball_instances`, expand source diversity/size, and create a new
versioned campaign. If coverage is incomplete, the autonomous workflow reruns
acquisition within its round/retry/download allowances. A source plan
that cannot supply adequate data needs a revised campaign. Never lower support
gates or call teacher output reviewed just to get training started.

Evaluation references are **agent-generated labels**, not independently verified
ground truth. Scores measure agreement with those labels. Shared teacher/agent
errors can inflate scores, especially for tiny balls. Every evaluation records
`independent_ground_truth: false`. Fully automatic annotation does not by itself
prove real-world detection accuracy.

## Fixed evaluator

| Metric | Macro across target classes | Ball alone |
| --- | --- | --- |
| AP50–95 | > 0.70 | > 0.70 |
| AP50 | > 0.85 | > 0.85 |
| Precision | > 0.90 | > 0.90 |
| Recall | > 0.90 | > 0.90 |
| F1 | > 0.85 | > 0.85 |

Use strict comparisons on unrounded values. Missing/invalid metrics cannot pass.
Report every class, support and false positives/negatives. Macro averages weight
classes equally. Precision/recall/F1 use IoU 0.5; AP uses IoUs 0.50–0.95 in steps of
0.05 and Ultralytics AP integration. Matching is class-aware, confidence-ordered,
one-to-one greedy matching. Evaluator version:
`confidence-greedy-v1-ultralytics-ap`; this is not claimed to reproduce Ultralytics'
internal validation matcher exactly.

Inference uses confidence floor 0.001, class-aware NMS IoU 0.7, maximum 300 detections
and no test-time augmentation. Resolution belongs to the recipe. Select one global
confidence from 0.01–0.99 on validation: maximize the worst normalized macro/ball
precision/recall, then macro F1, then prefer the lower confidence. Freeze that value
for the final test. No per-class or test-derived threshold tuning.

Among feasible candidates prefer fewer model parameters (counted before prediction
layer fusion), then lower measured p95 latency, then smaller checkpoints. Among
infeasible candidates rank the worst quality-to-threshold ratio, then ball AP50–95.
An infeasible candidate never replaces a feasible incumbent. Exploration and
promotion have separate incumbents because their training budgets differ.

Latency is optional, disabled by default. Set `benchmark_frames` to at least 200
and `max_p95_ms` to enable its acceptance gate. The benchmark warms up for 50
frames, cycles through split images and synchronizes the device. Batch-one predict
timing includes preprocessing/NMS and excludes image decoding. It reports p50/p95;
a pilot with few unique frames is not a deployment benchmark.

## Search space and records

Codex proposes validated recipe fields: starting checkpoint, image size, batch,
epochs, learning rate, optimizer, mosaic, scale and rotation. Proposals may use
only checkpoint paths in the original recipe list. Defaults compare YOLOv8n at
640 and 960, followed by YOLOv8s/m/l/x at 640. These six baseline trials leave two
agent-proposed trials within the default eight-trial exploration budget. The agent
can tune any of those student sizes; teacher selection does not restrict them.
The objective is minimum measured student parameters subject to **every** fixed
macro/ball gate, not maximum accuracy regardless of size. Extra accuracy never
allows a larger feasible model to outrank a smaller feasible one. This finds the
smallest passing model tested in the configured search space and budget, not a
proof that no smaller architecture or longer-trained candidate could work.
`proposals: queue` runs only configured recipes; annotation still uses Codex.
Arbitrary architecture/code mutation is outside this controller's current scope.

Code, relevant dependency versions, initial checkpoint hashes, campaign settings
and dataset snapshot must remain fixed. Each trial saves its job, logs, platform/
dependency details, completed epochs, predictions, metrics and checkpoint hash.
Start a new campaign for implementation, dataset or budget changes.

## Confirmation and final artifact

Promotion selects the best explored recipe. Confirmation selects the best promoted
recipe if available, otherwise the best explored one, and trains seeds 0/1/2 from
the same initialization. All three must pass validation. Select seed 0 of the best
eligible recipe for final testing, rather than the highest-scoring seed.

`finalize` evaluates that checkpoint on test at its saved validation confidence.
It records the attempt before launching and never repeats it in the same campaign,
even after a crash. Further training is blocked after test is opened. The normal
proposal loop receives no test predictions. This protocol is not a filesystem
security boundary against directly reading files or changing the implementation.
Do not tune a follow-up model on an exposed test split.

A pass is named `passed_against_agent_labels`. The final record identifies the
selected trial; that trial's metrics contain the checkpoint path. The final
`job.json` preserves resolution and confidence. Keep the checkpoint, taxonomy and
preprocessing settings together. No ONNX, pruning or quantization is involved.

## Verification

```bash
python -m pytest -q
python -m ruff check main.py src tests
python -m ruff format --check main.py src tests
# Optional network checks:
RUN_CODEX_SMOKE=1 python -m pytest -q tests/test_research.py -k live_codex
RUN_OPENCODE_SMOKE=1 python -m pytest -q tests/test_research.py -k live_opencode
RUN_YOUTUBE_SMOKE=1 python -m pytest -q tests/test_youtube_live.py
```

Tests cover resume, selection, budgets, process termination, leakage/mutation
checks, agent-corrected labels, metric matching, final-test behavior and real CPU
training/checkpoint/evaluation. Live Codex and YouTube checks use network resources.
Provider tests simulate quota exhaustion, fallback, return to Codex, waiting and
retry accounting. Successful smoke tests establish execution, not a trained football
model's quality.
