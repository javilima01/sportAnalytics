# Image tagging and checking

Tools for creating and reviewing YOLO bounding-box labels, with training support
for the resulting dataset.

- `generate`: sample frames from a local video or YouTube URL and create initial
  labels using a YOLO `.pt` checkpoint.
- `visualize`: inspect labeled images, save previews, or edit bounding boxes.
- `train`: fine-tune a YOLO model on the reviewed dataset and validate it.
- `research`: discover/download videos, let Codex label images, and run bounded,
  resumable YOLO experiments with a fixed evaluator.

## Setup

Use Python 3.10 or newer (tested here with Python 3.14).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Interactive viewing and editing require a desktop with OpenCV GUI support.
YouTube downloads use `yt-dlp` with its JavaScript support and the Deno runtime,
all included in the requirements. The downloader selects a video-only MP4 when
available, so audio merging and FFmpeg are unnecessary for this workflow.

## Generate initial labels

```bash
python main.py generate --model models/best.pt --video match.mp4 --output datasets/football --sample_prob 0.1 --segments 5:10 15:25
```

`--segments` uses minutes; omit it to process the whole video. `--video` also
accepts a YouTube URL. By default, only sampled frames with detections are saved.
Use `--include_empty` to retain frames for manual tagging or background examples,
and `--conf 0.4` to adjust the confidence threshold. Review automatic labels
before training.

The generated dataset contains `data.yaml`, `images/train`, `images/val`,
`images/test`, and matching `labels/` directories. Each label file uses normalized
YOLO detection coordinates: `class_id center_x center_y width height`.
The default train/validation/test split is `0.7 0.2 0.1`; override it with
`--splits`. Ratios must be nonnegative and sum to one. `--seed` controls repeatable
sampling and splitting. Overlapping segments are merged, and reruns skip existing
image/label pairs so reviewed work is preserved. Source paths/URLs identify videos;
moving a source or changing its URL creates a new identity. Datasets made by the
older filename scheme should use a new output directory when generating more frames.

For meaningful evaluation, consider keeping whole videos in separate splits;
randomly splitting adjacent frames can make validation results overly optimistic.

## Check and edit images

```bash
# Review a random sample, or save annotated previews with --save_dir previews
python main.py visualize --dataset datasets/football --split train --max_images 20

# Edit all images in a split
python main.py visualize --dataset datasets/football --split train --edit
```

The viewer supports both `images/<split>` / `labels/<split>` and
`<split>/images` / `<split>/labels` layouts. Existing datasets should include a
`data.yaml` with class names as a list (`names: [player, ball]`) or numeric mapping:

```yaml
names:
  0: player
  1: ball
```

Editor controls:

- Left-drag on empty space: add a box, initially class `0`.
- Left-click a box: select it; `C` cycles its class.
- Middle-click or Ctrl-left-click a box, or `X` on a selected box: delete it.
- `S`: save labels.
- Left/right arrow or `P`/`N`: navigate, saving edits to the current image.
- `D`: permanently delete the current image and its label file.
- Esc or closing the window: save edits and exit.

To replace a box, delete it and draw a new one. The editor can add labels to
images without a label file; label directories are created when saving. `S` also
saves an empty label file to mark an image as background. Malformed labels raise
an error instead of being silently dropped. JPG/JPEG, PNG, BMP, and WebP images
are supported, including uppercase extensions.

## Recheck all labels

Re-check every train/val/test image in a dataset in parallel and correct its
YOLO labels. The run is resumable: completed images are recorded under
`<dataset>/.recheck/done` and skipped on rerun. Use `pilot [N]` instead of `full`
to review a small stratified sample first.

```bash
cd /Users/jbilbao/Desktop/repositories/sportAnalytics
PARALLEL=3 nohup bash scripts/recheck_astra_high.sh full >> datasets/football-astra-high/.recheck/full_run.log 2>&1 &
```

## Train on reviewed labels

```bash
python main.py train --data datasets/football/data.yaml --model yolov8x.pt --epochs 50 --device cpu
```

Omit `--device` to let Ultralytics select a device, use `--device 0` for a CUDA GPU,
or `--device mps` for supported Apple hardware. Training retains `.pt` checkpoints in
the run's `weights/` directory, normally under
`training/finetune/yolov8x_football`. Existing runs receive a new directory.
The wrapper writes final training/validation metrics into that run's `tensorboard/`
directory. Advanced model and augmentation settings use Ultralytics defaults;
the application configuration only declares the options it controls.

Run `python main.py <command> --help` for the available options.

## Automatic research with Codex

With the Codex CLI installed and signed in:

```bash
python main.py research init
```

This single command starts the complete autonomous workflow: acquisition, labeling,
exploration, promotion, confirmation and final testing. It uses the included
[research.yaml](research.yaml), or creates defaults if that file is missing.
Edit it before first launch to choose budgets, source queries and checkpoints.
The agent discovers videos when no URLs are
supplied and corrects every accepted image's labels; no human labeling is required.
Defaults use this Mac's MPS device, 15-minute exploration trials and a pilot dataset.
The agent can request longer training, start from saved checkpoints, and increase working
budgets within the `autonomy` ceilings. Changes persist in `budget_changes.json`.

With diagnostics enabled, the agent chooses after each action whether to train,
evaluate full training fit, collect new train/validation/test matches, review labels
automatically, or adjust budgets. Dataset and benchmark changes are versioned;
original labels are preserved in review journals. Promotion and confirmation require
current-version results. Student test predictions remain unavailable until final testing.

For this Mac's prepared configuration, run:

```bash
caffeinate -i python main.py research init --config research-m5.yaml
```

It starts `football-m5-autonomous`, reuses the existing dataset, and reads the previous
M5 campaign as historical evidence and a source of warm-start checkpoints. Typical
trials start at 30 minutes; the agent can extend individual training runs to four hours
and campaign trial/diagnostic compute to 72 hours when justified. These are ceilings,
not durations that every experiment will consume. Use the same `--config` for status.

Keep the terminal open and the computer awake. Run the same `init` command to
resume after interruption; use `python main.py research status` from another
terminal to see progress. To create a configuration without starting work, use
`python main.py research init --config custom.yaml --config-only`.

Startup checks the primary CLI login before downloading: Codex uses its
[documented authentication status command](https://learn.chatgpt.com/docs/auth),
and OpenCode validates its model catalog instead.

Either provider can be the primary agent or the fallback: the provider is inferred
from the executable name (`codex` or `opencode`), and an explicit `provider:` key
overrides that inference. Each slot has its own model and reasoning setting
(`codex_reasoning_effort` for the primary, `fallback.reasoning_effort` for the
fallback); Codex receives `model_reasoning_effort` and OpenCode a model variant.
If the primary reaches its usage limit, requests automatically switch to the
fallback, which is retried at request boundaries after its cooldown. Both providers
use the same image-label and experiment schemas. The controller finds executables
through your interactive terminal PATH, checks OpenCode models for image support
(warning when the catalog lists a non-zero cost), and records provider changes.
If all configured providers hit limits, it saves progress and waits within the
separate `fallback.max_wait_hours` allowance.

See [EXPERIMENT.md](EXPERIMENT.md) for configuration, quality gates and limitations,
and [program.md](program.md) for agent instructions. Scores measure agreement with
agent-generated labels, not independently verified ground truth.

## Verification

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check main.py src tests
python -m ruff format --check main.py src tests
```

The suite covers label persistence, editor events, video generation, input validation,
and a real CPU inference/training/validation/checkpoint cycle on synthetic images.
It does not download pretrained weights. Run `python -m pytest -q -m 'not integration'`
for only the faster tests. The synthetic training check verifies execution, not accuracy.
YouTube download failures and cleanup are tested without contacting YouTube.

The live YouTube workflow was also verified on September 7, 2026 using
`https://www.youtube.com/watch?v=jNQXAC9IVRw` and `yolov8n.pt`: download, frame
extraction, nonempty detection labels, saved previews, and temporary-video cleanup
all passed. To repeat this opt-in network test (downloads weights if missing):

```bash
RUN_YOUTUBE_SMOKE=1 python -m pytest -q tests/test_youtube_live.py
```

An optional desktop smoke test opens the real editor, draws a box, saves, and closes:

```bash
RUN_GUI_SMOKE=1 python -m pytest -q tests/test_gui_smoke.py
```
