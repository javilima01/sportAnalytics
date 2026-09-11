#!/usr/bin/env bash
# Per-image worker: re-check and correct one YOLO label via the image-tagger subagent.
# Usage: recheck_worker.sh <split> <image_abs_path>
set -uo pipefail

REPO="/Users/jbilbao/Desktop/repositories/sportAnalytics"
DS="$REPO/datasets/football-astra-high"
WORK="$DS/.recheck"
DONE_DIR="$WORK/done"
LOG_DIR="$WORK/logs"
mkdir -p "$DONE_DIR" "$LOG_DIR"

split="$1"
img="$2"
base="$(basename "$img" .jpg)"
key="${split}__${base}"
lbl="$DS/labels/$split/$base.txt"
scratch="$WORK/scratch"
mkdir -p "$scratch"

# Resumable: skip if already completed
if [ -f "$DONE_DIR/$key.ok" ]; then
  echo "SKIP $key"
  exit 0
fi

# macOS has no `timeout`; use perl alarm as the per-image cap (seconds)
CAP="${CAP:-300}"

prompt="Re-check and correct the YOLO labels for this single football image by inspecting its actual pixels.
Class names: 0=player, 1=ball.
Image path: $img
Write the corrected label file to EXACTLY this path (overwrite it): $lbl
Use YOLO format, normalized coordinates in [0,1], one line per object: <class_id> <cx> <cy> <w> <h>.
Give a tight box around every clearly visible player and the ball. Fix wrong, missing, or extra boxes versus the existing label. If there are no target objects, write an empty file. Do not invent classes.
Be efficient: finish in a few analysis steps, do NOT iteratively crop dozens of regions. If you use Python/PIL, write any temporary files ONLY inside this workspace scratch dir: $scratch (never /tmp)."

log="$LOG_DIR/$key.log"
perl -e 'alarm shift; exec @ARGV' "$CAP" \
  opencode run --agent image-tagger --dir "$REPO" "$prompt" -f "$img" >"$log" 2>&1
rc=$?

# Keep manifest hashes in sync even when a run times out after writing labels.
python3 "$REPO/scripts/reconcile_manifest.py" "$DS" "labels/$split/$base.txt" >>"$log" 2>&1 || true

if [ "$rc" -eq 0 ]; then
  : > "$DONE_DIR/$key.ok"
  nlines=$(wc -l < "$lbl" 2>/dev/null | tr -d ' ')
  echo "OK   $key (label_lines=${nlines:-NA})"
else
  echo "ERR  $key (exit=$rc, see $log)"
fi
