#!/usr/bin/env bash
# Orchestrator: re-check labels for the football-astra-high dataset in parallel.
# Modes:
#   ./recheck_astra_high.sh pilot [N]   -> run a stratified pilot of N images (default 20)
#   ./recheck_astra_high.sh full        -> run all remaining (resumable) images
# Env:
#   PARALLEL (default 6)  concurrency
set -uo pipefail

REPO="/Users/jbilbao/Desktop/repositories/sportAnalytics"
DS="$REPO/datasets/football-astra-high"
WORK="$DS/.recheck"
WORKER="$REPO/scripts/recheck_worker.sh"
DONE_DIR="$WORK/done"
PARALLEL="${PARALLEL:-6}"

mkdir -p "$WORK" "$DONE_DIR"
chmod +x "$WORKER"

# Build the full image list: "split<space>image" (paths contain no spaces)
LIST="$WORK/all_images.txt"
: > "$LIST"
for s in train val test; do
  for img in "$DS/images/$s"/*.jpg; do
    [ -e "$img" ] || continue
    printf '%s %s\n' "$s" "$img" >> "$LIST"
  done
done
total=$(wc -l < "$LIST" | tr -d ' ')

# xargs -n 2 -> passes "<split> <img>" as two args to the worker
run_parallel() {
  xargs -P "$PARALLEL" -n 2 bash "$WORKER"
}

mode="${1:-pilot}"
case "$mode" in
  pilot)
    N="${2:-20}"
    PILOT="$WORK/pilot.txt"
    : > "$PILOT"
    for s in train val test; do
      cnt=$(grep -c "^$s " "$LIST")
      share=$(( N * cnt / total )); [ "$share" -lt 3 ] && share=3
      grep "^$s " "$LIST" | head -n "$share" >> "$PILOT"
    done
    pcount=$(wc -l < "$PILOT" | tr -d ' ')
    echo "PILOT: $pcount images (parallel=$PARALLEL)"
    run_parallel < "$PILOT"
    ;;
  full)
    PEND="$WORK/pending.txt"
    : > "$PEND"
    while read -r s img; do
      base="$(basename "$img" .jpg)"
      key="${s}__${base}"
      [ -f "$DONE_DIR/$key.ok" ] && continue
      printf '%s %s\n' "$s" "$img" >> "$PEND"
    done < "$LIST"
    pcount=$(wc -l < "$PEND" | tr -d ' ')
    echo "FULL: $pcount pending of $total (parallel=$PARALLEL)"
    run_parallel < "$PEND"
    ;;
  *)
    echo "usage: $0 [pilot [N] | full]"; exit 2;;
esac

# Final safety sweep: refresh any manifest hashes left stale by interrupted workers.
python3 "$REPO/scripts/reconcile_manifest.py" "$DS" || true

ok=$(ls "$DONE_DIR"/*.ok 2>/dev/null | wc -l | tr -d ' ')
echo "DONE markers: $ok / $total"
