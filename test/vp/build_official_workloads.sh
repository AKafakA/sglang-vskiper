#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:?set OUT_DIR to a new workload directory}"
CONFIG="${CONFIG:-$ROOT/test/vp/labeled_workload.example.json}"
# Uniform generation policy. It is a recorded builder input: the value lands in
# every request body, in every workload summary, and in the config identity
# hash, so a rebuild that forgets it produces different hashes instead of a
# silently penalty-free suite.
FREQUENCY_PENALTY="${FREQUENCY_PENALTY:-0.0}"

if [[ -e "$OUT_DIR" ]]; then
  echo "ERROR: refusing to reuse OUT_DIR=$OUT_DIR" >&2
  exit 1
fi
mkdir -p "$OUT_DIR"

build_one() {
  local workload="$1"
  local count="$2"
  shift 2
  "$PYTHON_BIN" "$ROOT/test/vp/build_labeled_workload.py" \
    --workload "$workload" \
    --workload-config "$CONFIG" \
    --frequency-penalty "$FREQUENCY_PENALTY" \
    --num-requests "$count" \
    --requests "$OUT_DIR/$workload.requests.jsonl" \
    --metadata "$OUT_DIR/$workload.metadata.jsonl" \
    --summary "$OUT_DIR/$workload.summary.json" \
    "$@"
}

# Counts cover the largest approved 180-second vanilla calibration cell.
build_one gsm8k 5760 --mode decode --datasets gsm8k --repeat-exhausted
build_one coqa 23040 --mode decode --datasets coqa --repeat-exhausted
build_one decode_mix 11520 --mode decode --repeat-exhausted
build_one mixed 11520 --mode mixed --repeat-exhausted

"$PYTHON_BIN" "$ROOT/test/vp/freeze_workload_suite.py" \
  --workload-dir "$OUT_DIR" \
  --output "$OUT_DIR/workload_suite_manifest.json"
