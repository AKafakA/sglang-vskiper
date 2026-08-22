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


# Validate inputs BEFORE creating the output directory. CONFIG defaults to
# test/vp/labeled_workload.example.json, which is not in this tree, and it was
# passed to every build_one call unchecked -- so the builder created OUT_DIR and
# then failed on the first workload, leaving a half-made suite behind.
# CONFIG must be a readable FILE and valid JSON, not merely something that
# exists: a directory passes -e, and a malformed config fails only once the
# first workload is already being written.
if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: CONFIG is not a file: $CONFIG" >&2
  exit 1
fi
if ! "$PYTHON_BIN" -c 'import json,sys; json.load(open(sys.argv[1]))' "$CONFIG" 2>/dev/null; then
  echo "ERROR: CONFIG is not valid JSON: $CONFIG" >&2
  exit 1
fi

for required in "$CONFIG" "$ROOT/test/vp/build_labeled_workload.py" \
                "$ROOT/test/vp/freeze_workload_suite.py"; do
  if [[ ! -e "$required" ]]; then
    echo "ERROR: required input does not exist: $required" >&2
    echo "       Set CONFIG=<path> explicitly; the default example config was" >&2
    echo "       dropped with the evaluation tooling (see the removed-feature" >&2
    echo "       register)." >&2
    exit 1
  fi
done

# Claim OUT_DIR ATOMICALLY. A separate [[ -e ]] test followed by mkdir -p is not
# atomic: two builders racing on the same absent path both pass the test and
# both proceed, and whichever fails first would then delete the other's suite
# through its EXIT trap. Plain mkdir fails if the directory exists, so exactly
# one process can own it -- and the cleanup trap is installed only AFTER this
# process is the owner, so it can never remove a directory it did not create.
if ! mkdir "$OUT_DIR" 2>/dev/null; then
  if [[ -e "$OUT_DIR" ]]; then
    echo "ERROR: refusing to reuse OUT_DIR=$OUT_DIR" >&2
  else
    echo "ERROR: could not create OUT_DIR=$OUT_DIR" >&2
  fi
  exit 1
fi

# Any failure past this point leaves a partial suite behind, and OUT_DIR refuses
# reuse -- so the next attempt would need manual cleanup before it could start.
# Remove the half-made directory on a non-zero exit instead. Success clears the
# trap, so a completed suite is never touched.
cleanup_partial() {
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "ERROR: build failed (rc=$rc); removing partial $OUT_DIR" >&2
    rm -rf "$OUT_DIR"
  fi
  exit $rc
}
trap cleanup_partial EXIT

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

trap - EXIT   # the suite is complete; keep it
