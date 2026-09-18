#!/usr/bin/env bash
# Execute one repository-owned Python tool with source paths bound to this tree.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
TOOL=${1:?usage: run.sh experiments/TOOL.py or analysis/TOOL.py [arguments]}
shift
case "$TOOL" in
  experiments/*.py|analysis/*.py) ;;
  *) echo 'Expected experiments/TOOL.py or analysis/TOOL.py' >&2; exit 2 ;;
esac
case "/$TOOL/" in
  */../*|*/./*) echo 'Tool path must stay inside the package' >&2; exit 2 ;;
esac
SOURCE="$ROOT/vskipper/src/vskipper/$TOOL"
test -f "$SOURCE"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/vskipper/src:$ROOT/python:$ROOT/vskipper/src/vskipper/experiments:$ROOT/vskipper/src/vskipper/analysis"
exec "${VSKIPPER_PYTHON:-python3}" "$SOURCE" "$@"
