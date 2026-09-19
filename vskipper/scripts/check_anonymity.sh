#!/usr/bin/env bash
# Check every tracked file, including this checker. Keep identifying patterns
# outside the repository and outside every reviewer export.
# Usage: bash vskipper/scripts/check_anonymity.sh --terms-file /private/terms.tsv [--list]
# Each non-comment line is LABEL<TAB>EXTENDED_REGULAR_EXPRESSION.
# Exit 0: clean; 1: identifying content; 2: invalid input or incomplete scan.
set -euo pipefail
terms_file=
show_matches=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --terms-file)
      [ "$#" -ge 2 ] || { echo 'Missing --terms-file argument' >&2; exit 2; }
      terms_file=$2; shift 2 ;;
    --list) show_matches=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$terms_file" ] && [ -f "$terms_file" ] || {
  echo 'Provide --terms-file with an external private TSV file.' >&2; exit 2;
}
terms_file=$(realpath -- "$terms_file")
repo_root=$(git rev-parse --show-toplevel) || exit 2
repo_root=$(realpath -- "$repo_root")
case "$terms_file" in
  "$repo_root"/*) echo 'The private terms file must be outside the repository.' >&2; exit 2 ;;
esac
cd -- "$repo_root"
rules=0
total=0
while IFS=$'\t' read -r label pattern || [ -n "$label$pattern" ]; do
  case "$label" in ''|'#'*) continue ;; esac
  [ -n "$pattern" ] || { echo "Missing pattern for $label" >&2; exit 2; }
  rules=$((rules + 1))
  status=0
  matches=$(git grep -lIE -- "$pattern" -- .) || status=$?
  [ "$status" -le 1 ] || { echo "Scan failed for $label" >&2; exit 2; }
  file_status=0
  name_matches=$(git ls-files | grep -E -- "$pattern") || file_status=$?
  [ "$file_status" -le 1 ] || { echo "Filename scan failed for $label" >&2; exit 2; }
  if [ -n "$matches$name_matches" ]; then
    total=$((total + 1))
    printf 'FOUND: %s\n' "$label"
    if [ "$show_matches" -eq 1 ]; then
      [ -z "$matches" ] || printf '%s\n' "$matches"
      [ -z "$name_matches" ] || printf '%s\n' "$name_matches"
    fi
  fi
done < "$terms_file"
[ "$rules" -gt 0 ] || { echo 'The terms file contains no rules.' >&2; exit 2; }
if [ "$total" -gt 0 ]; then
  printf 'BLOCKED: %s identifying rule(s) matched. Do not publish.\n' "$total"
  exit 1
fi
printf 'CLEAN: all tracked filenames and text content checked against %s private rules.\n' "$rules"
