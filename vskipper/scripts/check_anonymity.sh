#!/usr/bin/env bash
# Blocking gate: refuse a release push while identifying strings remain in TRACKED FILE CONTENT.
#
# Scope is deliberate. The anonymised branch is served by anonymous.4open.science, which publishes a
# static view of the FILE TREE and not git metadata, so commit authorship is out of scope and stays as
# it is. What a reviewer can read is the content of the files, and that is what this checks.
#
#   bash vskipper/scripts/check_anonymity.sh            # scan tracked files, exit 1 on any hit
#   bash vskipper/scripts/check_anonymity.sh --list     # print every hit with file and line
#
# Exit 0 means clean. Exit 1 means at least one term was found; the push must not proceed.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
LIST=0; [ "${1:-}" = "--list" ] && LIST=1

# Each entry is "label<TAB>extended-regex". Add a term here rather than in a caller.
TERMS=$(cat <<'EOT'
username	wd312
host alias	3090-vast|dev-gpu-wd312|openclaw|gxp-l40s
cluster	[Cc][Ss][Dd]3|login-icelake|hpc\.cam\.ac\.uk
account code	KALYVIANAKI[A-Z0-9-]*
absolute home	/home/wd312|/rds/user/[a-z0-9]+
institution	cam\.ac\.uk|cl\.cam\.ac\.uk
EOT
)

total=0
while IFS=$'\t' read -r label re; do
  [ -z "$label" ] && continue
  files=$(git grep -lIE "$re" -- . 2>/dev/null | grep -v '^vskipper/scripts/check_anonymity.sh$' || true)
  n=$(printf '%s' "$files" | grep -c . || true)
  printf '%-22s %4s file(s)\n' "$label" "$n"
  total=$((total + n))
  if [ "$LIST" = 1 ] && [ "$n" != 0 ]; then
    git grep -nIE "$re" -- . 2>/dev/null | grep -v '^vskipper/scripts/check_anonymity.sh:' | sed 's/^/    /'
  fi
done <<< "$TERMS"

echo
info_ids=$(git grep -lIE "\bD-[0-9]{3}\b" -- . 2>/dev/null | grep -v "^scripts/check_anonymity.sh$" | grep -c . || true)
printf '%-22s %4s file(s)  (internal decision ids: bookkeeping references, not identifying; informational only)\n' "decision id (info)" "$info_ids"
if [ "$total" = 0 ]; then
  echo "CLEAN: no identifying string in tracked file content."
  exit 0
fi
echo "BLOCKED: $total file-hits across the terms above. Do not push."
echo "Run with --list to see them. Host files belong in deploy/hosts/*.json.example with"
echo "placeholders; absolute paths belong behind a configurable root; decision ids belong in"
echo "docs/vskipper/decisions.md as restated rulings, keeping the reasoning in the comment."
exit 1
