#!/usr/bin/env python3
"""Drop internal decision identifiers from COMMENT LINES, keeping the reasoning they annotate.

The release branch ships without the decision log, so a bare `D-nnn` in a comment points at
something the reader cannot follow. The reasoning beside it is worth keeping -- reviewers valued it --
so this removes only the token and the punctuation that carried it, never the sentence.

DELIBERATELY NOT TOUCHED: `"decision": "D-nnn"` fields. Those are DATA in `design.py` and its
conformance test, recording which ruling authorised a non-default launch value, and the attestation
compares them. Changing one changes an attested field, so it waits for a run that can re-verify the
golden attestation. The script reports them instead.

usage: scrub_decision_ids.py [--apply]     (default is a dry run)
"""
import argparse, pathlib, re, subprocess, sys

ID = r"D-\d{3}"
# Ordered: the most specific shape first, so a bracket is not left half-emptied.
RULES = [
    (re.compile(r"\[Audit " + ID + r"(?: #\d+)?\]"), "[audit]"),
    (re.compile(r"\[" + ID + r"(?: add\.? ?\d*)?\]\s*"), ""),
    (re.compile(r"\((?:owner )?" + ID + r"(?: #\d+)?\)\s*"), ""),
    (re.compile(r"\(" + ID + r"(?: #\d+)?, ?"), "("),
    (re.compile(r",? ?" + ID + r"(?: #\d+)?\)"), ")"),
    (re.compile(r",? ?" + ID + r"(?: #\d+)?[;:]"), ":"),
    (re.compile(r"\s*\b" + ID + r"\b"), ""),
]
KEEP = re.compile(r'"decision"\s*:\s*"' + ID + r'"')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    # git grep uses POSIX ERE, which has no \d.
    files = subprocess.run(["git", "grep", "-lIE", r"\bD-[0-9]{3}\b", "--", "."],
                           capture_output=True, text=True).stdout.split()
    files = [f for f in files if f != "scripts/scrub_decision_ids.py"]
    changed = kept = 0
    for f in files:
        p = pathlib.Path(f)
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        out = []
        for line in text.split("\n"):
            # COMMENT LINES ONLY. A first attempt ran over every line and broke three files: the
            # rules fired inside docstrings and string literals, leaving "(: layout" where a
            # parenthetical had been. Narrowing the scope is the fix; a tidy-up pass over code is not.
            stripped = line.lstrip()
            if not (stripped.startswith("#") or stripped.startswith("*") or '"_comment"' in line):
                out.append(line)
                continue
            if KEEP.search(line):
                kept += 1
                out.append(line)
                continue
            new = line
            for rx, sub in RULES:
                new = rx.sub(sub, new)
            out.append(new.rstrip() if new != line else line)
        new_text = "\n".join(out)
        if new_text != text:
            changed += 1
            if args.apply:
                p.write_text(new_text, encoding="utf-8")
    print(f"files with an id: {len(files)}")
    print(f"files {'changed' if args.apply else 'that would change'}: {changed}")
    print(f'lines skipped as attested data ("decision": "D-nnn"): {kept}')
    if not args.apply:
        print("\ndry run; pass --apply to write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
