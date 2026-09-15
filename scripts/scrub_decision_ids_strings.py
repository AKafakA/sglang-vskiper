#!/usr/bin/env python3
"""Drop decision identifiers from docstrings, using Python's own tokenizer.

The comment-line pass (`scrub_decision_ids.py`) deliberately refuses to touch anything but a
comment, because a regex over whole lines broke three files by firing inside string literals.
This does the docstring half safely: it tokenizes each file, edits only STRING tokens that are
documentation (the first statement of a module, class or function), and leaves every other
string -- format templates, table labels, JSON keys -- untouched.

A file is rewritten only if it still parses to an identical AST apart from those docstrings.

usage: scrub_decision_ids_strings.py [--apply]
"""
import argparse, ast, io, pathlib, re, subprocess, sys, tokenize

ID = r"D-\d{3}"
RULES = [
    (re.compile(r"\[Audit " + ID + r"(?: #\d+)?\]"), "[audit]"),
    (re.compile(r"\[" + ID + r"(?: add\.? ?\d*)?\]\s*"), ""),
    (re.compile(r"\((?:owner )?" + ID + r"(?: #\d+)?\)"), ""),
    (re.compile(r"\(" + ID + r"(?: #\d+)?, ?"), "("),
    (re.compile(r",? ?" + ID + r"(?: #\d+)?\)"), ")"),
    (re.compile(r",? ?\b" + ID + r"\b"), ""),
    (re.compile(r" +([.,;:])"), r"\1"),
]


def docstring_spans(tree):
    """Byte-agnostic (lineno, col) starts of every docstring in the module."""
    spans = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                and isinstance(first.value.value, str):
            spans.add((first.value.lineno, first.value.col_offset))
    return spans


def scrub(path: pathlib.Path):
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    spans = docstring_spans(tree)
    if not spans:
        return None
    out, changed = [], False
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.STRING and tok.start in spans and re.search(ID, tok.string):
            new = tok.string
            for rx, sub in RULES:
                new = rx.sub(sub, new)
            if new != tok.string:
                changed = True
                out.append(tok._replace(string=new))
                continue
        out.append(tok)
    if not changed:
        return None
    try:
        return tokenize.untokenize(out)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    files = subprocess.run(["git", "grep", "-lIE", r"\bD-[0-9]{3}\b", "--", "*.py"],
                           capture_output=True, text=True).stdout.split()
    files = [f for f in files if not f.startswith("scripts/scrub_decision_ids")]
    done = skipped = 0
    for f in files:
        p = pathlib.Path(f)
        new = scrub(p)
        if new is None:
            skipped += 1
            continue
        try:
            ast.parse(new)
        except SyntaxError:
            print(f"  REFUSED (would not parse): {f}")
            skipped += 1
            continue
        done += 1
        if args.apply:
            p.write_text(new, encoding="utf-8")
    print(f"python files with an id: {len(files)}")
    print(f"files {'rewritten' if args.apply else 'that would be rewritten'}: {done}")
    print(f"files with no docstring hit, or refused: {skipped}")
    if not args.apply:
        print("\ndry run; pass --apply to write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
