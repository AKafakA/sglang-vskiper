#!/usr/bin/env python3
"""Package frozen reviewer evidence with this exact committed analysis source.

Artifact handling only: copies an allowlisted dependency subset, records every
source/export hash and refuses to overwrite a release. Reproduction is validated
separately on the assigned compute host before publication.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOTS = {"analysis", "evidence", "expected", "identity-preserved", "qwen8b-selection"}
DOCUMENTS = (
    "README.md", "REPRODUCE.md", "INDEX.md", "DATA_FORMAT.md", "code/README.md",
    "expected/generated/README.md", "identity-preserved/paper-carried/README.md",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_relative(name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"invalid relative artifact path: {name!r}")
    return path


def source_path(old: str) -> str:
    if old.startswith("test/vp/"):
        return old.replace("test/vp/", "vskipper/src/vskipper/analysis/", 1)
    if old.startswith("python/sglang/srt/vpipe/binary_cohort_configs/"):
        return old.replace("python/sglang/srt/vpipe/", "vskipper/src/vskipper/kernels/", 1)
    if old.startswith("python/sglang/srt/vpipe/"):
        return old.replace("python/sglang/srt/vpipe/", "vskipper/src/vskipper/runtime/", 1)
    if old == "scripts/reproduce_analysis.sh":
        return "vskipper/scripts/reproduce_analysis.sh"
    if old == "LICENSE":
        return old
    raise ValueError(f"unexpected source in control-pack allowlist: {old}")


def band_table(text: str):
    match = re.search(r"SERVED_DECODE_KV_BAND_BY_DEVICE[^=]*=\s*(\{.*?\n\})", text, re.S)
    if match is None:
        raise ValueError("missing static band-table declaration")
    return ast.literal_eval(match.group(1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    args = parser.parse_args()
    control, output, archive = args.control_pack.resolve(), args.output, args.zip
    for target in (output, archive):
        if target.exists() or target.is_symlink():
            raise SystemExit(f"refusing to overwrite {target}")
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if status:
        raise SystemExit("package only a clean, committed reference checkout")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest_bytes = (control / "MANIFEST.sha256").read_bytes()
    manifest = {}
    for line in manifest_bytes.decode().splitlines():
        expected, name = line.split("  ", 1)
        rel = safe_relative(name)
        path = control / rel
        if path.is_symlink() or not path.is_file() or digest(path.read_bytes()) != expected:
            raise SystemExit(f"control manifest mismatch: {name}")
        manifest[name] = expected
    output.mkdir(parents=True, exist_ok=False)

    def put(name: str, data: bytes):
        target = output / safe_relative(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ValueError(f"duplicate export path: {name}")
        target.write_bytes(data)

    preserved = []
    for name in sorted(manifest):
        rel = Path(name)
        if rel.parts[0] not in DATA_ROOTS or name in DOCUMENTS:
            continue
        if rel.suffix.lower() in {".pdf", ".zip", ".bib", ".sty", ".cls"} or rel.name in {"main.tex", "HANDOFF.md", "AGENTS.md"}:
            raise ValueError(f"non-reproduction material in control allowlist: {name}")
        put(name, (control / rel).read_bytes())
        preserved.append({"path": name, "sha256": manifest[name]})
    put("verify_results.py", (control / "verify_results.py").read_bytes())

    sources = []
    for line in (control / "CODE_INPUTS.sha256").read_text().splitlines():
        _, old = line.split("  ", 1)
        current = source_path(old)
        original = (ROOT / current).read_bytes()
        exported = original
        adjustment = None
        if old.endswith("/design.py"):
            exported = (control / "code" / old).read_bytes()
            if band_table(original.decode()) != band_table(exported.decode()):
                raise ValueError("static band excerpt disagrees with committed runtime")
            adjustment = "Static band-table excerpt; values and ordering preserved."
        if old.endswith("/device_roofline.json"):
            exported = original.replace(b"2026-09-16, CloudLab A100-40 row", b"40 GB device")
            if exported != original:
                adjustment = "Machine-independent device provenance label; numerical fields unchanged."
        put("code/" + current, exported)
        sources.append({"repository_path": current, "export_path": "code/" + current,
                        "source_sha256": digest(original), "export_sha256": digest(exported),
                        "adjustment": adjustment})

    dependency_path = "vskipper/requirements-analysis.txt"
    dependency_bytes = (ROOT / dependency_path).read_bytes()
    put("code/" + dependency_path, dependency_bytes)
    sources.append({"repository_path": dependency_path,
                    "export_path": "code/" + dependency_path,
                    "source_sha256": digest(dependency_bytes),
                    "export_sha256": digest(dependency_bytes), "adjustment": None})

    provenance = {"measurement_version": "v1.6", "reviewer_package": "r2",
                  "reference_commit": commit, "control_manifest_sha256": digest(manifest_bytes),
                  "preserved_data": preserved, "source_files": sources,
                  "verification": "Run reproduce_results.sh to check all 114 reference outputs; release validation is recorded separately."}
    put("SOURCE_PROVENANCE.json", (json.dumps(provenance, indent=2) + "\n").encode())
    adapter = '''#!/usr/bin/env bash
set -euo pipefail
PACK=$(cd -- "$(dirname -- "$0")" && pwd -P)
[ "$#" -eq 1 ] || { echo 'Usage: reproduce_results.sh NEW_OUTPUT_DIRECTORY' >&2; exit 2; }
if command -v sha256sum >/dev/null 2>&1; then HASH=(sha256sum); else HASH=(shasum -a 256); fi
(cd "$PACK/code" && "${HASH[@]}" -c "$PACK/CODE_INPUTS.sha256")
export PACK
exec bash "$PACK/code/vskipper/scripts/reproduce_analysis.sh" "$1"
'''
    put("reproduce_results.sh", adapter.encode())

    for name in DOCUMENTS:
        text = (control / name).read_text()
        text = text.replace("Python 3.10 or newer", "CPython 3.12.11")
        text = text.replace("python3 -m venv analysis-env", "python3.12 -m venv analysis-env")
        text = text.replace("python3 -m pip install numpy matplotlib",
                            "python3 -m pip install -r code/vskipper/requirements-analysis.txt")
        if name == "REPRODUCE.md":
            text = text.replace("## 2. Verify and regenerate",
                                "The exact-output check is validated with CPython 3.12.11 and the pinned\n"
                                "dependencies above. Python 3.11 introduces roundoff differences in five\n"
                                "JSON files; use the validated environment for exact comparison.\n\n"
                                "## 2. Verify and regenerate")
        text = text.replace("reviewer package r1", "reviewer package r2").replace("Reviewer-package r1", "Reviewer-package r2")
        text = text.replace("code/test/vp/", "code/vskipper/src/vskipper/analysis/")
        text = text.replace("python/sglang/srt/vpipe/design.py", "vskipper/src/vskipper/runtime/design.py")
        text = text.replace("selects the recipe's results-only stages", "runs the results-only recipe")
        text = text.replace("three\npackaging adjustments", "two\npackaging adjustments")
        text = text.replace("- One analysis progress message uses a generic native-evaluation label.\n", "")
        text = text.replace("The manuscript-build and manuscript-verification stages are omitted.",
                            "The recipe contains only analysis and result verification stages.")
        heading, rest = text.split("\n", 1)
        note = "\n\nThis r2 pack follows the reorganized reference repository. "
        note += "`SOURCE_PROVENANCE.json` records the exact source commit and per-file hashes; "
        note += "the numerical evidence and reference outputs are unchanged.\n"
        text = heading + note + rest
        put(name, text.encode())

    source_manifest = ''.join(f"{entry['export_sha256']}  {entry['export_path'].removeprefix('code/')}\n"
                              for entry in sorted(sources, key=lambda entry: entry['export_path']))
    put("CODE_INPUTS.sha256", source_manifest.encode())
    contents = ['path\tpurpose']
    for path in sorted(p for p in output.rglob('*') if p.is_file()):
        rel = path.relative_to(output).as_posix()
        purpose = ("frozen reproduction data/reference" if rel in {p['path'] for p in preserved}
                   else "bundled analysis source" if rel.startswith('code/') and rel != 'code/README.md'
                   else "reproduction instructions, provenance or entrypoint")
        contents.append(rel + '\t' + purpose)
    put("CONTENTS.tsv", ('\n'.join(contents) + '\n').encode())
    all_files = sorted(p for p in output.rglob('*') if p.is_file())
    put("MANIFEST.sha256", ''.join(f"{digest(p.read_bytes())}  {p.relative_to(output).as_posix()}\n"
                                   for p in all_files).encode())
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(p for p in output.rglob('*') if p.is_file()):
            bundle.write(path, 'v1.6/' + path.relative_to(output).as_posix())
    print(json.dumps({"reference_commit": commit, "preserved_files": len(preserved),
                      "source_files": len(sources), "zip_sha256": digest(archive.read_bytes()),
                      "status": "candidate; remote validation required"}, indent=2))


if __name__ == '__main__':
    main()
