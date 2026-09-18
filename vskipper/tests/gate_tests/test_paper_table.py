#!/usr/bin/env python3
"""The paper-table emitter/verifier, pinned by CORRUPTION rather than by a pass.

main.tex carries 996 numeric literals and every headline one is typed by hand. The checker
that used to guard them, vPipe-doc/codex/tools/verify_paper_numbers.py, is DEAD for this
campaign -- it globs the v1.3 directory layout, looks for workgate_*.log where the driver
writes .json, and hardcodes two VOID knees. A gate that cannot fire is worse than absent.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / "vskipper/src/vskipper/analysis/paper_table.py"
COLUMNS = ["TTFT p50", "TTFT mean", "TTFT p95", "TTFT p99",
           "TPOT p50", "TPOT mean", "TPOT p95", "TPOT p99",
           "E2E p50", "E2E mean", "E2E p95", "E2E p99", "makespan", "output TPS"]


def _report(path: Path, datasets=(("gsm8k", 11.0), ("bbh_cot", 25.0)), drop=None) -> Path:
    rows = []
    for dataset, knee in datasets:
        for mult in (0.75, 0.95, 1.25):
            rate = knee * mult
            suite = f"{dataset}_eqw_r{str(rate).replace('.', 'p')}"
            for index, column in enumerate(COLUMNS):
                if drop and column == drop:
                    continue
                rows.append({"dataset": dataset, "suite": suite, "metric": column,
                             "field": column, "n": 6, "mean_pct": -1.0 * index - 0.5,
                             "ci95_half": 0.3 + index * 0.1, "verdict": "x"})
    path.write_text(json.dumps({"baseline": "upstream", "treatment": "t", "rows": rows}))
    return path


def _run(*args) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(TOOL), *map(str, args)],
                          capture_output=True, text=True)


def test_emit_then_verify_is_clean(tmp_path: Path):
    report = _report(tmp_path / "r.json")
    out = _run(report, "--knee", "gsm8k=11", "--knee", "bbh_cot=25", "--emit")
    assert out.returncode == 0, out.stderr
    (tmp_path / "main.tex").write_text(out.stdout)
    check = _run(report, "--knee", "gsm8k=11", "--knee", "bbh_cot=25",
                 "--verify", tmp_path / "main.tex")
    assert check.returncode == 0, check.stderr


def test_a_corrupted_cell_is_caught_and_NAMED(tmp_path: Path):
    """The point of the tool: it must say which row and which column."""
    report = _report(tmp_path / "r.json")
    tex = _run(report, "--knee", "gsm8k=11", "--knee", "bbh_cot=25", "--emit").stdout
    (tmp_path / "bad.tex").write_text(tex.replace("-0.5", "-9.9", 1))
    check = _run(report, "--knee", "gsm8k=11", "--knee", "bbh_cot=25",
                 "--verify", tmp_path / "bad.tex")
    assert check.returncode == 1
    assert "TTFT p50" in check.stderr and "-9.9" in check.stderr


def test_a_changed_n_is_caught(tmp_path: Path):
    """A row that quietly drops to n=5 while its neighbours are 6 is invisible rendered."""
    report = _report(tmp_path / "r.json")
    tex = _run(report, "--knee", "gsm8k=11", "--knee", "bbh_cot=25", "--emit").stdout
    (tmp_path / "bad.tex").write_text(tex.replace("& 6 &", "& 5 &", 1))
    check = _run(report, "--knee", "gsm8k=11", "--knee", "bbh_cot=25",
                 "--verify", tmp_path / "bad.tex")
    assert check.returncode == 1 and "n=5" in check.stderr


def test_a_STALE_row_left_in_main_tex_is_caught(tmp_path: Path):
    """Both directions: main.tex must not keep a row the artifacts no longer have."""
    report = _report(tmp_path / "r.json", datasets=(("gsm8k", 11.0),))
    tex = _run(report, "--knee", "gsm8k=11", "--emit").stdout
    tex += "\nCoQA & $0.95\\times Q^*$ & 6 & " + " & ".join(["+1.0 $\\pm$ 0.1"] * 10) + " \\\\\n"
    (tmp_path / "bad.tex").write_text(tex)
    check = _run(report, "--knee", "gsm8k=11", "--verify", tmp_path / "bad.tex")
    assert check.returncode == 1 and "that the artifacts do not" in check.stderr


def test_a_MISSING_row_is_caught(tmp_path: Path):
    report = _report(tmp_path / "r.json")
    tex = _run(report, "--knee", "gsm8k=11", "--knee", "bbh_cot=25", "--emit").stdout
    (tmp_path / "bad.tex").write_text("\n".join(tex.splitlines()[1:]))
    check = _run(report, "--knee", "gsm8k=11", "--knee", "bbh_cot=25",
                 "--verify", tmp_path / "bad.tex")
    assert check.returncode == 1 and "that main.tex does not" in check.stderr


def test_a_rate_off_the_declared_rungs_is_REFUSED_not_labelled(tmp_path: Path):
    """A campaign rate with no table column must stop the emit, not appear unlabelled."""
    report = _report(tmp_path / "r.json")
    check = _run(report, "--knee", "gsm8k=7", "--knee", "bbh_cot=25", "--emit")
    assert check.returncode == 2 and "not 0.75/0.95/1.25" in check.stderr


def test_a_missing_COLUMN_is_refused(tmp_path: Path):
    report = _report(tmp_path / "r.json", drop="TPOT p99")
    check = _run(report, "--knee", "gsm8k=11", "--knee", "bbh_cot=25", "--emit")
    assert check.returncode == 2 and "missing columns" in check.stderr


def test_macros_are_latex_legal_names(tmp_path: Path):
    report = _report(tmp_path / "r.json")
    _run(report, "--knee", "gsm8k=11", "--knee", "bbh_cot=25", "--emit",
         "--macros", tmp_path / "m.tex")
    text = (tmp_path / "m.tex").read_text()
    assert text.startswith("% GENERATED")
    import re
    names = re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", text)
    expected = 2 * 3 * len(COLUMNS) * 2  # datasets x rungs x fields x (estimate, CI)
    assert len(names) == expected, len(names)
    assert len(set(names)) == expected, "macro names must be unique"
    assert all(n.isalpha() for n in names)


def test_the_rows_file_swallows_its_own_final_newline(tmp_path: Path):
    r"""`\input` inside a tabular leaves a space after the last `\\`, opening a new row, and the
    `\bottomrule` that follows is a `\noalign` -> "Misplaced \noalign". Found by building."""
    report = _report(tmp_path / "r.json")
    out = _run(report, "--knee", "gsm8k=11", "--knee", "bbh_cot=25", "--emit")
    assert out.stdout.rstrip("\n").endswith("%"), repr(out.stdout[-40:])


def test_verify_sees_rows_behind_an_input(tmp_path: Path):
    """Rows live in their own generated file; a verifier reading only main.tex would report
    every row missing -- a false alarm that trains the reader to ignore it."""
    report = _report(tmp_path / "r.json")
    rows = _run(report, "--knee", "gsm8k=11", "--knee", "bbh_cot=25", "--emit").stdout
    (tmp_path / "generated").mkdir()
    (tmp_path / "generated/headline_rows.tex").write_text(rows)
    (tmp_path / "main.tex").write_text(
        "\\begin{tabular}{l}\n\\input{generated/headline_rows}\n\\end{tabular}\n")
    check = _run(report, "--knee", "gsm8k=11", "--knee", "bbh_cot=25",
                 "--verify", tmp_path / "main.tex")
    assert check.returncode == 0, check.stderr


def test_verify_follows_the_TeX_PRIMITIVE_wrapper_too(tmp_path: Path):
    r"""The paper uses \inputrows (a \@@input wrapper) because \input inside a tabular breaks
    \bottomrule. A verifier that tracks only \input goes blind the moment the paper changes
    mechanism -- and reports every row missing, which reads like a real failure."""
    report = _report(tmp_path / "r.json")
    rows = _run(report, "--knee", "gsm8k=11", "--knee", "bbh_cot=25", "--emit").stdout
    (tmp_path / "generated").mkdir()
    (tmp_path / "generated/headline_rows.tex").write_text(rows)
    (tmp_path / "main.tex").write_text(
        "\\begin{tabular}{l}\n\\inputrows{generated/headline_rows.tex}\n\\end{tabular}\n")
    check = _run(report, "--knee", "gsm8k=11", "--knee", "bbh_cot=25",
                 "--verify", tmp_path / "main.tex")
    assert check.returncode == 0, check.stderr
