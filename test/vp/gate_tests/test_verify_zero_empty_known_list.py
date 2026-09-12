#!/usr/bin/env python3
"""[D-728] The zero-empty gate's known list: named and counted, never hidden; unknown still refuses.

Executed against real invocations of the gate on synthetic artifacts, not by reading its source."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GATE = ROOT / "test/vp/gates/verify_zero_empty.py"
KNOWN = json.loads((ROOT / "test/vp/gates/known_empties.json").read_text())
KNOWN_ID = KNOWN["entries"][0]["request_id"]


def _artifact(tmp: Path, texts: dict[str, str]) -> Path:
    path = tmp / "cell.jsonl"
    path.write_text(json.dumps({"request_ids": list(texts), "generated_texts": list(texts.values()),
                                "ignore_eos": False}) + "\n")
    return path


def _run(path: Path) -> tuple[int, str]:
    proc = subprocess.run([sys.executable, str(GATE), "--artifact", str(path)],
                          capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def test_every_entry_carries_a_witness_and_evidence():
    for entry in KNOWN["entries"]:
        assert entry["witness"] and entry["evidence"] and entry["request_id"], entry


def test_known_empty_passes_but_is_named_and_counted(tmp_path):
    rc, out = _run(_artifact(tmp_path, {"coqa:train:1:1": " Paris", KNOWN_ID: "", "coqa:train:2:2": " No"}))
    assert rc == 0, out
    assert f"KNOWN empty {KNOWN_ID}" in out and "inherited" in out, out
    assert "empty generations  : 1" in out, out          # still counted


def test_unknown_empty_still_refuses_even_beside_a_known_one(tmp_path):
    rc, out = _run(_artifact(tmp_path, {KNOWN_ID: "", "coqa:train:3249:10": "", "coqa:train:2:2": " No"}))
    assert rc == 1, out
    assert "coqa:train:3249:10" in out and "not on the known list" in out, out


def test_lmeval_lane_is_untouched(tmp_path):
    d = tmp_path / "lm"; d.mkdir()
    (d / "samples_coqa_x.jsonl").write_text(json.dumps({"doc_id": 80, "resps": [[""]]}) + "\n"
                                            + json.dumps({"doc_id": 81, "resps": [[" yes"]]}) + "\n")
    proc = subprocess.run([sys.executable, str(GATE), "--lmeval-dir", str(d)], capture_output=True, text=True)
    assert proc.returncode == 1 and "REFUSING" in proc.stdout, proc.stdout


def test_an_entry_without_a_witness_fails_closed(tmp_path):
    bad = tmp_path / "known_empties.json"
    bad.write_text(json.dumps({"entries": [{"request_id": "x:1:1"}]}))
    sys.path.insert(0, str(GATE.parent))
    import importlib; m = importlib.import_module("verify_zero_empty")
    try:
        m.load_known_empties(bad)
    except SystemExit as e:
        assert "no witness" in str(e)
    else:
        raise AssertionError("an entry without a witness was accepted")


def test_the_accounting_validator_consults_the_same_known_list():
    """[D-728] Found by running the harvest: validate_qps_artifact has its OWN empty-text
    check, which refused the cell the gate had just passed. Both must read one file."""
    src = (ROOT / "test/vp/validate_qps_artifact.py").read_text()
    assert 'gates" / "known_empties.json"' in src and "known_empty_request_ids" in src
    assert "empty_rows = [index for index in empty_rows if index not in known_empty_rows]" in src
