"""paired_dod_2x2: the d-o-d is paired per document; arms scoring different documents refuse."""
import json, subprocess, sys
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "paired_dod_2x2.py"


def _write(root: Path, arm: str, scores: dict[int, float], flt: str = "flexible-extract") -> Path:
    d = root / arm / "served"; d.mkdir(parents=True)
    with open(d / "samples_gsm8k_2026-01-01T00-00-00.000000.jsonl", "w") as fh:
        for doc_id, v in scores.items():
            # the composite rule reads each filter's extraction: strict parses nothing here, so the
            # flexible verdict is the document's verdict (its extraction is the answer it scored)
            fh.write(json.dumps({"doc_id": doc_id, "filter": flt, "exact_match": v,
                                 "filtered_resps": ["1"]}) + "\n")
            fh.write(json.dumps({"doc_id": doc_id, "filter": "strict-match", "exact_match": 0.0,
                                 "filtered_resps": ["[invalid]"]}) + "\n")
    return root / arm


def test_paired_dod_is_the_mean_of_per_document_differences(tmp_path):
    # Per document: (B-A) = [0, -1, 0, -1]; the served pair reproduces it exactly (C = A, D = B),
    # so x_i = (D_i-C_i) - (B_i-A_i) = 0 for every document: d-o-d mean 0, interval 0 -> no resolved difference.
    a = _write(tmp_path, "A", {0: 1, 1: 1, 2: 0, 3: 1})
    b = _write(tmp_path, "B", {0: 1, 1: 0, 2: 0, 3: 0})
    c = _write(tmp_path, "C", {0: 1, 1: 1, 2: 0, 3: 1})
    d = _write(tmp_path, "D", {0: 1, 1: 0, 2: 0, 3: 0})
    out = tmp_path / "r.json"
    subprocess.run([sys.executable, str(TOOL), "--dataset", "gsm8k", "--arm", f"A={a}", "--arm", f"B={b}",
                    "--arm", f"C={c}", "--arm", f"D={d}", "--json", str(out)], check=True, capture_output=True)
    r = json.loads(out.read_text())
    assert r["n_docs"] == 4
    assert abs(r["B_minus_A"]["mean_pp"] - (-50.0)) < 1e-9 and abs(r["D_minus_C"]["mean_pp"] - (-50.0)) < 1e-9
    assert abs(r["dod"]["mean_pp"]) < 1e-9 and r["dod"]["ci95_half_pp"] < 1e-9
    assert r["verdict"].startswith("no resolved difference")
    # and a case where the served pair differs on one document: x = [0, +1, 0, 0] -> mean +25 pp
    d2 = _write(tmp_path / "v2", "D", {0: 1, 1: 1, 2: 0, 3: 0})
    subprocess.run([sys.executable, str(TOOL), "--dataset", "gsm8k", "--arm", f"A={a}", "--arm", f"B={b}",
                    "--arm", f"C={c}", "--arm", f"D={d2}", "--json", str(out)], check=True, capture_output=True)
    r = json.loads(out.read_text())
    assert abs(r["dod"]["mean_pp"] - 25.0) < 1e-9


def test_arms_scoring_different_documents_refuse(tmp_path):
    a = _write(tmp_path, "A", {0: 1, 1: 1}); b = _write(tmp_path, "B", {0: 1, 1: 0})
    c = _write(tmp_path, "C", {0: 1, 1: 1}); d = _write(tmp_path, "D", {0: 1})  # one document short
    p = subprocess.run([sys.executable, str(TOOL), "--dataset", "gsm8k", "--arm", f"A={a}", "--arm", f"B={b}",
                        "--arm", f"C={c}", "--arm", f"D={d}"], capture_output=True, text=True)
    assert p.returncode != 0 and "do not score the same documents" in (p.stderr + p.stdout)
