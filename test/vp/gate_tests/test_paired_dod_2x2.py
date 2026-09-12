"""paired_dod_2x2: the d-o-d is paired per document; arms scoring different documents refuse."""
import json, subprocess, sys
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "paired_dod_2x2.py"


def _write(root: Path, arm: str, scores: dict[int, float], flt: str = "flexible-extract") -> Path:
    d = root / arm / "served"; d.mkdir(parents=True)
    with open(d / "samples_gsm8k_2026-01-01T00-00-00.000000.jsonl", "w") as fh:
        for doc_id, v in scores.items():
            fh.write(json.dumps({"doc_id": doc_id, "filter": flt, "exact_match": v}) + "\n")
            fh.write(json.dumps({"doc_id": doc_id, "filter": "strict-match", "exact_match": 0.0}) + "\n")
    return root / arm


def test_paired_dod_is_the_mean_of_per_document_differences(tmp_path):
    # docs: A,B,C,D per doc -> x = (D-C)-(B-A) = [0, -1, +1, 0] -> mean 0, sd 0.8165
    a = _write(tmp_path, "A", {0: 1, 1: 1, 2: 0, 3: 1})
    b = _write(tmp_path, "B", {0: 1, 1: 0, 2: 0, 3: 0})
    c = _write(tmp_path, "C", {0: 1, 1: 1, 2: 0, 3: 1})
    d = _write(tmp_path, "D", {0: 1, 1: 1, 2: 1, 3: 0})
    out = tmp_path / "r.json"
    subprocess.run([sys.executable, str(TOOL), "--dataset", "gsm8k", "--arm", f"A={a}", "--arm", f"B={b}",
                    "--arm", f"C={c}", "--arm", f"D={d}", "--json", str(out)], check=True, capture_output=True)
    r = json.loads(out.read_text())
    assert r["n_docs"] == 4 and abs(r["dod"]["mean_pp"]) < 1e-9
    assert abs(r["B_minus_A"]["mean_pp"] - (-50.0)) < 1e-9 and abs(r["D_minus_C"]["mean_pp"] - (-50.0 + 50.0)) < 1e-9
    assert r["verdict"].startswith("parity")


def test_arms_scoring_different_documents_refuse(tmp_path):
    a = _write(tmp_path, "A", {0: 1, 1: 1}); b = _write(tmp_path, "B", {0: 1, 1: 0})
    c = _write(tmp_path, "C", {0: 1, 1: 1}); d = _write(tmp_path, "D", {0: 1})  # one document short
    p = subprocess.run([sys.executable, str(TOOL), "--dataset", "gsm8k", "--arm", f"A={a}", "--arm", f"B={b}",
                        "--arm", f"C={c}", "--arm", f"D={d}"], capture_output=True, text=True)
    assert p.returncode != 0 and "do not score the same documents" in (p.stderr + p.stdout)
