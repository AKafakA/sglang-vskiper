#!/usr/bin/env python3
"""The quality lane must feed every arm the SAME token ids -- pinned at the source.

Found by RUNNING the arms on the two requests that emptied (D-724, 2026-09-12): the driver
passed `tokenized_requests=False`, so lm-eval sent TEXT to the served arms and SGLang's
tokenizer prepended <|begin_of_text|> on raw prompts, while lm-eval's HF backend (arms A/B)
encodes with add_special_tokens=False. On coqa doc 80 that one token flipped the checkpoint's
first token from ' Island' to EOS, and the 2x2 compared (D-C) against (B-A) on different inputs.
lm-eval's API backend with tokenized_requests=True encodes exactly as its HF backend does.

Source-level on purpose: the flag is a literal inside the model_args string, and the shape of
that string is what needs protecting."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DRIVER = (ROOT / "vskipper/src/vskipper/experiments/run_lmeval_quality.py").read_text()


def _code(text: str) -> str:
    return "\n".join(l for l in text.splitlines() if not l.strip().startswith("#"))


def test_served_arms_get_token_ids_not_text():
    code = _code(DRIVER)
    assert "tokenized_requests=True" in code, "lm-eval must tokenize client-side and send ids"
    assert "tokenized_requests=False" not in code, "the server-side text path re-adds BOS (D-724)"


def test_hf_arms_do_not_add_bos_either():
    """Arms A/B keep lm-eval's HF default (add_special_tokens=False). If someone adds
    add_bos_token=True to make A/B 'match' a BOS-adding server, both lanes drift together away
    from the perf lane's suites, which are encoded without BOS."""
    assert "add_bos_token" not in _code(DRIVER)


def test_prefill_runner_gate_is_the_design_not_an_env_var():
    """[D-734] The runner's prefill-regime machinery was gated on SGLANG_FD_WEIGHTS, which
    D-609 stopped exporting; every headline cell then served with the escape and the
    counters dead. The serving path must never read that variable again."""
    runner = _code((ROOT / "python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py").read_text())
    assert 'os.environ.get("SGLANG_FD_WEIGHTS"' not in runner
    # no serving-path module may key behaviour on the deleted variable (D-609/D-734)
    import subprocess
    hits = subprocess.run(["grep", "-rln", 'environ.get("SGLANG_FD_WEIGHTS"', str(ROOT / "python/sglang/srt"),
                           str(ROOT / "vskipper/src/vskipper/runtime"),
                           str(ROOT / "vskipper/src/vskipper/integration"),
                           str(ROOT / "vskipper/src/vskipper/kernels")],
                          capture_output=True, text=True).stdout.split()
    assert hits == [], hits
    assert "skipper_deployed()" in runner and "flexidepth_weights_path()" in runner
    src = (ROOT / "vskipper/src/vskipper/runtime/attestation.py").read_text()
    assert 'regime_counters.pop("prefill", None)' not in src, "the prefill counters are live again"
