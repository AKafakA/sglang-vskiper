"""Block 1B-2 (D-419) artifact contract for the binary-cohort count-GEMM configs.

CPU-only: reads the committed per-device artifacts beside ``sglang.srt.vpipe.kernel``.
Guards the two properties the cohort body now relies on:
  1. every device artifact carries the per-op projector keys (``projgd``, ``projup``) at
     every band its ``gateup``/``down`` keys cover — ``select_config`` is fail-closed, so a
     missing key would abort the first cohort pass at capture time;
  2. the projector keys keep the SAME BK as the RUN key they replaced (projgd <- gateup,
     projup <- down) per band, and re-tuned RUN keys keep their previous BK — the
     k-accumulation order is unchanged, which is what makes the re-tune bit-identical.
"""

import json
from pathlib import Path

import pytest

ARTIFACT_DIR = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "sglang"
    / "srt"
    / "vpipe"
    / "binary_cohort_configs"
)
ARTIFACTS = sorted(ARTIFACT_DIR.glob("*.json"))
REF_OP = {"projgd": "gateup", "projup": "down"}


def _load(path):
    return json.loads(path.read_text())


def _bands(artifact, op):
    return sorted(int(k.split("@")[1]) for k in artifact if k.split("@")[0] == op)


@pytest.mark.parametrize("path", ARTIFACTS, ids=[p.stem for p in ARTIFACTS])
def test_projector_keys_cover_every_band(path):
    art = _load(path)
    for proj_op, ref_op in REF_OP.items():
        assert _bands(art, proj_op) == _bands(art, ref_op), (
            f"{path.name}: {proj_op} bands must equal {ref_op} bands (fail-closed selection)"
        )


@pytest.mark.parametrize("path", ARTIFACTS, ids=[p.stem for p in ARTIFACTS])
def test_config_records_are_five_ints(path):
    art = _load(path)
    for key, rec in art.items():
        cfg = rec["config"]
        assert len(cfg) == 5 and all(isinstance(v, int) and v > 0 for v in cfg), key
        bm, bn, bk, warps, stages = cfg
        assert bm % 16 == 0 and bn % 16 == 0 and bk % 16 == 0, key


@pytest.mark.parametrize("path", ARTIFACTS, ids=[p.stem for p in ARTIFACTS])
def test_bk_unchanged_where_bit_identity_is_claimed(path):
    art = _load(path)
    for key, rec in art.items():
        op, _, band = key.partition("@")
        bk = rec["config"][2]
        if op in REF_OP:
            ref = art[f"{REF_OP[op]}@{band}"]
            # the projector key replaced the RUN key's config: same BK by construction
            assert bk == ref.get("prev_config", ref["config"])[2], key
        if rec.get("identical_to_prev"):
            assert bk == rec["prev_config"][2], f"{key}: BK changed but bit-identity claimed"
            assert rec.get("bk_fixed") == bk, key


def test_select_config_returns_projector_keys():
    from sglang.srt.vpipe.kernel import select_config

    art = _load(ARTIFACTS[0])
    for op in ("projgd", "projup", "gateup", "down"):
        cfg = select_config(art, op, 256)
        assert set(cfg) == {"block_m", "block_n", "block_k", "num_warps", "num_stages"}
    with pytest.raises(ValueError):
        select_config({"gateup@205": {"config": [64, 128, 64, 4, 3]}}, "projgd", 256)
