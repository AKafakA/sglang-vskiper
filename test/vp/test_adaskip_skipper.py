"""AdaSkip is a SECOND skipper family; this gates the pluggable contract.

The four canonical serving arms all use FlexiDepth or the deterministic mock, so
AdaSkip's 1,387 lines had no gate in this tree even though it is fully
implemented -- a dedicated sublayer executor path and device-tape sublayer
masks. Untested is how a live path rots into a dead one.

The fixture is REAL calibration output from a CSD3 run on 2026-08-19 against the
frozen Llama-3-8B revision 53346005, not a synthetic stand-in, so the schema and
layer geometry are the ones the implementation actually meets in practice.

These assertions are about the SKIPPER CONTRACT -- what the adapter declares and
what the runtime does with it. They need no GPU.
"""
import pathlib
import sys

FIXTURE = pathlib.Path(__file__).resolve().parent / "fixtures" / "adaskip_fixed_profile.json"


def _env(**over):
    env = {
        "SGLANG_VP_FULL_GRAPH_SKIPPER": "adaskip",
        "SGLANG_VP_ADASKIP_PROFILE": str(FIXTURE),
    }
    env.update(over)
    return env


def test_profile_fixture_present():
    assert FIXTURE.is_file() and FIXTURE.stat().st_size > 1000, (
        f"the AdaSkip calibration fixture is missing or truncated: {FIXTURE}"
    )


FROZEN_REVISION = "53346005fb0ef11d3b6a83b12c895cca40156b6c"


def test_profile_loads_and_matches_the_frozen_model():
    """Assert PROVENANCE, not just depth.

    The first version of this test checked only num_hidden_layers == 32 despite
    its name. Every Llama-3-8B derivative has 32 layers, so it stayed green for
    any checkpoint -- and that blind spot is exactly what let the adapter ship
    without binding its calibration to the served model.
    """
    from sglang.srt.vpipe.adaskip_profile import load_adaskip_profile

    profile = load_adaskip_profile(FIXTURE)
    assert profile.num_hidden_layers == 32, profile.num_hidden_layers
    assert profile.model_revision == FROZEN_REVISION, profile.model_revision
    assert "Meta-Llama-3-8B-Instruct" in profile.model_id, profile.model_id


def test_profile_is_bound_to_the_served_checkpoint():
    """A same-depth profile from ANOTHER checkpoint must be REFUSED.

    Skip masks and compensation scales are checkpoint-specific. Applying
    another model's profile changes what the server generates, with no error
    and no attestation difference -- a silent correctness failure.
    """
    from sglang.srt.vpipe.common import resolve_full_graph_skipper

    adapter = resolve_full_graph_skipper(_env())
    cases = [
        ({"revision": FROZEN_REVISION}, True, "matching revision"),
        ({"revision": "0" * 40}, False, "same depth, WRONG revision"),
        ({}, False, "served checkpoint identity unknown"),
        # A DIRECTORY NAME IS NOT IDENTITY. The previous version of this test
        # accepted a matching basename, which BLESSED a bypass in the adapter:
        # renaming any same-depth checkpoint to the calibrated snapshot's
        # directory name restored the original silent miscalibration. Both of
        # these must now be refused -- the path carries no weight identity at
        # all, whether or not it happens to match.
        ({"model_id": "/any/Meta-Llama-3-8B-Instruct-53346005"}, False,
         "matching dir name is NOT identity"),
        ({"model_id": "/any/Some-Other-Checkpoint"}, False, "differing dir name"),
        ({"revision": FROZEN_REVISION, "model_id": "/any/Wrong-Looking-Name"}, True,
         "revision decides; a mismatched path does not veto it"),
    ]
    for identity, want_ok, label in cases:
        try:
            adapter.routed_layer_ids(
                num_hidden_layers=32, flexidepth_layer_ids=(),
                model_identity=identity,
            )
            got_ok = True
        except ValueError:
            got_ok = False
        assert got_ok == want_ok, f"{label}: accepted={got_ok}, expected {want_ok}"


def test_adapter_declares_sublayer_actions():
    """AdaSkip's independent attention/MLP actions must NOT be lowered to binary.

    The production executor supports exactly RUN/PROJECT_ONLY; silently
    lowering SKIP_ATTN/SKIP_MLP into that would change the algorithm while
    still reporting AdaSkip. The adapter must keep declaring the sublayer set.
    """
    from sglang.srt.vpipe.common import resolve_full_graph_skipper
    from sglang.srt.vpipe.env import SUBLAYER_EXECUTION
    from sglang.srt.vpipe.types import LogicalAction

    adapter = resolve_full_graph_skipper(_env())
    assert adapter.name == "adaskip"
    assert adapter.execution_kind == SUBLAYER_EXECUTION
    assert adapter.supported_actions == frozenset(
        (LogicalAction.RUN, LogicalAction.SKIP_ATTN, LogicalAction.SKIP_MLP)
    )
    assert adapter.requires_flexidepth_weights is False


def test_adaskip_requires_its_profile():
    """A missing profile must FAIL CLOSED, not fall back to some default."""
    from sglang.srt.vpipe.common import resolve_full_graph_skipper

    try:
        resolve_full_graph_skipper({"SGLANG_VP_FULL_GRAPH_SKIPPER": "adaskip"})
    except ValueError as exc:
        assert "ADASKIP_PROFILE" in str(exc) or "profile" in str(exc).lower()
    else:
        raise AssertionError("AdaSkip resolved with no profile configured")


def test_adaskip_rejects_deterministic_mock_settings():
    """Two skipper families must not be configured at once."""
    from sglang.srt.vpipe.common import resolve_full_graph_skipper

    try:
        # the real mock knob, from env.py:_MOCK_CONFIG_ENVS -- my first version
        # guessed a name that does not exist, so the check appeared broken when
        # it was the test that was wrong
        resolve_full_graph_skipper(
            _env(SGLANG_VP_FULL_GRAPH_MOCK_TOKEN_SKIP_RATE="0.5")
        )
    except ValueError:
        pass
    else:
        raise AssertionError("AdaSkip accepted deterministic-mock settings")


def main():
    checks = [
        test_profile_fixture_present,
        test_profile_loads_and_matches_the_frozen_model,
        test_profile_is_bound_to_the_served_checkpoint,
        test_adapter_declares_sublayer_actions,
        test_adaskip_requires_its_profile,
        test_adaskip_rejects_deterministic_mock_settings,
    ]
    bad = 0
    for fn in checks:
        try:
            fn()
            print(f"  ok   {fn.__name__}")
        except Exception as exc:
            print(f"  FAIL {fn.__name__}: {str(exc)[:110]}")
            bad += 1
    print(f"ADASKIP SKIPPER CONTRACT: {'FAIL' if bad else 'PASS'} "
          f"({len(checks) - bad}/{len(checks)})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
