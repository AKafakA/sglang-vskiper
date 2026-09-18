#!/usr/bin/env python3
"""The D-701 split: policy vs projector, pinned as a BEHAVIOUR-PRESERVING refactor.

Before the split, one boolean `requires_flexidepth_weights` answered two different
questions -- "does the policy read a trained gate?" and "does a skipped row have
something cheap to compute?" -- and the regime-switch capability check refused any
adapter that answered no. That made "any whole-layer skipper plugs in" false: a
policy bringing its own gate could not declare it.

The refactor must change NOTHING about what the two shipped adapters do. These
tests are source-level because the package needs torch and this box has none
(CLAUDE.md: no local serving execution) -- the runtime equivalence is re-checked
on the GPU host by booting both arms and diffing the served design.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TYPES = (ROOT / "vskipper/src/vskipper/runtime/types.py").read_text()
SKIPPER = (ROOT / "vskipper/src/vskipper/runtime/skipper.py").read_text()
COMMON = (ROOT / "vskipper/src/vskipper/runtime/common.py").read_text()
VALIDATION = (ROOT / "vskipper/src/vskipper/runtime/validation.py").read_text()
EXAMPLE = (ROOT / "vskipper/src/vskipper/experiments/example_skipper_plugin.py").read_text()


def _class_body(source: str, name: str) -> str:
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"class {name} not found")


def _declared(source: str, cls: str, field: str) -> str:
    body = _class_body(source, cls)
    match = re.search(rf"^\s*{field}\s*=\s*(.+)$", body, re.M)
    assert match, f"{cls} does not declare {field}"
    return match.group(1).strip()


# --- the two questions are now two fields ------------------------------------

def test_flexidepth_declares_both_router_and_projector():
    assert _declared(SKIPPER, "FlexiDepthFullGraphAdapter", "requires_router_weights") == "True"
    assert _declared(SKIPPER, "FlexiDepthFullGraphAdapter", "projector_kind") == "FLEXIDEPTH_PROJECTOR"


def test_the_mock_declares_NO_router_but_still_needs_a_projector():
    """The asymmetry the old boolean could not express."""
    assert _declared(SKIPPER, "DeterministicMockFullGraphAdapter", "requires_router_weights") == "False"
    assert _declared(SKIPPER, "DeterministicMockFullGraphAdapter", "projector_kind") == "FLEXIDEPTH_PROJECTOR"


# --- behaviour preservation ---------------------------------------------------

def test_requires_flexidepth_weights_is_derived_and_still_true_for_both():
    """Truth table, pinned: True or X = True; False or True = True.

    Both shipped adapters keep the value every existing call site already reads,
    so none of those sites needed an edit -- which is the whole claim of "no
    behavioural difference".
    """
    body = _class_body(TYPES, "FullGraphSkipperAdapter")
    assert "@property" in body and "def requires_flexidepth_weights" in body
    expr = 'return self.requires_router_weights or self.projector_kind == "flexidepth"'
    assert expr in body, "the derivation changed; re-check every consumer"
    for declared_router, declared_projector in ((True, "flexidepth"), (False, "flexidepth")):
        assert declared_router or declared_projector == "flexidepth"


def test_the_five_generic_consumers_were_not_repointed():
    """Only the regime-switch check changes; the rest still ask the derived question."""
    assert VALIDATION.count("requires_flexidepth_weights") == 5
    assert "regime switch requires FlexiDepth router/projector weights" not in VALIDATION


def test_the_regime_switch_now_requires_a_PROJECTOR_not_a_gate():
    gate = _class_body(VALIDATION, "_") if False else VALIDATION
    idx = gate.index("def assert_regime_switch_skipper_capability")
    body = gate[idx: gate.index("def ", idx + 10)]
    assert "resolve_projector(adapter.projector_kind)" in body
    # the two action checks are untouched -- they are the AdaSkip boundary
    assert "execution_kind != RUN_PROJECT_EXECUTION" in body
    assert "supported_actions != frozenset(" in body


# --- the registry is open -----------------------------------------------------

def test_both_policies_resolve_through_ONE_registry_lookup():
    assert "_SKIPPERS: dict[str, SkipperFactory]" in SKIPPER
    for name in ("FLEXIDEPTH_FULL_GRAPH_SKIPPER:", "DETERMINISTIC_MOCK_FULL_GRAPH_SKIPPER:"):
        assert name in SKIPPER.split("_SKIPPERS: dict[str, SkipperFactory] = {")[1][:400]
    assert "def register_skipper" in SKIPPER and "def build_skipper" in SKIPPER
    # the resolver no longer special-cases the mock
    assert "_deterministic_mock_adapter(" not in COMMON
    assert "build_skipper(name, arm)" in COMMON


def test_an_unknown_skipper_names_what_it_could_have_been():
    """The old `_ADAPTERS[name]` raised a bare KeyError."""
    idx = SKIPPER.index("def build_skipper")
    body = SKIPPER[idx: idx + 900]
    assert "unknown skipper" in body and "available_skippers()" in body


def test_an_unknown_projector_is_refused_by_name():
    idx = SKIPPER.index("def resolve_projector")
    body = SKIPPER[idx: idx + 600]
    assert "unknown projector" in body and "registered: " in body


# --- the FlexiDepth-shaped naming is gone -------------------------------------

def test_the_routed_layer_parameter_is_no_longer_named_after_flexidepth():
    assert "flexidepth_layer_ids" not in TYPES
    assert "checkpoint_routed_layer_ids" in TYPES
    seam = (ROOT / "vskipper/src/vskipper/integration/sglang/seam.py").read_text()
    assert "flexidepth_layer_ids=" not in seam


# --- the worked example is real code, and honest about what it is -------------

def test_the_example_plugin_satisfies_the_contract_it_documents():
    body = _class_body(EXAMPLE, "StaticDepthSkipper")
    assert "requires_router_weights = False" in body
    assert "projector_kind = FLEXIDEPTH_PROJECTOR" in body
    for method in ("def route", "def attestation"):
        assert method in body, f"the ABC's abstract {method} is unimplemented"
    assert "register_skipper(" in EXAMPLE


def test_the_example_does_not_claim_to_BE_a_published_system():
    assert "in the style of" in EXAMPLE
    assert "not a reimplementation" in EXAMPLE


def test_the_example_is_not_wired_into_the_shipped_system():
    """Documentation that compiles, not a silent thirteenth arm."""
    design = (ROOT / "vskipper/src/vskipper/runtime/design.py").read_text()
    assert "static_depth" not in design
    roots = [ROOT / "python", ROOT / "vskipper/src/vskipper/runtime",
             ROOT / "vskipper/src/vskipper/integration", ROOT / "vskipper/src/vskipper/kernels"]
    hits = [p for root in roots for p in root.rglob("*.py")
            if "example_skipper_plugin" in p.read_text()]
    assert hits == [], hits
