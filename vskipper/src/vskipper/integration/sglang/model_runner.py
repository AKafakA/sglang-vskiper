"""SGLang model-runner activation checks and dispatch accounting.

These hooks preserve their original position relative to backend initialization,
graph eligibility, execution and the runner's finally-based stamp cleanup.
"""
import os
import torch

from vskipper.runtime.coverage import (
    account_covered_dispatch,
    coverage_dense_enabled,
    fd_skip_decode_deployed,
    stamp_coverage_dense,
)
from vskipper.runtime.common import (
    REQUEST_BODY_FD,
    REQUEST_BODY_STOCK,
    VP_BODY_MIXED,
    regime_switch_config,
)
from vskipper.runtime.regime import pinned_decode_body_of

def validate_attention_backends(self, logger):
    # Fail-closed FlexiDepth backend assertion (owner ruling, 2026-08-18;
    # D-186/D-197/D-248 lineage): only the triton attention kernels read
    # the FlexiDepth run mask. With FD weights active, any other resolved
    # backend would silently ignore routing (masked attention becomes a
    # real contribution) — refuse to boot instead. This asserts; it never
    # selects or overrides a backend.
    # Gate on the FULL-GRAPH path, not merely on FD weights being present.
    # The invariant is that only the triton kernels read the FlexiDepth run
    # mask -- and the run mask is set exclusively by vpipe/executor.py, i.e.
    # the full_graph dispatch. vpipe/eager.py never sets it: direct_eager
    # runs self_attn for every row and masks the returned OUTPUT itself, so
    # it is backend-agnostic. Keying on SGLANG_FD_WEIGHTS alone refused a
    # supported direct_eager launch on any non-triton backend.
    from vskipper.runtime.common import flexidepth_execution_mode
    from vskipper.runtime.design import flexidepth_weights_path, skipper_deployed
    from vskipper.runtime.env import FD_EXECUTION_FULL_GRAPH

    # [D-734] Was keyed on SGLANG_FD_WEIGHTS, which D-609 stopped exporting -- both
    # assertions below were inert in every served cell since 2026-09-09. The design
    # (arm + host config) is the only source of "a skipper with weights is deployed".
    if (
        skipper_deployed()
        and flexidepth_weights_path().strip()
        and flexidepth_execution_mode() == FD_EXECUTION_FULL_GRAPH
    ):
        resolved_backends = {
            "prefill": self.prefill_attention_backend_str,
            "decode": self.decode_attention_backend_str,
        }
        non_triton = {
            phase: name
            for phase, name in resolved_backends.items()
            if name != "triton"
        }
        if non_triton:
            raise ValueError(
                "FlexiDepth full-graph execution is active but the "
                f"resolved attention backends are {resolved_backends}; "
                "only triton reads the FlexiDepth run mask. Pin "
                "--attention-backend triton (and per-phase flags) — "
                "refusing to boot rather than silently dropping routing."
            )
        # Fail-closed MIXED-chunk assertion (W4/F5): a MIXED batch
        # (decode rows folded into a chunked-prefill pass) reaches
        # flexidepth_forward_phase as is_extend()==True and is phased
        # "prefill", so its decode rows would be routed under prefill
        # semantics (or silently run dense when prefill is inactive).
        # Either way the route contract breaks — refuse to boot.
        # [D-738] The decode band the arm DECLARES must be the band the roofline rule
        # gives for THIS device; otherwise refuse to boot (a re-tuned band would
        # otherwise serve silently on a second card).
        from vskipper.runtime.common import regime_switch_config as _regime_cfg_fn
        from vskipper.runtime.design import active_arm as _active_arm
        from vskipper.kernels.kernel import canonical_device_key as _canon
        from vskipper.runtime.roofline import (
            arm_kv_rule_inputs,
            assert_kv_band_follows_rule,
            band_device_key,
        )

        _rs = _regime_cfg_fn()
        if _rs is not None and _rs.decode.enabled and _rs.decode.kv_criterion:
            _band_policy = _active_arm().get("decode_kv_band_policy", "rule")
            if _band_policy == "shared":
                # [D-778] A DECLARED deviation: the arm serves the global band under its own
                # inputs (the Qwen shared-band posture). Attested as decode_kv_band_policy.
                logger.warning(
                    "[D-778] decode_kv_band_policy=shared: serving band (exit=%s, enter=%s) "
                    "as a declared deviation from the roofline rule for this arm",
                    _rs.decode.exit_kv_tokens, _rs.decode.enter_kv_tokens,
                )
            elif _band_policy == "gate":
                # [D-849] A GATE-ONLY declared deviation: a tiny band so the pinning
                # correctness checks can drive both pins in a 40-request smoke.
                logger.warning(
                    "decode_kv_band_policy=gate: serving band (exit=%s, enter=%s) "
                    "as a gate-only declared deviation from the roofline rule (never a paper arm)",
                    _rs.decode.exit_kv_tokens, _rs.decode.enter_kv_tokens,
                )
            elif _band_policy != "rule":
                raise ValueError(f"unknown decode_kv_band_policy {_band_policy!r}")
            else:
                assert_kv_band_follows_rule(
                    served_exit_kv_tokens=_rs.decode.exit_kv_tokens,
                    served_enter_kv_tokens=_rs.decode.enter_kv_tokens,
                    device_key=band_device_key(
                        _canon(torch.cuda.get_device_name(self.device)),
                        torch.cuda.get_device_properties(self.device).total_memory,
                    ),
                    **arm_kv_rule_inputs(_active_arm()),
                )
        if self.server_args.enable_mixed_chunk:
            raise ValueError(
                "FlexiDepth is active (a skipper with weights is deployed) but "
                "--enable-mixed-chunk is on; MIXED batches phase their "
                "decode rows as prefill under FlexiDepth routing. "
                "Disable mixed chunking — refusing to boot rather than "
                "mis-phasing decode rows."
            )


def prepare_decode_dispatch(self, forward_batch, can_run_graph):
    if self._vp_runtime_enabled and (
        os.environ.get("SGLANG_FD_EXECUTION_MODE", "").strip().lower()
        == "full_graph"
    ):
        from vskipper.runtime.attestation import (
            record_model_runner_dispatch,
        )

        record_model_runner_dispatch(self, forward_batch, can_run_graph)

    # (c3) C-A coverage stamp at the dispatch seam: consume the
    # ALREADY-computed can_run_graph local (the runner's own predicate;
    # never a re-derived bs <= max_bs) and decide the decode BODY for
    # this exact executing batch. An uncovered pass — rows above the
    # realized ladder, an eligibility veto, or no runner at all — is
    # stamped dense fail-closed (R-E): flexidepth_phase_enabled then
    # returns False for the pass and the KV-complete base-Llama dense
    # fall-through runs with production attention (same pairing as the
    # W1 low band). Production (no FD hooks) never enters this block —
    # zero stamps, zero counters, byte-identical (E-C3). The seam
    # observe keeps the W1 band + its attestation live through eager
    # episodes; level-triggered and idempotent in rows (F22).
    coverage_stamped = False
    stock_eager_stamped = False
    if (
        forward_batch.forward_mode.is_decode()
        and fd_skip_decode_deployed()
        and coverage_dense_enabled()
    ):
        runner = self.decode_cuda_graph_runner
        # [D-849] A pinned (uniform) pass records its pinned body; an
        # unpinned pass advances the band exactly as in version 1.
        pinned_body = pinned_decode_body_of(forward_batch)
        band_body = (
            runner._vp_regime_dispatch.observe_or_pin(
                int(forward_batch.batch_size),
                int(forward_batch.seq_lens_sum),
                pinned_body,
            )
            if runner is not None
            else pinned_body
        )
        coverage_stamped = stamp_coverage_dense(
            forward_batch,
            runner=runner,
            can_run_graph=can_run_graph,
            w1_active=regime_switch_config() is not None,
        )
        if coverage_stamped and forward_batch.vp_body in (REQUEST_BODY_FD, VP_BODY_MIXED):
            # A pin violation: fd-pinned rows served by the dense
            # fall-through. Counted; verify_skipping_executed refuses > 0.
            fd_rows = (
                int(forward_batch.batch_size)
                if forward_batch.vp_body == REQUEST_BODY_FD
                else sum(1 for p in forward_batch.vp_body_rows if p == REQUEST_BODY_FD)
            )
            self._vp_split_counters["coverage_dense_violation_rows"] += fd_rows
        if not coverage_stamped:
            account_covered_dispatch(forward_batch, band_body)
    elif (
        forward_batch.forward_mode.is_decode()
        and forward_batch.vp_body == REQUEST_BODY_STOCK
        and not can_run_graph
        and fd_skip_decode_deployed()
    ):
        # [D-849] Stock-pinned rows outside graph coverage with the (c3)
        # stamp off: honour the pin on the eager path with the same
        # pairing the captured stock graph uses.
        forward_batch.vp_fd_decode_dense = True
        forward_batch.fd_full_graph_force_production_attention = True
        forward_batch.vp_seam_batch_routed = None
        forward_batch.vp_seam_batch_eager = None
        stock_eager_stamped = True
        self._vp_split_counters["stock_eager_passes"] += 1
    return coverage_stamped, stock_eager_stamped
