"""Full decode backend with sequential FlexiDepth conditional graph nodes."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Optional
from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
    FullCudaGraphBackend,
)
from sglang.srt.vpipe.graphs import (
    ConditionalGraphHelper,
)
from sglang.srt.vpipe.attestation import (
    binary_cohort_attestation,
    full_graph_commit_overlap_enabled,
    full_graph_conditional_max_rows,
    full_graph_conditional_production_all_run_enabled,
    full_graph_defer_project_kv_diagnostic_stage,
    full_graph_repair_group_size,
)
from sglang.srt.vpipe.common import (
    full_graph_contiguous_routed_qkv_config,
)
from sglang.srt.vpipe.common import (
    full_graph_compact_routed_qkv_enabled,
)
from sglang.srt.vpipe.config import (
    full_graph_defer_project_kv_enabled,
)
from sglang.srt.vpipe.env import (
    FD_CONDITIONAL_GRAPH_HELPER_ENV,
)
from sglang.srt.vpipe.graphs import (
    LlamaConditionalGraphCapture,
    capture_llama_flexidepth_conditional_graph,
)
from sglang.srt.vpipe.regime import (
    decode_regime_from_variant_label,
    regime_body_dispatches_stock_decode,
)


def replan_orphaned_flashinfer_decode_wrappers(
    attn_backend, forward_batch, decode_wrappers: list
) -> None:
    """Re-plan an orphaned FlashInfer cuda-graph decode wrapper for replay.

    Ported verbatim (D-250) out of the deleted FlexiDepthLayerRoutedAttnBackend.
    It never belonged to that class: the aliasing it repairs is a property of
    FLASHINFER's one-wrapper-per-batch-size registry, not of layer routing.
    Keeping it as a method forced the conditional-graph backend to require a
    VP-only backend, which is what made LAYER_ROUTED=0 crash V-dec-rs at boot.

    FlashInfer keeps ONE cuda-graph decode wrapper per batch size in
    ``decode_cuda_graph_metadata[bs]``. The regime-switch low band captures a
    stock decode graph bound to the wrapper in that slot; the sibling "skip"
    high-band capture at the SAME bs overwrites the slot, orphaning it. Left
    alone the orphan still carries its capture-time plan (built from the
    cuda-graph seq-len fill value of 1), so the captured decode reads a
    near-empty KV span and collapses into degenerate repetition. Re-plan the
    orphan against live sequence lengths, then restore the surviving slot so the
    high band's own plan is left byte-identical.
    """

    if not forward_batch.forward_mode.is_decode_or_idle():
        raise ValueError(
            "orphaned decode-graph wrapper re-plan requires decode mode"
        )
    metadata = attn_backend.decode_cuda_graph_metadata
    batch_size = int(forward_batch.batch_size)
    surviving = metadata[batch_size]
    metadata[batch_size] = decode_wrappers
    try:
        attn_backend.init_forward_metadata_out_graph(forward_batch)
    finally:
        metadata[batch_size] = surviving
class FlexiDepthConditionalCudaGraphBackend(FullCudaGraphBackend):
    """Capture one device-conditional Llama decode graph per batch bucket.

    W1 decode leg (I6b, fork option (b)): when the regime switch is on, each
    band bucket captures BOTH bodies keyed by ``ShapeKey.variant_label``
    (Strategy B): the ``prod_allrun`` low band as a STOCK production decode graph
    (the inherited full-graph capture of the plain base-Llama forward with
    production/flashinfer decode attention and no FlexiDepth hooks — byte- and
    speed-identical to a no-FD server's decode), and the ``skip`` high band as
    the M3.38 FlexiDepth conditional (or whole-model) body. Off / no-regime keeps
    the unchanged single-body dispatch.
    """

    requires_forward_batch = True

    def __init__(self, cuda_graph_runner, *, enable_memory_saver: bool = False):
        if enable_memory_saver:
            raise ValueError(
                "FlexiDepth conditional graphs do not support memory saver"
            )
        super().__init__(cuda_graph_runner, enable_memory_saver=False)
        self._runner = cuda_graph_runner
        self._captures: dict[Any, LlamaConditionalGraphCapture] = {}
        self._whole_model_shapes: set[Any] = set()
        # W1 low-band ("prod_allrun") STOCK production decode graphs, captured via
        # the inherited full-graph path and keyed by their regime variant_label.
        self._stock_decode_shapes: set[Any] = set()
        # The production (FlashInfer) decode-graph wrapper EACH stock low-band
        # graph is bound to, snapshotted at capture BEFORE the sibling "skip"
        # capture at the same bs evicts it from decode_cuda_graph_metadata[bs].
        # Re-planned at replay so the orphaned wrapper carries a live KV plan.
        self._stock_decode_graph_wrappers: dict[Any, list] = {}
        self._conditional_max_rows = full_graph_conditional_max_rows()
        helper_path = os.environ.get(FD_CONDITIONAL_GRAPH_HELPER_ENV, "")
        self._helper = ConditionalGraphHelper(helper_path)
        self._validate_runner()

    def _validate_runner(self) -> None:
        runner = self._runner
        model_runner = runner.model_runner
        if runner.enable_torch_compile:
            raise ValueError(
                "FlexiDepth conditional graphs do not support torch.compile"
            )
        if runner.enable_pdmux or runner.enable_two_batch_overlap:
            raise ValueError(
                "FlexiDepth conditional graphs do not support PDMux or TBO"
            )
        if not model_runner.spec_algorithm.is_none():
            raise ValueError(
                "FlexiDepth conditional graphs do not support speculative decode"
            )
        if model_runner.server_args.enable_lora:
            raise ValueError("FlexiDepth conditional graphs do not support LoRA")
        if runner.pp_size != 1 or runner.dp_size != 1:
            raise ValueError(
                "FlexiDepth conditional graphs currently require PP=DP=1"
            )

    def _regime_body(self, shape_key) -> Optional[str]:
        """Decode the W1 regime body from ``ShapeKey.variant_label``.

        FlexiDepth conditional graphs disallow LoRA (``_validate_runner``), so
        ``variant_label`` is exactly the regime body ("prod_allrun" / "skip") or
        ``None`` when the regime switch is off — never a LoRA-composed label.
        """

        return decode_regime_from_variant_label(shape_key.variant_label)

    def _capture_stock_low_band(
        self, shape_key, forward_fn, dummies, post_warmup_hook
    ) -> None:
        """Capture the STOCK production decode graph for a "prod_allrun" bucket.

        Fork option (b): the low band bypasses the FD conditional backend and
        runs the plain base-Llama decode forward with production (flashinfer)
        attention for EVERY layer and no FlexiDepth hooks — byte-identical and
        full-speed vs a no-FD server. Realized as the inherited full-graph
        capture with two per-pass stamps on the static ForwardBatch:

          * ``vp_fd_decode_dense`` flips ``flexidepth_phase_enabled`` off for this
            decode pass, so ``prepare_full_graph_batch`` no-ops and every
            ``LlamaDecoderLayer`` takes the dense fall-through (no FD qkv/o_proj/
            dense-weighted-MLP hooks — the divergence source the parity gate
            found);
          * ``fd_full_graph_force_production_attention`` routes all layers
            (incl. routed 16-31) to the flashinfer production backend, not triton.

        KV-complete by construction: the dense body's ``self_attn`` writes each
        token's own-layer K/V at every layer, exactly like stock production, so a
        skip<->stock band crossing always reads complete K/V (defer_project_kv is
        held off by the decode-leg validator).
        """

        if dummies is None:
            raise RuntimeError(
                "FlexiDepth low-band stock decode capture requires the static "
                "ForwardBatch"
            )
        dummies.vp_fd_decode_dense = True
        dummies.fd_full_graph_force_production_attention = True
        try:
            super().capture_one(
                shape_key,
                forward_fn,
                dummies=dummies,
                post_warmup_hook=post_warmup_hook,
            )
        finally:
            dummies.vp_fd_decode_dense = False
            dummies.fd_full_graph_force_production_attention = False
        self._stock_decode_shapes.add(shape_key)
        # Snapshot the production (FlashInfer) decode-graph wrapper this stock
        # graph is bound to, while decode_cuda_graph_metadata[bs] still holds it
        # (the sibling "skip" capture at the same bs will overwrite that slot).
        # replay() re-plans this wrapper so the captured decode reads a live KV
        # plan instead of the capture-time seq-len-fill plan.
        # D-250: the orphaned-wrapper workaround is FLASHINFER-SPECIFIC. FlashInfer
        # keeps ONE cuda-graph decode wrapper per batch size in
        # decode_cuda_graph_metadata[bs], so the sibling high-band capture at the
        # same bs orphans the low band's. A Triton decode backend has no wrapper
        # slot at all — nothing is aliased, nothing is orphaned, and there is
        # nothing to re-plan. Keying on the FlashInfer wrapper registry (rather
        # than requiring a VP-only method on the resolved backend) is what lets
        # the stock backend serve this path at all: requiring it is exactly the
        # coupling that made LAYER_ROUTED=0 crash V-dec-rs at boot.
        registry = getattr(
            self._runner.attn_backend, "decode_cuda_graph_metadata", None
        )
        self._stock_decode_graph_wrappers[shape_key] = (
            registry[int(shape_key.size)] if registry is not None else None
        )

    def capture_one(
        self,
        shape_key,
        forward_fn,
        dummies=None,
        post_warmup_hook=None,
    ) -> None:
        # W1 decode leg (fork option (b)): the "prod_allrun" low band dispatches
        # the STOCK production decode graph (bypassing the FD conditional
        # backend). Every (bucket, regime) carries its own variant_label, so the
        # low and high siblings are distinct graphs — no key collapsing.
        if regime_body_dispatches_stock_decode(self._regime_body(shape_key)):
            self._capture_stock_low_band(
                shape_key, forward_fn, dummies, post_warmup_hook
            )
            return
        # "skip" high band, or switch off: the M3.38 FlexiDepth body — whole-model
        # graph above conditional_max_rows, else the device-conditional graph.
        if (
            self._conditional_max_rows is not None
            and int(shape_key.size) > self._conditional_max_rows
        ):
            super().capture_one(
                shape_key,
                forward_fn,
                dummies=dummies,
                post_warmup_hook=post_warmup_hook,
            )
            self._whole_model_shapes.add(shape_key)
            return
        if dummies is None:
            raise RuntimeError(
                "FlexiDepth conditional capture requires the static ForwardBatch"
            )
        if self._capture_stream is None or self._pool is None:
            raise RuntimeError("FlexiDepth conditional capture session is inactive")
        capture = capture_llama_flexidepth_conditional_graph(
            model=self._runner.model_runner.model,
            forward_batch=dummies,
            attn_backend=self._runner.attn_backend,
            helper=self._helper,
            stream=self._capture_stream,
            pool=self._pool,
            post_warmup_hook=post_warmup_hook,
        )
        self._captures[shape_key] = capture
        self._graphs[shape_key] = capture.graph
        self._outputs[shape_key] = capture.output

    @contextmanager
    def replay_session(self):
        yield

    def plan_stock_low_band_decode_wrapper(self, shape_key, fb_view) -> None:
        """Plan a stock low-band graph's orphaned decode wrapper, once, at load_batch.

        Called by the decode runner's ``load_batch`` for a resolved stock
        low-band ("prod_allrun") replay key (the runner gates on
        :func:`regime_body_dispatches_stock_decode`). FlashInfer keeps ONE
        cuda-graph decode wrapper per ``bs`` in ``decode_cuda_graph_metadata[bs]``;
        the sibling "skip" capture at this ``bs`` evicted this graph's own
        production decode wrapper from that slot, so the low graph replays against
        the ORPHANED wrapper, not the surviving slot. Re-plan the orphan against
        the live sequence lengths (production attention only, reusing the runner's
        ``load_batch`` ``fb_view``) so the captured low-band decode replays a live
        KV plan.

        I6d vs I6c: I6c planned the surviving slot in ``load_batch`` and then
        re-planned the orphan AGAIN at replay (a redundant second FlashInfer decode
        plan + a second ``build_replay_fb_view`` per low-band step). I6d plans ONLY
        the orphan, here in ``load_batch``, reusing the fb_view — matching a no-FD
        server's single per-step decode plan, so :meth:`replay` needs no re-plan.

        Fails closed if the key has no captured stock low-band wrapper: Strategy B
        captures the stock low graph for every bucket, so a miss means the runner
        resolved a low-band key the backend never captured.
        """

        if shape_key not in self._stock_decode_graph_wrappers:
            raise RuntimeError(
                "regime switch resolved a stock low-band decode replay key with "
                f"no captured production decode wrapper: {shape_key}"
            )
        decode_wrappers = self._stock_decode_graph_wrappers[shape_key]
        # This call carries TWO responsibilities, and only the first is
        # FlashInfer-specific (D-250, review finding 1 — my earlier fix dropped
        # the second and turned a loud boot crash into SILENT degenerate output):
        #   (a) repair FlashInfer's orphaned per-bs wrapper (registry aliasing);
        #   (b) PLAN the decode metadata for this replay — universal, and on a
        #       Triton backend it is the per-step refill of the shared
        #       cuda_graph kv_indices / kv_indptr / num_kv_splits buffers. It is
        #       the ONLY replay-time metadata prep on the stock-low-band branch,
        #       so skipping it replays against the capture-time plan (seq-len
        #       fill value 1) and collapses attention into repetition.
        # So: always plan; swap the registry slot only when a registry exists.
        attn_backend = self._runner.attn_backend
        if decode_wrappers is not None:
            replan_orphaned_flashinfer_decode_wrappers(
                attn_backend, fb_view, decode_wrappers
            )
        else:
            attn_backend.init_forward_metadata_out_graph(fb_view)

    def replay(self, shape_key, static_forward_batch, **kwargs):
        if shape_key in self._captures:
            del static_forward_batch, kwargs
            self._captures[shape_key].graph.replay()
            return self._outputs[shape_key]
        # Stock low-band ("prod_allrun") graph or whole-model skip graph: both
        # live in self._graphs keyed by their own (bucket, regime) variant_label,
        # so the composed replay key resolves the right one directly. The stock
        # low band's orphaned decode wrapper was already planned once in the
        # runner's load_batch (see plan_stock_low_band_decode_wrapper), so this
        # replay is a plain captured-graph replay — no re-plan, no second fb_view.
        return super().replay(shape_key, static_forward_batch, **kwargs)

    def attestation(self) -> dict[str, Any]:
        stage_counts = {
            str(shape_key): len(capture.routed_layers)
            for shape_key, capture in self._captures.items()
        }
        branch_counts = {}
        run_bodies = 0
        mixed_bodies = 0
        for shape_key, capture in self._captures.items():
            counters = capture.body_execution_counts
            if counters is None:
                continue
            values = [
                [int(value) for value in row]
                for row in counters.detach().cpu().tolist()
            ]
            branch_counts[str(shape_key)] = values
            run_bodies += sum(row[0] for row in values)
            mixed_bodies += sum(row[1] for row in values)
        side_body_counts = {
            str(shape_key): capture.graph.attestation.side_body_count
            for shape_key, capture in self._captures.items()
        }
        join_body_counts = {
            str(shape_key): capture.graph.attestation.join_body_count
            for shape_key, capture in self._captures.items()
        }
        join_overlap_shapes = {
            str(shape_key): capture.graph.attestation.join_overlap
            for shape_key, capture in self._captures.items()
        }
        epilogue_body_counts = {
            str(shape_key): capture.graph.attestation.epilogue_body_count
            for shape_key, capture in self._captures.items()
        }
        repair_input_buffers = {
            str(shape_key): capture.repair_input_buffers
            for shape_key, capture in self._captures.items()
        }
        repair_kv_output_buffers = {
            str(shape_key): capture.repair_kv_output_buffers
            for shape_key, capture in self._captures.items()
        }
        repair_commit_graphs = {
            str(shape_key): capture.repair_commit_graphs
            for shape_key, capture in self._captures.items()
        }
        repair_commit_batched = {
            str(shape_key): capture.repair_commit_batched
            for shape_key, capture in self._captures.items()
        }
        repair_graph_pool_counts = {
            str(shape_key): len(capture.repair_graph_pools)
            for shape_key, capture in self._captures.items()
        }
        repair_capture_stream_counts = {
            str(shape_key): len(capture.repair_capture_streams)
            for shape_key, capture in self._captures.items()
        }
        repair_group_counts = {
            str(shape_key): capture.repair_group_count
            for shape_key, capture in self._captures.items()
        }
        deferred_project_kv = full_graph_defer_project_kv_enabled()
        compact_routed_qkv = full_graph_compact_routed_qkv_enabled()
        contiguous_routed_qkv, _, _, _ = (
            full_graph_contiguous_routed_qkv_config()
        )
        repair_group_size = full_graph_repair_group_size()
        repair_diagnostic_stage = (
            full_graph_defer_project_kv_diagnostic_stage()
        )
        production_all_run = (
            full_graph_conditional_production_all_run_enabled()
        )
        expected_run_body = (
            "production_flashinfer_attention_dense_weighted_mlp"
            if production_all_run
            else "masked_attention_filtered_run_only"
        )
        captured_run_bodies = {
            capture.run_body for capture in self._captures.values()
        }
        # W1 decode leg (fork option (b)): the "prod_allrun" low band is a STOCK
        # decode graph (not a conditional capture), so self._captures only ever
        # holds the "skip" high body — the single-body invariant is restored.
        if captured_run_bodies and captured_run_bodies != {expected_run_body}:
            raise RuntimeError(
                "FlexiDepth conditional run-body capture state changed"
            )
        return {
            "enabled": True,
            "backend": (
                "shape_adaptive_conditional_or_whole_model"
                if self._conditional_max_rows is not None
                else "sequential_device_conditional"
            ),
            "captured_shapes": len(self._graphs),
            "conditional_shapes": len(self._captures),
            "whole_model_shapes": len(self._whole_model_shapes),
            "stock_production_decode_shapes": len(self._stock_decode_shapes),
            "conditional_max_rows": self._conditional_max_rows,
            "shape_modes": {
                str(shape_key): (
                    "conditional_stages"
                    if shape_key in self._captures
                    else "stock_production_decode"
                    if shape_key in self._stock_decode_shapes
                    else "whole_model_graph"
                )
                for shape_key in self._graphs
            },
            "stage_counts": stage_counts,
            "branch_counters_enabled": bool(branch_counts),
            "branch_counts": branch_counts,
            "run_body_executions": run_bodies,
            "mixed_body_executions": mixed_bodies,
            "production_all_run_body_enabled": production_all_run,
            "production_all_run_body_executions": (
                run_bodies if production_all_run else 0
            ),
            "filtered_all_run_body_executions": (
                0 if production_all_run else run_bodies
            ),
            "stable_state_buffers": 2,
            "side_body_counts": side_body_counts,
            "join_body_counts": join_body_counts,
            "commit_overlap_enabled": full_graph_commit_overlap_enabled(),
            "join_overlap_shapes": join_overlap_shapes,
            "epilogue_body_counts": epilogue_body_counts,
            "deferred_project_kv": deferred_project_kv,
            "repair_diagnostic_stage": repair_diagnostic_stage,
            "repair_semantic_kv_complete": (
                not deferred_project_kv or repair_diagnostic_stage == "full"
            ),
            "repair_input_buffers": repair_input_buffers,
            "repair_kv_output_buffers": repair_kv_output_buffers,
            "repair_commit_graphs": repair_commit_graphs,
            "repair_commit_batched": repair_commit_batched,
            "binary_cohort": binary_cohort_attestation(),
            "repair_graph_pool_counts": repair_graph_pool_counts,
            "repair_capture_stream_counts": repair_capture_stream_counts,
            "repair_group_size": repair_group_size,
            "repair_group_counts": repair_group_counts,
            "repair_memory_isolation": (
                "group_private_streams_graph_pools_and_kv_outputs"
                if deferred_project_kv and repair_group_size > 1
                else "stage_private_streams_graph_pools_and_kv_outputs"
                if deferred_project_kv
                else None
            ),
            "repair_compute": (
                "stage_side_fixed_capacity_project_kv_cublas_mapped_overflow_rope"
                if deferred_project_kv
                and repair_diagnostic_stage == "full"
                and contiguous_routed_qkv
                else "grouped_side_mapped_project_kv_rope_into_stable_buffers"
                if deferred_project_kv
                and repair_diagnostic_stage == "full"
                and compact_routed_qkv
                else "side_qkv_rope_into_stable_private_buffers"
                if deferred_project_kv and repair_diagnostic_stage == "full"
                else f"diagnostic_{repair_diagnostic_stage}"
                if deferred_project_kv
                else None
            ),
            "repair_cache_write": (
                "joined_standard_writer_non_project_to_padding_slot_0"
                if deferred_project_kv and repair_diagnostic_stage == "full"
                else None
            ),
            "repair_topology": (
                "route_group_fork_compact_side_compute_overlapped_commit_suffix_evidence"
                if deferred_project_kv
                and repair_diagnostic_stage == "full"
                and compact_routed_qkv
                and full_graph_commit_overlap_enabled()
                else "route_prefix_fork_side_compute_overlapped_commit_suffix_evidence"
                if deferred_project_kv
                and repair_diagnostic_stage == "full"
                and full_graph_commit_overlap_enabled()
                else "route_group_fork_compact_side_compute_join_cache_commit_suffix"
                if deferred_project_kv
                and repair_diagnostic_stage == "full"
                and compact_routed_qkv
                else "route_prefix_fork_side_compute_join_cache_commit_suffix"
                if deferred_project_kv and repair_diagnostic_stage == "full"
                else "route_prefix_fork_diagnostic_side_compute_suffix_join"
                if deferred_project_kv
                else None
            ),
            "foreground_projection": (
                "mapped_run_qkv"
                if compact_routed_qkv
                else "full_batch_qkv_with_run_only_cache_write"
                if deferred_project_kv
                else "full_batch_qkv_with_complete_cache_write"
            ),
            "repair_barrier": (
                "foreground_side_join_before_cache_commit_and_logits"
                if deferred_project_kv and repair_diagnostic_stage == "full"
                else "suffix_joins_diagnostic_side_compute"
                if deferred_project_kv
                else None
            ),
            "run_body": expected_run_body,
            "host_route_readback": False,
            "cuda_runtime_version": self._helper.cuda_runtime_version,
        }

    def cleanup(self) -> None:
        for capture in self._captures.values():
            capture.graph.close()
            for child in capture.child_graphs:
                child.reset()
        self._captures.clear()
        self._graphs.clear()
        self._outputs.clear()
        self._whole_model_shapes.clear()
        self._stock_decode_shapes.clear()
        self._stock_decode_graph_wrappers.clear()
        self._pool = None
