"""The skipper contract — logical actions and the adapter base classes.

Kept in its own module so `skipper.py` (FlexiDepth, RandomSkip) can depend on
it without a cycle, and so a future non-binary adapter has one seam to extend.

``LogicalAction`` is a skipper's decision for one token at one layer. The
production executor supports exactly the binary pair ``RUN`` / ``PROJECT_ONLY``;
every other action is declared so an adapter emitting one FAILS CLOSED rather
than being silently lowered into the binary executor. (The AdaSkip sublayer
adapter and its ``sublayer`` execution kind were dropped 2026-08-23 by owner
ruling — see the removed-feature register; the fail-closed rejection of any
non-binary execution kind is what remains of that seam.)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum, IntEnum
from typing import Any, Mapping, Optional
import torch
from vskipper.runtime.env import (
    RUN_PROJECT_EXECUTION,
)


FULL_GRAPH_ACTION_CONTRACT = "vp-full-graph-logical-action-v1"
class LogicalAction(IntEnum):
    """Skipper-visible action, independent of the scheduler's queue lane.

    The integer values are part of the decision-tape format.  ``RouteTag`` is
    intentionally coarser: every conditional action currently enters the JUMP
    queue, while the logical action records what the executor must preserve.
    """

    RUN = 0
    PROJECT_ONLY = 1
    REUSE = 2
    SKIP_ATTN = 3
    SKIP_MLP = 4
    VETO = 5
_LOGICAL_ACTION_CODES = {
    action.name.lower(): int(action) for action in LogicalAction
}
@dataclass(frozen=True, slots=True)
class FullGraphActionBatch:
    """One policy's device-resident decisions for a routed layer.

    ``branch_weights`` remains the exact continuous payload consumed by the
    executor.  ``write_run_storage`` lowers the supported logical actions to
    the existing boolean tape, where True means RUN and False means
    PROJECT_ONLY.  Passing ``out`` preserves the allocation-free tape write.
    """

    adapter_name: str
    branch_weights: Any
    supported_actions: frozenset[LogicalAction]
    threshold: Optional[float] = None
    explicit_run_mask: Any = None
    forced_action: Optional[LogicalAction] = None
    payload_semantics: str = ""
    execution_kind: str = RUN_PROJECT_EXECUTION

    def __post_init__(self) -> None:
        if not self.adapter_name:
            raise ValueError("full-graph action batch requires an adapter name")
        shape = getattr(self.branch_weights, "shape", None)
        if shape is None or len(shape) != 2 or int(shape[1]) != 1:
            raise ValueError("full-graph branch weights must have shape [rows, 1]")
        if self.execution_kind == RUN_PROJECT_EXECUTION:
            self._validate_run_project(shape)
        else:
            raise ValueError(
                "this executor supports only the binary RUN/PROJECT execution "
                f"kind; got {self.execution_kind!r} (non-binary kinds fail "
                "closed rather than being silently lowered)"
            )
        if not self.payload_semantics:
            raise ValueError("full-graph action batch requires payload semantics")

    def _validate_run_project(self, shape: Any) -> None:
        if self.supported_actions != frozenset(
            (LogicalAction.RUN, LogicalAction.PROJECT_ONLY)
        ):
            raise ValueError(
                "RUN/PROJECT full-graph execution supports exactly RUN and "
                "PROJECT_ONLY"
            )
        if self.explicit_run_mask is not None:
            if getattr(self.explicit_run_mask, "shape", None) != shape:
                raise ValueError(
                    "explicit full-graph actions must match the branch-weight rows"
                )
            if self.threshold is not None:
                raise ValueError(
                    "full-graph actions cannot set both an explicit mask and threshold"
                )
            if self.forced_action is not None:
                raise ValueError(
                    "full-graph actions cannot set both an explicit mask and "
                    "forced action"
                )
        elif self.threshold is None and self.forced_action is None:
            raise ValueError(
                "full-graph actions require a threshold, explicit mask, or "
                "forced action"
            )
        if self.forced_action is not None and self.forced_action not in (
            LogicalAction.RUN,
            LogicalAction.PROJECT_ONLY,
        ):
            raise ValueError(
                "the production full-graph executor cannot lower the forced action "
                f"{self.forced_action.name}"
            )

    @property
    def route_weights(self) -> Any:
        """Compatibility name used by the existing FlexiDepth executor."""

        if self.execution_kind != RUN_PROJECT_EXECUTION:
            raise RuntimeError(
                "non-binary actions do not have RUN/PROJECT weights"
            )
        return self.branch_weights

    def write_run_storage(self, out: Any = None) -> Any:
        """Lower RUN/PROJECT_ONLY into the legacy boolean execution storage."""

        import torch

        if self.execution_kind != RUN_PROJECT_EXECUTION:
            raise RuntimeError(
                "non-binary actions cannot be lowered to the boolean tape"
            )

        if out is not None:
            if getattr(out, "shape", None) != self.branch_weights.shape:
                raise RuntimeError(
                    "full-graph action weights do not match the route-tape rows"
                )
            if out.dtype != torch.bool:
                raise RuntimeError("full-graph route-tape storage must be boolean")
            if out.device != self.branch_weights.device:
                raise RuntimeError(
                    "full-graph action weights and route-tape storage must share "
                    "a device"
                )

        if self.forced_action is not None:
            value = self.forced_action == LogicalAction.RUN
            if out is None:
                return torch.full_like(
                    self.branch_weights, value, dtype=torch.bool
                )
            out.fill_(value)
            return out

        if self.explicit_run_mask is not None:
            if self.explicit_run_mask.dtype != torch.bool:
                raise RuntimeError("explicit full-graph actions must be boolean")
            if self.explicit_run_mask.device != self.branch_weights.device:
                raise RuntimeError(
                    "explicit actions and branch weights must share a device"
                )
            if out is None:
                return self.explicit_run_mask
            out.copy_(self.explicit_run_mask)
            return out

        if out is None:
            return self.branch_weights > self.threshold
        torch.gt(self.branch_weights, self.threshold, out=out)
        return out

PROJECTOR_CONTRACT = "vp-project-only-payload-v1"


class ProjectorAdapter(ABC):
    """Supplies the PROJECT_ONLY payload for a routed layer.

    Split out of the policy 2026-09-11 (D-701). The two are different plugins
    answering different questions:

      * a ``FullGraphSkipperAdapter`` decides WHICH tokens take PROJECT_ONLY;
      * a ``ProjectorAdapter`` decides WHAT a PROJECT_ONLY row computes.

    Before the split both lived behind one boolean, ``requires_flexidepth_weights``,
    so a policy that brings its own gate -- the deterministic mock hashes, it reads
    no trained router -- was still described as *requiring FlexiDepth weights*,
    and the regime-switch capability check refused any adapter that said otherwise
    even though what the regime switch actually needs is a PROJECTOR (something
    must produce the layer output and its K/V for a skipped row; K/V-completeness
    is not optional).

    Exactly ONE implementation exists (``flexidepth``), and this refactor is
    deliberately behaviour-preserving: the seam still builds and loads the same
    ``FDProj`` module, and the executor bodies still receive and call that module
    directly, so nothing is added inside the CUDA-graph capture region. What
    changes is that the requirement is now DECLARED by name instead of inferred
    from a FlexiDepth-shaped boolean.
    """

    name: str
    #: Weight tensors, per routed layer, this projector loads from the checkpoint.
    checkpoint_tensor_suffixes: tuple[str, ...] = ()

    @abstractmethod
    def attestation(self) -> dict[str, Any]:
        """Describe the payload this projector computes for a skipped row."""


class FullGraphSkipperAdapter(ABC):
    """Policy decision interface consumed by the production graph executor."""

    name: str
    supported_actions: frozenset[LogicalAction]
    requires_stable_request_ids: bool = False
    route_digest_requires_stable_request_ids: bool = False
    requires_request_slots: bool = False
    execution_kind: str = RUN_PROJECT_EXECUTION

    #: Does the POLICY read trained gate weights from the checkpoint? FlexiDepth
    #: does; the deterministic mock hashes (request_id, token_epoch) and does not.
    #: Declarative only -- see `requires_flexidepth_weights` below for why the
    #: mock arm still LOADS the router today.
    requires_router_weights: bool = True
    #: Which `ProjectorAdapter` supplies the PROJECT_ONLY payload. Every binary
    #: RUN/PROJECT policy needs one: a skipped row must still produce this layer's
    #: output and its own K/V.
    projector_kind: str = "flexidepth"

    @property
    def requires_flexidepth_weights(self) -> bool:
        """Does serving this adapter need FlexiDepth checkpoint weights loaded?

        Derived, not declared, so the five call sites that ask this question keep
        asking it unchanged. TRUE for both shipped adapters: FlexiDepth needs the
        router AND the projector; the mock needs only the projector, which is the
        FlexiDepth one.

        The mock arm therefore still loads the ROUTER it never reads. Skipping
        that load is a behaviour change, not a cleanup: the router occupies HBM,
        and `server.max_total_num_tokens` is a DECLARED cross-arm field measured
        at -0.98 % on the treatment arm (D-699). Dropping the unused weights would
        move it, so it is left alone until a campaign is ready to re-measure.
        """

        return self.requires_router_weights or self.projector_kind == "flexidepth"

    def routed_layer_ids(
        self,
        *,
        num_hidden_layers: int,
        checkpoint_routed_layer_ids: tuple[int, ...],
        model_identity: Optional[Mapping[str, str]] = None,
    ) -> tuple[int, ...]:
        """Return ordered layers owned by this adapter in the model forward."""

        if any(
            layer_id < 0 or layer_id >= num_hidden_layers
            for layer_id in checkpoint_routed_layer_ids
        ):
            raise ValueError("checkpoint routed layers are outside the model")
        return checkpoint_routed_layer_ids

    def attention_routed_layer_ids(
        self,
        *,
        num_hidden_layers: int,
        checkpoint_routed_layer_ids: tuple[int, ...],
        model_identity: Optional[Mapping[str, str]] = None,
    ) -> tuple[int, ...]:
        """Return layers whose attention backend consumes component masks."""

        return self.routed_layer_ids(
            num_hidden_layers=num_hidden_layers,
            checkpoint_routed_layer_ids=checkpoint_routed_layer_ids,
        )

    def prepare_batch(
        self,
        *,
        hidden_states: Any,
        request_ids: Any,
        token_epochs: Any,
        valid_rows: Any,
        route_layer_order: tuple[int, ...],
        request_slots: Any = None,
        phase: Optional[str] = None,
        batch_request_ids: Any = None,
        batch_request_slots: Any = None,
    ) -> Any:
        """Build graph-resident policy state once per routed batch."""

        del (
            hidden_states,
            request_ids,
            request_slots,
            token_epochs,
            valid_rows,
            route_layer_order,
            phase,
            batch_request_ids,
            batch_request_slots,
        )
        return None

    def finalize_batch(self, *, batch_state: Any) -> None:
        """Commit optional graph-resident policy state once per forward."""

        del batch_state

    def validate_runtime_capacities(
        self,
        *,
        request_pool_slots: int,
        decode_graph_rows: int,
    ) -> None:
        """Fail before serving when adapter state cannot cover the runtime."""

        del request_pool_slots, decode_graph_rows

    def reset_runtime_state(self) -> None:
        """Clear adapter-owned state after graph capture."""

    @abstractmethod
    def route(
        self,
        hidden_states: Any,
        *,
        router: Any,
        forced_action: Optional[LogicalAction] = None,
        layer_id: Optional[int] = None,
        batch_state: Any = None,
    ) -> FullGraphActionBatch:
        """Return device decisions without executing either branch."""

    @abstractmethod
    def attestation(self) -> dict[str, Any]:
        """Describe exact decision and payload semantics for result manifests."""
