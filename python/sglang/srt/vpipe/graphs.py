"""Conditional CUDA-graph capture for the routed decode body.

The routed decode body is only efficient while captured. Above the captured
ceiling it runs eager and un-amortised, which is the occupancy runaway: the cap
must cover the realised occupancy, and its derivation must read the
req_to_token_pool ceiling rather than assume one."""

from __future__ import annotations

from sglang.srt.layers.dp_attention import set_dp_buffer_len, set_is_extend_in_batch
from typing import Callable, Mapping
from pathlib import Path
from dataclasses import dataclass, replace
import ctypes
import os
from typing import Any, Callable, Optional
import torch
from sglang.srt.vpipe.batch import (
    finalize_full_graph_batch,
    prepare_full_graph_batch,
)
from sglang.srt.vpipe.common import (
    full_graph_compact_routed_qkv_enabled,
)
from sglang.srt.vpipe.config import (
    full_graph_defer_project_kv_enabled,
)
from sglang.srt.vpipe.executor import (
    fd_execute_prepared_layer_route_full_graph,
)
from sglang.srt.vpipe.routing import (
    fd_prepare_layer_route_full_graph,
)
from sglang.srt.vpipe.attestation import (
    full_graph_commit_overlap_enabled,
    full_graph_conditional_branch_counters_enabled,
    full_graph_conditional_production_all_run_enabled,
    full_graph_defer_project_kv_diagnostic_stage,
    full_graph_repair_group_size,
)
from sglang.srt.vpipe.common import (
    _fixed_capacity_mapped_linear,
    full_graph_contiguous_routed_qkv_config,
)
from sglang.srt.vpipe.env import (
    FD_BATCHED_COMMIT_ENV,
)
from sglang.srt.vpipe.mlp_compact import (
    _compact_capacity,
)
from sglang.srt.vpipe.routing import (
    FullGraphPreparedLayerRoute,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.cuda_utils import (
    checkCudaErrors,
)


try:
    from cuda.bindings import runtime as cuda_rt
except ImportError:
    cuda_rt = None
_LAYER_BACKING_ALIGN_BYTES = 256
@dataclass(frozen=True, slots=True)
class ConditionalGraphAttestation:
    cuda_runtime_version: int
    predicate_location: str = "device_int32"
    condition_update: str = "device_kernel"
    branch_count: int = 2
    stage_count: int = 1
    pre_body_count: int = 0
    side_body_count: int = 0
    join_body_count: int = 0
    join_overlap: bool = False
    epilogue_body_count: int = 0
    host_route_readback: bool = False
def fd_project_kv_repair_cache_locations(
    cache_locations: torch.Tensor,
    project_rows: torch.Tensor,
) -> torch.Tensor:
    """Map every non-PROJECT repair row to the reserved padding slot."""

    if cache_locations.shape != project_rows.shape:
        raise RuntimeError("PROJECT repair cache locations are not aligned")
    if project_rows.dtype != torch.bool:
        raise RuntimeError("PROJECT repair row mask must be boolean")
    if cache_locations.device != project_rows.device:
        raise RuntimeError("PROJECT repair cache tensors must share a device")
    return torch.where(
        project_rows,
        cache_locations,
        cache_locations.new_zeros(()),
    )
@dataclass(frozen=True, slots=True)
class ConditionalGraphStage:
    """One route prefix followed by an exactly-one-of-two device branch."""

    predicate: torch.Tensor
    true_body: torch.cuda.CUDAGraph
    false_body: torch.cuda.CUDAGraph
    route_prefix: Optional[torch.cuda.CUDAGraph] = None
    side_bodies: tuple[torch.cuda.CUDAGraph, ...] = ()
class ConditionalCudaGraph:
    """Own one parent graph containing sequential device-selected bodies."""

    def __init__(
        self,
        *,
        graph: Any,
        executable: Any,
        condition_handles: tuple[Any, ...],
        setter_graphs: tuple[torch.cuda.CUDAGraph, ...],
        helper: ConditionalGraphHelper,
        retained_graphs: tuple[torch.cuda.CUDAGraph, ...],
        pre_body_count: int,
        side_body_count: int,
        join_body_count: int,
        join_overlap: bool = False,
        epilogue_body_count: int = 0,
        branch_count: int = 2,
    ) -> None:
        self._graph = graph
        self._executable = executable
        self._condition_handles = condition_handles
        self._setter_graphs = setter_graphs
        self._helper = helper
        self._retained_graphs = retained_graphs
        self._closed = False
        self.attestation = ConditionalGraphAttestation(
            cuda_runtime_version=helper.cuda_runtime_version,
            branch_count=branch_count,
            stage_count=len(condition_handles),
            pre_body_count=pre_body_count,
            side_body_count=side_body_count,
            join_body_count=join_body_count,
            join_overlap=join_overlap,
            epilogue_body_count=epilogue_body_count,
        )

    @staticmethod
    def _require_runtime() -> None:
        if cuda_rt is None:
            raise ConditionalGraphUnavailable("cuda.bindings is unavailable")
        required = (
            "cudaGraphCreate",
            "cudaGraphConditionalHandleCreate",
            "cudaGraphAddChildGraphNode",
            "cudaGraphAddNode",
            "cudaGraphInstantiateWithFlags",
            "cudaGraphLaunch",
        )
        missing = [name for name in required if not hasattr(cuda_rt, name)]
        if missing:
            raise ConditionalGraphUnavailable(
                "cuda.bindings lacks conditional graph APIs: " + ", ".join(missing)
            )
        try:
            torch.cuda.CUDAGraph(keep_graph=True)
        except (TypeError, RuntimeError) as error:
            raise ConditionalGraphUnavailable(
                "PyTorch does not expose mutable raw CUDA graphs"
            ) from error

    @staticmethod
    def _raw_graph(graph: torch.cuda.CUDAGraph) -> Any:
        if not hasattr(graph, "raw_cuda_graph"):
            raise ConditionalGraphUnavailable(
                "captured child graph does not expose raw_cuda_graph"
            )
        return graph.raw_cuda_graph()


    @classmethod
    def compose_stages(
        cls,
        *,
        stages: tuple[ConditionalGraphStage, ...],
        helper: ConditionalGraphHelper,
        stream: torch.cuda.Stream,
        pool: Any,
        prefix: Optional[torch.cuda.CUDAGraph] = None,
        join: Optional[torch.cuda.CUDAGraph] = None,
        suffix: Optional[torch.cuda.CUDAGraph] = None,
        overlap_join: bool = False,
        epilogue: Optional[torch.cuda.CUDAGraph] = None,
    ) -> "ConditionalCudaGraph":
        """Compose ordered route/branch stages into one replayable graph.

        With ``overlap_join`` the join body becomes a peer of the suffix
        (both depend on the last stage plus every side node; neither gates
        the other), and the optional ``epilogue`` body depends on both —
        the placement for work that must observe the join's effects (e.g.
        K/V readiness evidence) without serializing join before suffix.
        Graph-launch completion still fences every leaf, so stream order
        after replay observes all bodies.
        """

        cls._require_runtime()
        if not stages:
            raise ValueError("conditional graph requires at least one stage")
        if overlap_join and join is None:
            raise ValueError("overlap_join requires a join body")
        if epilogue is not None and not overlap_join:
            raise ValueError(
                "epilogue requires overlap_join; serialized joins already "
                "order the suffix after the join"
            )
        for index, stage in enumerate(stages):
            predicate = stage.predicate
            if predicate.device.type != "cuda":
                raise ValueError(
                    f"conditional graph stage {index} predicate must be on CUDA"
                )
            if predicate.dtype != torch.int32 or predicate.numel() != 1:
                raise ValueError(
                    f"conditional graph stage {index} predicate must be one "
                    "int32 value"
                )

        graph = None
        executable = None
        setter_graphs: list[torch.cuda.CUDAGraph] = []
        condition_handles: list[Any] = []
        side_nodes: list[Any] = []
        try:
            graph = checkCudaErrors(cuda_rt.cudaGraphCreate(0))
            assign_default = (
                cuda_rt.cudaGraphConditionalHandleFlags.cudaGraphCondAssignDefault
            )
            dependency = None
            if prefix is not None:
                dependency = checkCudaErrors(
                    cuda_rt.cudaGraphAddChildGraphNode(
                        graph, None, 0, cls._raw_graph(prefix)
                    )
                )

            for stage in stages:
                if stage.route_prefix is not None:
                    route_dependencies = (
                        [dependency] if dependency is not None else None
                    )
                    dependency = checkCudaErrors(
                        cuda_rt.cudaGraphAddChildGraphNode(
                            graph,
                            route_dependencies,
                            len(route_dependencies or ()),
                            cls._raw_graph(stage.route_prefix),
                        )
                    )

                side_dependencies = (
                    [dependency] if dependency is not None else None
                )
                for side_body in stage.side_bodies:
                    # Side work consumes route-prefix state but does not gate the
                    # next foreground stage. The suffix joins every side node.
                    side_nodes.append(
                        checkCudaErrors(
                            cuda_rt.cudaGraphAddChildGraphNode(
                                graph,
                                side_dependencies,
                                len(side_dependencies or ()),
                                cls._raw_graph(side_body),
                            )
                        )
                    )

                condition = checkCudaErrors(
                    cuda_rt.cudaGraphConditionalHandleCreate(
                        graph, 0, assign_default
                    )
                )
                condition_handles.append(condition)
                setter_graph = torch.cuda.CUDAGraph(keep_graph=True)
                setter_graphs.append(setter_graph)
                with torch.cuda.stream(stream):
                    setter_graph.capture_begin(pool=pool)
                    helper.launch(int(condition), stage.predicate, stream)
                    setter_graph.capture_end()

                setter_dependencies = (
                    [dependency] if dependency is not None else None
                )
                setter_node = checkCudaErrors(
                    cuda_rt.cudaGraphAddChildGraphNode(
                        graph,
                        setter_dependencies,
                        len(setter_dependencies or ()),
                        cls._raw_graph(setter_graph),
                    )
                )

                params = cuda_rt.cudaGraphNodeParams()
                params.type = cuda_rt.cudaGraphNodeType.cudaGraphNodeTypeConditional
                params.conditional.handle = condition
                params.conditional.type = (
                    cuda_rt.cudaGraphConditionalNodeType.cudaGraphCondTypeIf
                )
                params.conditional.size = 2
                conditional_node = checkCudaErrors(
                    cuda_rt.cudaGraphAddNode(
                        graph, [setter_node], None, 1, params
                    )
                )
                bodies = params.conditional.phGraph_out
                if bodies is None:
                    raise ConditionalGraphUnavailable(
                        "CUDA did not return conditional body graphs"
                    )
                checkCudaErrors(
                    cuda_rt.cudaGraphAddChildGraphNode(
                        bodies[0], None, 0, cls._raw_graph(stage.true_body)
                    )
                )
                checkCudaErrors(
                    cuda_rt.cudaGraphAddChildGraphNode(
                        bodies[1], None, 0, cls._raw_graph(stage.false_body)
                    )
                )
                dependency = conditional_node

            join_node = None
            if join is not None:
                join_dependencies = [dependency, *side_nodes]
                join_node = checkCudaErrors(
                    cuda_rt.cudaGraphAddChildGraphNode(
                        graph,
                        join_dependencies,
                        len(join_dependencies),
                        cls._raw_graph(join),
                    )
                )
                if not overlap_join:
                    dependency = join_node
                    side_nodes.clear()

            suffix_node = None
            if suffix is not None:
                suffix_dependencies = [dependency, *side_nodes]
                suffix_node = checkCudaErrors(
                    cuda_rt.cudaGraphAddChildGraphNode(
                        graph,
                        suffix_dependencies,
                        len(suffix_dependencies),
                        cls._raw_graph(suffix),
                    )
                )

            if epilogue is not None:
                epilogue_dependencies = [
                    node
                    for node in (join_node, suffix_node)
                    if node is not None
                ]
                checkCudaErrors(
                    cuda_rt.cudaGraphAddChildGraphNode(
                        graph,
                        epilogue_dependencies,
                        len(epilogue_dependencies),
                        cls._raw_graph(epilogue),
                    )
                )
            executable = checkCudaErrors(
                cuda_rt.cudaGraphInstantiateWithFlags(graph, 0)
            )
            retained = tuple(
                child
                for child in (
                    prefix,
                    *(
                        item
                        for stage in stages
                        for item in (
                            stage.route_prefix,
                            stage.true_body,
                            stage.false_body,
                            *stage.side_bodies,
                        )
                    ),
                    join,
                    suffix,
                    epilogue,
                )
                if child is not None
            )
            return cls(
                graph=graph,
                executable=executable,
                condition_handles=tuple(condition_handles),
                setter_graphs=tuple(setter_graphs),
                helper=helper,
                retained_graphs=retained,
                pre_body_count=0,
                side_body_count=sum(
                    len(stage.side_bodies) for stage in stages
                ),
                join_body_count=int(join is not None),
                join_overlap=bool(overlap_join and join is not None),
                epilogue_body_count=int(epilogue is not None),
                branch_count=2,
            )
        except Exception:
            if executable is not None:
                checkCudaErrors(cuda_rt.cudaGraphExecDestroy(executable))
            if graph is not None:
                checkCudaErrors(cuda_rt.cudaGraphDestroy(graph))
            for setter_graph in setter_graphs:
                setter_graph.reset()
            raise


    def replay(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        if self._closed:
            raise RuntimeError("conditional CUDA graph is closed")
        target = stream or torch.cuda.current_stream()
        checkCudaErrors(
            cuda_rt.cudaGraphLaunch(self._executable, target.cuda_stream)
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        checkCudaErrors(cuda_rt.cudaGraphExecDestroy(self._executable))
        checkCudaErrors(cuda_rt.cudaGraphDestroy(self._graph))
        for setter_graph in self._setter_graphs:
            setter_graph.reset()
        self._setter_graphs = ()
        self._condition_handles = ()
        self._retained_graphs = ()

    def __enter__(self) -> "ConditionalCudaGraph":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        if not getattr(self, "_closed", True):
            try:
                self.close()
            except Exception:
                pass
def capture_raw_graph(
    fn: Callable[[], None],
    *,
    stream: torch.cuda.Stream,
    pool: Any,
    warmup: int = 2,
    post_warmup_hook: Optional[Callable[[], None]] = None,
) -> torch.cuda.CUDAGraph:
    """Capture one mutable raw graph for direct gates and VP composition."""

    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    current = torch.cuda.current_stream()
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            fn()
            if post_warmup_hook is not None:
                post_warmup_hook()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.stream(stream):
        graph.capture_begin(pool=pool)
        fn()
        graph.capture_end()
    current.wait_stream(stream)
    return graph
def full_graph_batched_commit_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether the deferred K/V commit runs as one batched launch."""

    values = os.environ if environ is None else environ
    value = str(values.get(FD_BATCHED_COMMIT_ENV, "0")).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{FD_BATCHED_COMMIT_ENV} must be a boolean value")
def fd_execute_project_kv_repair_full_graph(
    layer: Any,
    positions: torch.Tensor,
    forward_batch: Any,
    prepared: "FullGraphPreparedLayerRoute",
    *,
    repair_hidden_states: Optional[torch.Tensor] = None,
    repair_kv_output: Optional[torch.Tensor] = None,
    repair_k_output: Optional[torch.Tensor] = None,
    repair_v_output: Optional[torch.Tensor] = None,
    repair_q_scratch: Optional[torch.Tensor] = None,
) -> None:
    """Compute PROJECT-row own-layer K/V in graph-owned side work.

    The M3.34 correctness path preserves the released fused QKV projection
    shape. The compact path projects only own-weight K/V for mapped PROJECT
    rows. Live cache mutation remains deferred to the joined commit graph so
    side compute never races foreground attention writes.
    """

    if not full_graph_defer_project_kv_enabled():
        raise RuntimeError("deferred PROJECT K/V repair is not enabled")
    attention = layer.self_attn
    diagnostic_stage = full_graph_defer_project_kv_diagnostic_stage()
    hidden_states = (
        prepared.hidden_states
        if repair_hidden_states is None
        else repair_hidden_states
    )
    if hidden_states.shape != prepared.hidden_states.shape:
        raise RuntimeError(
            "deferred PROJECT K/V repair input shape changed"
        )
    valid_rows = forward_batch.fd_full_graph_valid_rows
    if valid_rows is None or valid_rows.shape != (hidden_states.shape[0],):
        raise RuntimeError(
            "deferred PROJECT K/V repair requires aligned valid rows"
        )
    if diagnostic_stage != "readiness_only":
        if full_graph_compact_routed_qkv_enabled():
            if diagnostic_stage != "full":
                raise RuntimeError(
                    "compact routed K/V repair requires full semantic mode"
                )
            if prepared.project_row_map is None or prepared.route_counts is None:
                raise RuntimeError(
                    "compact routed K/V repair requires PROJECT row metadata"
                )
            expected_shape = (
                int(hidden_states.shape[0]),
                2 * int(attention.kv_size),
            )
            if repair_kv_output is None or repair_kv_output.shape != expected_shape:
                raise RuntimeError(
                    "compact routed K/V repair requires a stable combined output"
                )
            radix_attention = attention.attn
            expected_q_shape = (
                int(hidden_states.shape[0]),
                int(radix_attention.qk_head_dim),
            )
            if repair_q_scratch is None or repair_q_scratch.shape != expected_q_shape:
                raise RuntimeError(
                    "compact routed K/V repair requires stable rotary Q scratch"
                )
            qkv_weight = attention.qkv_proj.weight
            expected_width = attention.q_size + 2 * attention.kv_size
            if (
                getattr(attention.qkv_proj, "bias", None) is not None
                or qkv_weight.ndim != 2
                or int(qkv_weight.shape[0]) != expected_width
            ):
                raise RuntimeError(
                    "compact routed K/V repair requires compatible bias-free weights"
                )
            (
                contiguous_routed_qkv,
                routed_qkv_min_rows,
                routed_qkv_multiple,
                routed_qkv_capacities,
            ) = full_graph_contiguous_routed_qkv_config()
            layer_id = int(getattr(attention.attn, "layer_id", prepared.layer_id))
            capacity_fractions = routed_qkv_capacities.get(layer_id)
            if contiguous_routed_qkv and hidden_states.shape[0] >= routed_qkv_min_rows:
                if capacity_fractions is None:
                    raise RuntimeError(
                        "contiguous routed K/V is missing the active layer"
                    )
                project_capacity = _compact_capacity(
                    int(hidden_states.shape[0]),
                    capacity_fractions[1],
                    routed_qkv_multiple,
                )
                _fixed_capacity_mapped_linear(
                    hidden_states,
                    qkv_weight[attention.q_size : expected_width],
                    prepared.project_row_map,
                    prepared.route_counts[1:2],
                    repair_kv_output,
                    capacity=project_capacity,
                )
            else:
                from sglang.srt.vpipe.cohort import mapped_linear

                mapped_linear(
                    hidden_states,
                    qkv_weight[attention.q_size : expected_width],
                    prepared.project_row_map,
                    prepared.route_counts[1:2],
                    repair_kv_output,
                )
            k, v = repair_kv_output.split(
                [attention.kv_size, attention.kv_size], dim=-1
            )
            q = repair_q_scratch
        else:
            qkv, _ = attention.qkv_proj(hidden_states)
            q, k, v = qkv.split(
                [attention.q_size, attention.kv_size, attention.kv_size], dim=-1
            )
        if diagnostic_stage != "qkv_only":
            q, k = attention.rotary_emb(positions, q, k)
            del q
        if diagnostic_stage == "full":
            radix_attention = attention.attn
            k = k.view(
                -1, radix_attention.tp_k_head_num, radix_attention.qk_head_dim
            )
            v = v.view(
                -1, radix_attention.tp_v_head_num, radix_attention.v_head_dim
            )
            if repair_k_output is None or repair_v_output is None:
                raise RuntimeError(
                    "full deferred PROJECT K/V repair requires stable K/V "
                    "outputs"
                )
            if (
                repair_k_output.shape != k.shape
                or repair_v_output.shape != v.shape
            ):
                raise RuntimeError("deferred PROJECT K/V output shape changed")
            if repair_k_output.data_ptr() != k.data_ptr():
                repair_k_output.copy_(k)
            if repair_v_output.data_ptr() != v.data_ptr():
                repair_v_output.copy_(v)
    if diagnostic_stage == "full":
        return
    device_tape = getattr(
        forward_batch, "fd_full_graph_device_route_tape", None
    )
    if device_tape is None or prepared.inline_kv_index is None:
        raise RuntimeError(
            "deferred PROJECT K/V repair requires device-tape readiness"
        )
    device_tape.record_inline_kv_ready(
        int(layer.layer_id), index=prepared.inline_kv_index
    )
def fd_commit_project_kv_repair_full_graph(
    layer: Any,
    forward_batch: Any,
    prepared: "FullGraphPreparedLayerRoute",
    repair_k: torch.Tensor,
    repair_v: torch.Tensor,
    project_rows: torch.Tensor,
) -> None:
    """Commit computed repair K/V after foreground and side graphs join."""

    if full_graph_defer_project_kv_diagnostic_stage() != "full":
        raise RuntimeError("diagnostic repair stages cannot commit K/V")
    valid_rows = forward_batch.fd_full_graph_valid_rows
    if valid_rows is None or project_rows.shape != valid_rows.shape:
        raise RuntimeError("deferred PROJECT K/V commit requires aligned rows")
    if project_rows.dtype != torch.bool:
        raise RuntimeError("deferred PROJECT K/V commit mask must be boolean")
    cache_locations = forward_batch.out_cache_loc
    if cache_locations is None or cache_locations.shape != project_rows.shape:
        raise RuntimeError(
            "deferred PROJECT K/V commit requires aligned cache locations"
        )
    radix_attention = layer.self_attn.attn
    expected_k_shape = (
        int(project_rows.numel()),
        radix_attention.tp_k_head_num,
        radix_attention.qk_head_dim,
    )
    expected_v_shape = (
        int(project_rows.numel()),
        radix_attention.tp_v_head_num,
        radix_attention.v_head_dim,
    )
    if repair_k.shape != expected_k_shape or repair_v.shape != expected_v_shape:
        raise RuntimeError("deferred PROJECT K/V commit buffer shape changed")
    from sglang.srt.model_executor.forward_context import get_attn_backend

    kv_pool = get_attn_backend().token_to_kv_pool
    if getattr(kv_pool, "use_hnd", False):
        raise RuntimeError(
            "deferred PROJECT K/V repair currently requires NHD cache layout"
        )
    repair_cache_locations = fd_project_kv_repair_cache_locations(
        cache_locations,
        project_rows,
    )
    kv_pool.set_kv_buffer(
        radix_attention,
        repair_cache_locations,
        repair_k,
        repair_v,
        radix_attention.k_scale,
        radix_attention.v_scale,
    )
    device_tape = getattr(
        forward_batch, "fd_full_graph_device_route_tape", None
    )
    if device_tape is None or prepared.inline_kv_index is None:
        raise RuntimeError(
            "deferred PROJECT K/V commit requires device-tape readiness"
        )
    device_tape.record_inline_kv_ready(
        int(layer.layer_id), index=prepared.inline_kv_index
    )
def fd_all_valid_rows_run_predicate(
    prepared: FullGraphPreparedLayerRoute,
    valid_rows: torch.Tensor,
) -> torch.Tensor:
    """Return a device int32 scalar; bucket padding cannot veto RUN."""

    run_rows = prepared.run_mask.squeeze(-1)
    if run_rows.shape != valid_rows.shape or valid_rows.dtype != torch.bool:
        raise ValueError(
            "all-RUN predicate requires aligned boolean route and valid rows"
        )
    return (run_rows | ~valid_rows).all().to(torch.int32)
class ConditionalGraphUnavailable(RuntimeError):
    """Raised when the installed CUDA/PyTorch stack cannot build the graph."""
class ConditionalGraphHelper:
    """Load the AOT kernel that writes a CUDA graph condition on-device."""

    def __init__(self, library_path: str | Path) -> None:
        path = Path(library_path).expanduser().resolve()
        if not path.is_file():
            raise ConditionalGraphUnavailable(
                f"conditional graph helper is missing: {path}"
            )
        self.path = path
        self.library = ctypes.CDLL(str(path))
        self.library.vp_launch_set_conditional.argtypes = (
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_void_p,
        )
        self.library.vp_launch_set_conditional.restype = ctypes.c_int
        self.library.vp_launch_set_conditional_value.argtypes = (
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_void_p,
        )
        self.library.vp_launch_set_conditional_value.restype = ctypes.c_int
        self.library.vp_conditional_graph_cuda_runtime_version.argtypes = ()
        self.library.vp_conditional_graph_cuda_runtime_version.restype = ctypes.c_int
        self.cuda_runtime_version = int(
            self.library.vp_conditional_graph_cuda_runtime_version()
        )
        if self.cuda_runtime_version < 12040:
            raise ConditionalGraphUnavailable(
                "CUDA conditional graph nodes require runtime 12.4 or newer"
            )

    def launch(
        self,
        handle: int,
        predicate: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> None:
        if predicate.device.type != "cuda":
            raise ValueError("conditional graph predicate must be a CUDA tensor")
        if predicate.dtype != torch.int32 or predicate.numel() != 1:
            raise ValueError("conditional graph predicate must be one int32 value")
        status = self.library.vp_launch_set_conditional(
            ctypes.c_uint64(int(handle)),
            ctypes.c_void_p(predicate.data_ptr()),
            ctypes.c_void_p(stream.cuda_stream),
        )
        if status != 0:
            raise RuntimeError(
                f"conditional graph setter launch failed with CUDA status {status}"
            )

@dataclass(slots=True)
class LlamaConditionalGraphCapture:
    """One composed graph plus the raw children it owns."""

    graph: ConditionalCudaGraph
    output: Any
    child_graphs: tuple[torch.cuda.CUDAGraph, ...]
    routed_layers: tuple[int, ...]
    body_execution_counts: Optional[torch.Tensor]
    run_body: str
    stable_state_buffers: int = 2
    repair_input_buffers: int = 0
    repair_kv_output_buffers: int = 0
    repair_commit_graphs: int = 0
    repair_commit_batched: bool = False
    repair_group_size: int = 1
    repair_group_count: int = 0
    repair_graph_pools: tuple[Any, ...] = ()
    repair_capture_streams: tuple[torch.cuda.Stream, ...] = ()
    retained_repair_state: tuple[Any, ...] = ()
def _validate_model(model: Any, forward_batch: Any) -> tuple[Any, tuple[int, ...]]:
    llama = getattr(model, "model", None)
    if llama is None or not hasattr(model, "logits_processor"):
        raise RuntimeError(
            "FlexiDepth conditional graphs currently require LlamaForCausalLM"
        )
    if not llama.pp_group.is_first_rank or not llama.pp_group.is_last_rank:
        raise RuntimeError("FlexiDepth conditional graphs currently require PP=1")
    if llama.layers_to_capture:
        raise RuntimeError(
            "FlexiDepth conditional graphs do not support auxiliary layer capture"
        )
    if getattr(model, "capture_aux_hidden_states", False):
        raise RuntimeError(
            "FlexiDepth conditional graphs do not support auxiliary hidden states"
        )
    routed_layers = tuple(
        int(layer.layer_id)
        for layer in llama.layers
        if getattr(layer, "fd_router", None) is not None
    )
    if not routed_layers:
        raise RuntimeError("FlexiDepth conditional graph has no routed layers")
    if routed_layers != tuple(llama._fd_full_graph_route_layer_order):
        raise RuntimeError("FlexiDepth routed layer order changed during capture")
    if routed_layers[0] <= llama.start_layer:
        raise RuntimeError(
            "FlexiDepth conditional graph requires a non-routed prefix layer"
        )
    if getattr(forward_batch, "lora_ids", None) is not None:
        raise RuntimeError("FlexiDepth conditional graphs do not support LoRA")
    return llama, routed_layers
def _group_stage_indices(
    stage_count: int, group_size: int
) -> tuple[tuple[int, ...], ...]:
    if stage_count <= 0 or group_size <= 0:
        raise ValueError("repair stage and group counts must be positive")
    return tuple(
        tuple(range(start, min(stage_count, start + group_size)))
        for start in range(0, stage_count, group_size)
    )
def _layer_contiguous_views(
    num_layers: int,
    rows: int,
    flat_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> list[torch.Tensor]:
    """Per-layer (rows, flat_dim) views into one layer-contiguous backing.

    The per-layer stride is padded up to ``_LAYER_BACKING_ALIGN_BYTES`` so
    every layer's view starts at an allocation-grade aligned address — the
    unpadded stride is only coincidentally aligned for common model dims
    (HBM D-299+1 review finding). The backing outlives the views through
    their storage references; callers need only the views.
    """
    itemsize = torch.empty((), dtype=dtype).element_size()
    if _LAYER_BACKING_ALIGN_BYTES % itemsize:
        raise ValueError(
            f"backing alignment {_LAYER_BACKING_ALIGN_BYTES}B is not a "
            f"multiple of {dtype} itemsize {itemsize}"
        )
    align_elems = _LAYER_BACKING_ALIGN_BYTES // itemsize
    per_layer = rows * flat_dim
    padded = ((per_layer + align_elems - 1) // align_elems) * align_elems
    backing = torch.empty((num_layers, padded), dtype=dtype, device=device)
    return [
        backing[index, :per_layer].view(rows, flat_dim)
        for index in range(num_layers)
    ]
def capture_llama_flexidepth_conditional_graph(
    *,
    model: Any,
    forward_batch: Any,
    attn_backend: Any,
    helper: ConditionalGraphHelper,
    stream: torch.cuda.Stream,
    pool: Any,
    post_warmup_hook: Optional[Callable[[], None]] = None,
) -> LlamaConditionalGraphCapture:
    """Capture prefix, sequential route/bodies, and suffix into one graph.

    The all-RUN body flavor is read from the process env flag
    (``full_graph_conditional_production_all_run_enabled``).  Under W1 the decode
    leg keeps that flag off, so this always captures the M3.38 masked/filtered
    skip body — the "prod_allrun" low band is served by a separate STOCK
    production decode graph (I6b fork option (b)), never through this conditional
    capture, so no per-regime body-flavor argument is needed here.
    """

    llama, routed_layers = _validate_model(model, forward_batch)
    input_ids = forward_batch.input_ids
    positions = forward_batch.positions
    num_tokens = int(input_ids.numel())
    hidden_size = int(model.config.hidden_size)
    dtype = llama.layers[routed_layers[0]].mlp.gate_up_proj.weight.dtype
    device = input_ids.device
    hidden_buffers = [
        torch.empty(
            (num_tokens, hidden_size),
            dtype=dtype,
            device=device,
        )
        for _ in range(2)
    ]
    residual_buffers = [torch.empty_like(hidden_buffers[0]) for _ in range(2)]
    defer_project_kv = full_graph_defer_project_kv_enabled()
    repair_semantic_kv = (
        defer_project_kv
        and full_graph_defer_project_kv_diagnostic_stage() == "full"
    )
    compact_routed_qkv = full_graph_compact_routed_qkv_enabled()
    repair_group_size = full_graph_repair_group_size() if defer_project_kv else 1
    repair_groups = (
        _group_stage_indices(len(routed_layers), repair_group_size)
        if defer_project_kv
        else ()
    )
    # HBM D-299+1: allocate the per-layer repair families as SINGLE
    # layer-contiguous backings (aligned per-layer stride) and hand out
    # per-layer views. Stabilizes the caching-allocator layout across
    # defer/inline arms (the measured GEMM tile-selection coupling) and
    # provides the layer-contiguous pending layout that batched commit
    # (proposal 2) requires. Views are per-layer-contiguous; downstream
    # identity checks (data_ptr) still hold per view; allocation remains
    # pre-capture.
    repair_input_buffers = (
        _layer_contiguous_views(
            len(routed_layers), num_tokens, hidden_size, dtype, device
        )
        if defer_project_kv
        else []
    )
    repair_radix_attention = (
        llama.layers[routed_layers[0]].self_attn.attn
        if repair_semantic_kv
        else None
    )
    repair_kv_size = (
        int(llama.layers[routed_layers[0]].self_attn.kv_size)
        if repair_semantic_kv
        else 0
    )
    repair_kv_buffers = (
        _layer_contiguous_views(
            len(routed_layers), num_tokens, 2 * repair_kv_size, dtype, device
        )
        if repair_semantic_kv and compact_routed_qkv
        else []
    )
    repair_k_buffers = (
        [
            buffer[:, :repair_kv_size].view(
                num_tokens,
                repair_radix_attention.tp_k_head_num,
                repair_radix_attention.qk_head_dim,
            )
            for buffer in repair_kv_buffers
        ]
        if repair_semantic_kv and compact_routed_qkv
        else [
            view.view(
                num_tokens,
                repair_radix_attention.tp_k_head_num,
                repair_radix_attention.qk_head_dim,
            )
            for view in _layer_contiguous_views(
                len(routed_layers),
                num_tokens,
                repair_radix_attention.tp_k_head_num
                * repair_radix_attention.qk_head_dim,
                dtype,
                device,
            )
        ]
        if repair_semantic_kv
        else []
    )
    repair_v_buffers = (
        [
            buffer[:, repair_kv_size:].view(
                num_tokens,
                repair_radix_attention.tp_v_head_num,
                repair_radix_attention.v_head_dim,
            )
            for buffer in repair_kv_buffers
        ]
        if repair_semantic_kv and compact_routed_qkv
        else [
            view.view(
                num_tokens,
                repair_radix_attention.tp_v_head_num,
                repair_radix_attention.v_head_dim,
            )
            for view in _layer_contiguous_views(
                len(routed_layers),
                num_tokens,
                repair_radix_attention.tp_v_head_num
                * repair_radix_attention.v_head_dim,
                dtype,
                device,
            )
        ]
        if repair_semantic_kv
        else []
    )
    repair_q_scratch = (
        [
            torch.zeros(
                (num_tokens, repair_radix_attention.qk_head_dim),
                dtype=dtype,
                device=device,
            )
            for _ in repair_groups
        ]
        if repair_semantic_kv and compact_routed_qkv
        else []
    )
    # Layer-contiguous [L, N] mask backing: the batched commit kernel indexes
    # masks as one flat table; per-layer views keep the existing interface.
    repair_project_mask_backing = (
        torch.empty(
            (len(routed_layers), num_tokens), dtype=torch.bool, device=device
        )
        if repair_semantic_kv
        else None
    )
    repair_project_masks = (
        list(repair_project_mask_backing.unbind(0))
        if repair_semantic_kv
        else []
    )
    # Group-private streams and pools let repair superstages overlap foreground
    # execution without multiplying graph pools by every routed layer.
    repair_graph_pools = (
        tuple(torch.cuda.graph_pool_handle() for _ in repair_groups)
        if defer_project_kv
        else ()
    )
    repair_capture_streams = (
        tuple(torch.cuda.Stream(device=device) for _ in repair_groups)
        if defer_project_kv
        else ()
    )
    child_graphs: list[torch.cuda.CUDAGraph] = []
    composed: Optional[ConditionalCudaGraph] = None
    body_execution_counts = (
        torch.zeros(
            (len(routed_layers), 2),
            dtype=torch.int64,
            device=device,
        )
        if full_graph_conditional_branch_counters_enabled()
        else None
    )
    production_all_run = full_graph_conditional_production_all_run_enabled()
    run_body_name = (
        "production_flashinfer_attention_dense_weighted_mlp"
        if production_all_run
        else "masked_attention_filtered_run_only"
    )

    def prefix_fn() -> None:
        attn_backend.init_forward_metadata_in_graph(forward_batch)
        forward_batch.dp_local_start_pos = None
        forward_batch.dp_local_num_tokens = None
        set_dp_buffer_len(
            forward_batch.global_dp_buffer_len,
            num_tokens,
            forward_batch.dp_padding_mode.is_max_len(),
            forward_batch.global_num_tokens_cpu,
        )
        set_is_extend_in_batch(False)
        hidden_states = llama.embed_tokens(input_ids)
        residual = None
        prepare_full_graph_batch(
            forward_batch,
            hidden_states,
            positions=positions,
            route_layer_ids=llama._fd_full_graph_route_layer_ids,
            route_layer_order=llama._fd_full_graph_route_layer_order,
            compact_evidence_specs=llama._fd_full_graph_compact_evidence_specs,
        )
        for layer_id in range(llama.start_layer, routed_layers[0]):
            hidden_states, residual = llama.layers[layer_id](
                positions,
                hidden_states,
                forward_batch,
                residual,
            )
        if residual is None:
            raise RuntimeError("FlexiDepth conditional prefix produced no residual")
        hidden_buffers[0].copy_(hidden_states)
        residual_buffers[0].copy_(residual)

    try:
        prefix = capture_raw_graph(
            prefix_fn,
            stream=stream,
            pool=pool,
            post_warmup_hook=post_warmup_hook,
        )
        child_graphs.append(prefix)
        stages: list[ConditionalGraphStage] = []
        prepared_routes: list[Any] = []
        current_buffer = 0
        next_layer = routed_layers[0]

        for stage_index, layer_id in enumerate(routed_layers):
            if layer_id < next_layer:
                raise RuntimeError("FlexiDepth routed layer order is not increasing")
            next_buffer = 1 - current_buffer
            predicate = torch.empty((), dtype=torch.int32, device=device)
            prepared_holder: list[Any] = []

            def route_prefix_fn(
                *,
                stage_index: int = stage_index,
                layer_id: int = layer_id,
                start_layer: int = next_layer,
                source_buffer: int = current_buffer,
            ) -> None:
                hidden_states = hidden_buffers[source_buffer]
                residual = residual_buffers[source_buffer]
                for gap_layer_id in range(start_layer, layer_id):
                    hidden_states, residual = llama.layers[gap_layer_id](
                        positions,
                        hidden_states,
                        forward_batch,
                        residual,
                    )
                tape = forward_batch.fd_full_graph_device_route_tape
                if tape is None:
                    raise RuntimeError(
                        "conditional capture requires the device route tape"
                    )
                tape.recorded_layers = stage_index
                tape.kv_recorded_layers = stage_index
                layer = llama.layers[layer_id]
                prepared = fd_prepare_layer_route_full_graph(
                    layer,
                    hidden_states,
                    forward_batch,
                    residual,
                    layer.fd_router,
                )
                if defer_project_kv:
                    repair_input_buffers[stage_index].copy_(
                        prepared.hidden_states
                    )
                if repair_semantic_kv:
                    repair_project_masks[stage_index].copy_(
                        (~prepared.run_mask.squeeze(-1))
                        & forward_batch.fd_full_graph_valid_rows
                    )
                predicate.copy_(
                    fd_all_valid_rows_run_predicate(
                        prepared,
                        forward_batch.fd_full_graph_valid_rows,
                    )
                )
                prepared_holder[:] = [prepared]

            route_prefix = capture_raw_graph(
                route_prefix_fn,
                stream=stream,
                pool=pool,
                post_warmup_hook=post_warmup_hook,
            )
            child_graphs.append(route_prefix)
            with torch.cuda.stream(stream):
                route_prefix.replay()
            stream.synchronize()
            prepared = prepared_holder[0]
            prepared_routes.append(prepared)
            layer = llama.layers[layer_id]

            def run_body_fn(
                *,
                layer: Any = layer,
                prepared: Any = prepared,
                target_buffer: int = next_buffer,
            ) -> None:
                if body_execution_counts is not None:
                    body_execution_counts[stage_index, 0].add_(1)
                hidden_states, residual = (
                    fd_execute_prepared_layer_route_full_graph(
                        layer,
                        positions,
                        forward_batch,
                        layer.fd_proj,
                        prepared,
                        force_dense_all_run=production_all_run,
                        force_filtered_all_run=not production_all_run,
                        force_production_attention=production_all_run,
                    )
                )
                hidden_buffers[target_buffer].copy_(hidden_states)
                residual_buffers[target_buffer].copy_(residual)

            run_body = capture_raw_graph(
                run_body_fn,
                stream=stream,
                pool=pool,
                post_warmup_hook=post_warmup_hook,
            )
            child_graphs.append(run_body)
            with torch.cuda.stream(stream):
                route_prefix.replay()
            stream.synchronize()

            def mixed_body_fn(
                *,
                layer: Any = layer,
                prepared: Any = prepared,
                target_buffer: int = next_buffer,
            ) -> None:
                if body_execution_counts is not None:
                    body_execution_counts[stage_index, 1].add_(1)
                hidden_states, residual = (
                    fd_execute_prepared_layer_route_full_graph(
                        layer,
                        positions,
                        forward_batch,
                        layer.fd_proj,
                        prepared,
                        force_dense_all_run=False,
                        force_production_attention=False,
                    )
                )
                hidden_buffers[target_buffer].copy_(hidden_states)
                residual_buffers[target_buffer].copy_(residual)

            mixed_body = capture_raw_graph(
                mixed_body_fn,
                stream=stream,
                pool=pool,
                post_warmup_hook=post_warmup_hook,
            )
            child_graphs.append(mixed_body)
            stages.append(
                ConditionalGraphStage(
                    predicate=predicate,
                    route_prefix=route_prefix,
                    true_body=run_body,
                    false_body=mixed_body,
                )
            )
            current_buffer = next_buffer
            next_layer = layer_id + 1

        for group_index, stage_indices in enumerate(repair_groups):

            def repair_group_body_fn(
                *,
                stage_indices: tuple[int, ...] = stage_indices,
                group_index: int = group_index,
            ) -> None:
                for repair_stage_index in stage_indices:
                    layer_id = routed_layers[repair_stage_index]
                    fd_execute_project_kv_repair_full_graph(
                        llama.layers[layer_id],
                        positions,
                        forward_batch,
                        prepared_routes[repair_stage_index],
                        repair_hidden_states=repair_input_buffers[
                            repair_stage_index
                        ],
                        repair_kv_output=(
                            repair_kv_buffers[repair_stage_index]
                            if compact_routed_qkv
                            else None
                        ),
                        repair_k_output=(
                            repair_k_buffers[repair_stage_index]
                            if repair_semantic_kv
                            else None
                        ),
                        repair_v_output=(
                            repair_v_buffers[repair_stage_index]
                            if repair_semantic_kv
                            else None
                        ),
                        repair_q_scratch=(
                            repair_q_scratch[group_index]
                            if repair_q_scratch
                            else None
                        ),
                    )

            repair_group_body = capture_raw_graph(
                repair_group_body_fn,
                stream=repair_capture_streams[group_index],
                pool=repair_graph_pools[group_index],
                post_warmup_hook=post_warmup_hook,
            )
            child_graphs.append(repair_group_body)
            terminal_stage = stage_indices[-1]
            stages[terminal_stage] = replace(
                stages[terminal_stage], side_bodies=(repair_group_body,)
            )

        repair_commit = None
        commit_overlap = full_graph_commit_overlap_enabled()
        batched_commit = (
            full_graph_batched_commit_enabled() if repair_semantic_kv else False
        )
        commit_graph_pool = None
        if repair_semantic_kv and batched_commit:
            # HBM proposal 2 (D-299+1): one cross-layer kernel launch per
            # step over the layer-contiguous repair/mask backings replaces
            # the per-layer masked pool writes. The plan builder fail-closes
            # on any layout it cannot prove equivalent; the per-layer
            # readiness recording (cheap copy_) is unchanged.
            from sglang.srt.model_executor.forward_context import (
                get_attn_backend,
            )
            from sglang.srt.vpipe.cohort import (
                build_batched_commit_plan,
                run_batched_commit,
            )

            batched_plan = build_batched_commit_plan(
                routed_layers=routed_layers,
                llama=llama,
                kv_pool=get_attn_backend().token_to_kv_pool,
                repair_k_buffers=repair_k_buffers,
                repair_v_buffers=repair_v_buffers,
                mask_backing=repair_project_mask_backing,
                cache_locations=forward_batch.out_cache_loc,
                num_tokens=num_tokens,
            )

            def repair_commit_fn() -> None:
                run_batched_commit(batched_plan)
                device_tape = getattr(
                    forward_batch, "fd_full_graph_device_route_tape", None
                )
                if device_tape is None:
                    raise RuntimeError(
                        "batched K/V commit requires device-tape readiness"
                    )
                for stage_index, layer_id in enumerate(routed_layers):
                    prepared = prepared_routes[stage_index]
                    if prepared.inline_kv_index is None:
                        raise RuntimeError(
                            "batched K/V commit requires device-tape readiness"
                        )
                    device_tape.record_inline_kv_ready(
                        int(layer_id), index=prepared.inline_kv_index
                    )

        elif repair_semantic_kv:

            def repair_commit_fn() -> None:
                for stage_index, layer_id in enumerate(routed_layers):
                    fd_commit_project_kv_repair_full_graph(
                        llama.layers[layer_id],
                        forward_batch,
                        prepared_routes[stage_index],
                        repair_k_buffers[stage_index],
                        repair_v_buffers[stage_index],
                        repair_project_masks[stage_index],
                    )

        if repair_semantic_kv:

            # Overlap replays the commit concurrently with the suffix, and
            # graphs sharing one memory pool must never run concurrently
            # (allocator reuse would alias the commit's temporaries into the
            # suffix) — same rule the repair side bodies already follow.
            if commit_overlap:
                commit_graph_pool = torch.cuda.graph_pool_handle()
            repair_commit = capture_raw_graph(
                repair_commit_fn,
                stream=stream,
                pool=commit_graph_pool if commit_overlap else pool,
                post_warmup_hook=post_warmup_hook,
            )
            child_graphs.append(repair_commit)

        if commit_overlap and repair_commit is None:
            raise RuntimeError(
                "FD commit overlap requires the deferred semantic K/V commit"
            )

        output_holder: list[Any] = []

        def suffix_fn() -> None:
            hidden_states = hidden_buffers[current_buffer]
            residual = residual_buffers[current_buffer]
            for layer_id in range(next_layer, llama.end_layer):
                hidden_states, residual = llama.layers[layer_id](
                    positions,
                    hidden_states,
                    forward_batch,
                    residual,
                )
            finalize_full_graph_batch(
                forward_batch,
                llama._fd_full_graph_route_counters,
                llama._fd_full_graph_layer_route_counters,
                llama._fd_full_graph_route_digest_counters,
                llama._fd_full_graph_inline_kv_readiness_counters,
                llama._fd_full_graph_phase_route_counters,
                llama._fd_full_graph_low_row_counters,
            )
            hidden_states, _ = llama.norm(hidden_states, residual)
            output_holder[:] = [
                model.logits_processor(
                    input_ids,
                    hidden_states,
                    model.lm_head,
                    forward_batch,
                )
            ]

        def suffix_compute_fn() -> None:
            # Overlap split: the K/V commit runs beside this body, so route
            # evidence (which asserts inline-K/V readiness the commit records)
            # moves to the post-join epilogue below.
            hidden_states = hidden_buffers[current_buffer]
            residual = residual_buffers[current_buffer]
            for layer_id in range(next_layer, llama.end_layer):
                hidden_states, residual = llama.layers[layer_id](
                    positions,
                    hidden_states,
                    forward_batch,
                    residual,
                )
            hidden_states, _ = llama.norm(hidden_states, residual)
            output_holder[:] = [
                model.logits_processor(
                    input_ids,
                    hidden_states,
                    model.lm_head,
                    forward_batch,
                )
            ]

        def route_evidence_fn() -> None:
            finalize_full_graph_batch(
                forward_batch,
                llama._fd_full_graph_route_counters,
                llama._fd_full_graph_layer_route_counters,
                llama._fd_full_graph_route_digest_counters,
                llama._fd_full_graph_inline_kv_readiness_counters,
                llama._fd_full_graph_phase_route_counters,
                llama._fd_full_graph_low_row_counters,
            )

        route_evidence = None
        if commit_overlap:
            suffix = capture_raw_graph(
                suffix_compute_fn,
                stream=stream,
                pool=pool,
                post_warmup_hook=post_warmup_hook,
            )
            child_graphs.append(suffix)
            route_evidence = capture_raw_graph(
                route_evidence_fn,
                stream=stream,
                pool=pool,
                post_warmup_hook=post_warmup_hook,
            )
            child_graphs.append(route_evidence)
        else:
            suffix = capture_raw_graph(
                suffix_fn,
                stream=stream,
                pool=pool,
                post_warmup_hook=post_warmup_hook,
            )
            child_graphs.append(suffix)
        composed = ConditionalCudaGraph.compose_stages(
            stages=tuple(stages),
            helper=helper,
            stream=stream,
            pool=pool,
            prefix=prefix,
            join=repair_commit,
            suffix=suffix,
            overlap_join=commit_overlap,
            epilogue=route_evidence,
        )
        if body_execution_counts is not None:
            with torch.cuda.stream(stream):
                body_execution_counts.zero_()
            stream.synchronize()
        return LlamaConditionalGraphCapture(
            graph=composed,
            output=output_holder[0],
            child_graphs=tuple(child_graphs),
            routed_layers=routed_layers,
            body_execution_counts=body_execution_counts,
            run_body=run_body_name,
            repair_input_buffers=len(repair_input_buffers),
            repair_kv_output_buffers=(
                len(repair_kv_buffers)
                if compact_routed_qkv
                else len(repair_k_buffers) + len(repair_v_buffers)
            ),
            repair_commit_graphs=int(repair_commit is not None),
            repair_commit_batched=bool(
                batched_commit and repair_commit is not None
            ),
            repair_group_size=repair_group_size,
            repair_group_count=len(repair_groups),
            repair_graph_pools=(
                (*repair_graph_pools, commit_graph_pool)
                if commit_graph_pool is not None
                else repair_graph_pools
            ),
            repair_capture_streams=repair_capture_streams,
            retained_repair_state=(
                *repair_input_buffers,
                *repair_kv_buffers,
                *repair_k_buffers,
                *repair_v_buffers,
                *repair_q_scratch,
                *repair_project_masks,
                *prepared_routes,
                # The batched plan's pointer/stride tables are recorded
                # kernel arguments — retain them for the graph's lifetime.
                *((batched_plan,) if batched_commit else ()),
            ),
        )
    except Exception:
        if composed is not None:
            composed.close()
        for child in child_graphs:
            child.reset()
        raise
