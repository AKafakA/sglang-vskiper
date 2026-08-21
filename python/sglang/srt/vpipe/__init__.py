"""vPipe — graph-compatible dynamic layer skipping for production SGLang.

Serves a conditional-depth (layer-skipping) LLM so the skipper's saved FLOPs
become wall-clock, while preserving its logical actions and K/V state.

Per routed layer each token takes one of two actions:
  RUN           — the full transformer layer.
  PROJECT_ONLY  — skip attention+MLP compute, but still produce K/V for that
                  layer from THAT LAYER'S OWN projection weights.

The invariant is K/V-completeness: every generated token has K/V at every layer
a later attention may read, and K/V is never copied across layers.

Modules:
  kernel      count-adaptive routed-MLP GEMM (replaces grouped fused_moe)
  routing     router forward -> RUN/PROJECT mask -> device-resident index maps
  executor    attention path + policy-selected MLP body
  graphs      conditional CUDA-graph capture for the routed decode body
  kv_commit   deferred PROJECT_ONLY K/V production, ordering-fenced
  skipper     pluggable policy: FlexiDepth / RandomSkip / AdaSkip
"""

