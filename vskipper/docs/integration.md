# SGLang integration map

vSkipper owns its runtime in a separate Python package. Each runtime module has
one canonical `vskipper.*` import path. The model hooks, conditional graph
backend, graph capture helpers and runner dispatch checks are grouped under
`vskipper.integration.sglang`.

Two mapped decode-attention kernels are in `vskipper.kernels.mapped_attention`;
their launch sites remain in SGLang's attention implementation. Count-adaptive
MLP kernels and device resources are in `vskipper.kernels`.

The mirrored SGLang tree retains the framework-side extensions that own model,
batch, scheduler, attention and graph lifecycle state:

- Llama/Qwen model methods call the shared model seam.
- Forward-batch fields carry routing and request identity with the batch.
- Attention signatures carry the mask and active-row maps into dispatch.
- Decode/prefill graph runners own capture buffers and invoke the vSkipper
  graph backend and policy helpers.
- ModelRunner invokes activation checks and dispatch accounting at their
  existing execution points; its `finally` block resets coverage stamps.
- Scheduler and worker hooks expose runtime accounting and attestations.

These boundaries preserve the serving framework's lifecycle and the original
tensor arithmetic, routing decisions, synchronization and execution order.
Validation compares the reorganized source to frozen reference `461f0af45e`.

Deployment binds `vskipper/src` and `python` from the same checkout. The campaign
preflight checks both paths for fork arms and only the upstream `python` path
for external baseline arms. Installing a second source tree is not a substitute
for that explicit binding.
