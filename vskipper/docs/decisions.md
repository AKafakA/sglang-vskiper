# Design and measurement decisions

## Measurement

The external baseline is upstream SGLang from its own pinned checkout. The
fork's `stock` arm serves as an inertness control. Load calibration uses the
declared upstream configuration and its matching graph ladder.

Every paired comparison uses one frozen upstream-derived length bank. The
cross-arm gate verifies request IDs and actual input/output token counts before
the arms enter a shared table. Open-loop offered rate controls performance
measurements; all submitted requests and their tails remain in accounting.

The design and named arms live in `runtime/design.py`. The durable active-arm
file selects the arm; a host-config pointer supplies filesystem locations.
Resolved runtime attestation verifies configuration, and post-traffic counters
verify execution of the required mechanism.

Quality analysis uses the task's `lm-eval` filters and metrics on saved
generations. The paired per-document difference-of-differences analysis checks
the served implementation against the checkpoint's own base-model gap with the
declared one-percentage-point margin. Preserve the recorded interval verdict.

## Runtime

Count-bounded cohort execution gathers routed rows into dense compute cohorts.
The engagement rule uses K/V traffic and the routed body's fixed cost to decide
when to engage it. RUN performs the transformer layer; PROJECT_ONLY produces
the layer's own K/V entries while skipping its attention and MLP computation.

The skipper interface represents whole-layer run-or-project decisions and
checks that capability at startup. Routing choices, tensor arithmetic,
synchronization and dispatch order are preserved across the module reorganization.

Cells, suites and banks use protocol and output-policy names. The reproduction
pack retains numerical evidence and reference outputs independently of the
source directory layout.
