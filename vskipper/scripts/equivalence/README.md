# Reference equivalence checks

`ab_compare.py` compares the label and post-traffic attestation format emitted
by the historical `box_vpcov_arm.sh` probe. It remains available for inspecting
those artifacts.

The current module-reorganization validation is
[`../validate_reference.py`](../validate_reference.py). It consumes a sealed
JSON contract, uses one allocated GPU, preserves full per-token responses,
compares serial greedy generation and served configurations, checks routed
activation, and runs the retained tests and frozen-output reproduction.

Validation applies to the exact source revisions in the run contract. Results
from earlier refactor revisions are historical records, not a pass for a new
reference revision. See the [gate map](../../docs/gates.md).
