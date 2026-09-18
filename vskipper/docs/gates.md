# Validation and result-integrity gates

The experiment drivers and their gates live in
`vskipper/src/vskipper/experiments/`. Focused regression checks live in
`vskipper/tests/`.

| Check | Entry point | Evidence checked |
|---|---|---|
| Staged deployment | `gates/campaign_preflight.py` | Named assets, verified upstream content and serving source paths |
| Served configuration | `gates/verify_served_design.py` | Endpoint attestation against the selected arm |
| Allowed execution differences | `gates/verify_execution_differences.py` | Both arms' snapshots against the declared comparison contract |
| Default configuration | `gates/verify_default_conformance.py` | Resolved settings against the source's declared exemptions |
| Completed-cell accounting | `gates/cell_gates.py` | Responses, arrival fidelity, routing activation and cell accounting |
| Cross-arm configuration | `gates/verify_cross_arm_config.py` | Allowed differences between paired arms |
| Identical work | `cross_arm_work_gate.py` | Per-request identity and identical input/output counts |
| Numerical reproduction | Pack `verify_results.py` | All 114 frozen reference outputs |

Run tools through `bash vskipper/scripts/run.sh experiments/TOOL.py ...` to bind
their source paths to this checkout. Read each tool's `--help` for its input
schema. Keep its complete report with the cell artifacts.

Correctness probes establish startup, completion, numerical agreement and
activation. Performance comparisons use the complete declared workload/load
contract and all relevant gates. A passing probe does not replace that campaign.

Historical host-specific migration chains are retired; the
[cleanup ledger](removed.md) records their replacements and recovery revision.
