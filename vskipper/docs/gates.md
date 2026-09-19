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
| Numerical reproduction | Pack `verify_results.py` | All 120 frozen reference outputs |

Run tools through `bash vskipper/scripts/run.sh experiments/TOOL.py ...` to bind
their source paths to this checkout. Read each tool's `--help` for its input
schema. Keep its complete report with the cell artifacts.

Correctness probes establish startup, completion, numerical agreement and
activation. Performance comparisons use the complete declared workload/load
contract and all relevant gates. A passing probe does not replace that campaign.

Historical host-specific migration chains are retired; the
[cleanup ledger](removed.md) records their replacements and recovery revision.

## Anonymous-release check

Run the gate from the anonymous source checkout with an external private TSV
of identifying terms. Each non-comment line is a label, a tab and an extended
regular expression. Include owner handles, account names, host identifiers and
institutional domains in that private file; never add the file to a release.

```bash
bash vskipper/scripts/check_anonymity.sh --terms-file /private/terms.tsv --list
```

Exit 0 means all tracked text and filenames passed; 1 means a match; 2 means
invalid inputs or an incomplete scan. The checker scans its own content.
Inspect exported archive members and binary metadata separately. Maintained
tests exercise clean, identifying, missing-rule and invalid-rule cases.
