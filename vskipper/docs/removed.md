# File relocation and cleanup

The [per-file ledger](relocation-ledger.tsv) covers every file added by the
pre-reorganization reference relative to upstream. It records each old path,
destination, retention status, reason and text references. Runtime extraction
also moves two mapped-attention kernels and runner checks out of existing
SGLang files; see the [integration map](integration.md).

Twelve historical harness files are retired:

- `test/vp/csd3_debug_cell.sh`: historical host paths, an absent cell-count
  contract and runner flags rejected by the current contract.
- `test/vp/run_production_calibration.sh`: requires a missing runtime
  expectation and default calibration contract.
- `test/vp/gates/gates_box.sh`: rewrites its frozen comparison source and
  installs packages into the shared environment. Retained individual tests
  provide the relevant checks without that wrapper.
- The nine files under `test/vp/refactor_ab/`: the earlier `srt.vp` to
  `srt.vpipe` migration rig, tied to historical trees and deployments.

Recover any retired file from reference commit
`461f0af45e457391515cf2e7bb50f92c0fb77b5e` using its old path.

The earlier removal list incorrectly described 37 tracked files as deleted.
This ledger replaces that list. Numerical kernel references, workload builders,
scorers, stream parsing, training/export tools and focused tests are retained.
An unreferenced standalone entrypoint is marked `unresolved-retained` rather
than classified as dead solely because it has no textual caller. Historical
launchers retained on that basis are not entrypoints in the reproduction guide.
