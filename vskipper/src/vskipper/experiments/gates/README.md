# Result-integrity gates

The maintained [gate map](../../../../docs/gates.md) and
[measurement workflow](../../../../docs/02-run-experiments.md) describe the
current entrypoints and their inputs.

`box_vpcov_arm.sh` and `box_conc16_stress.sh` are retained historical host probes.
Their fixed request/concurrency settings and output handling must be inspected
before reuse. The reference-reorganization validation uses
`vskipper/scripts/validate_reference.py` with its own explicit contract.

The earlier `gates_box.sh` and `refactor_ab/` migration chains are retired.
Individual validators and their regression tests remain available.
