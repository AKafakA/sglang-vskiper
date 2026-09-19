# vSkipper

vSkipper is a serving runtime for conditional-depth language models. It combines
route-aware execution, K/V-complete projection and CUDA-graph dispatch to turn
layer-skipping decisions into executable serving work.

This repository is a **mirrored fork of SGLang**, based on upstream revision
`602c8615a1afbb2ad13b80334643c64970884bac`. SGLang supplies the serving framework;
the project-owned implementation, tools and documentation live in
[`vskipper/`](vskipper/). The original
[SGLang README](README.sglang.md) and [Apache-2.0 license](LICENSE) are preserved.

## Reproduce the results

The reviewer data pack includes frozen evidence, the required analysis source,
reference outputs and a results-only entrypoint:

```bash
bash /path/to/unpacked-pack/reproduce_results.sh /path/to/new-output
```

This route uses Python, NumPy and Matplotlib on CPU. It produces numerical
tables, macros and data-driven plots, then checks 120 reference outputs. It
requires neither the manuscript nor a TeX installation.

Start with the [reproduction guide](vskipper/docs/README.md):

- [Environment and assets](vskipper/docs/01-environment.md)
- [GPU measurement workflow](vskipper/docs/02-run-experiments.md)
- [Analysis from frozen evidence](vskipper/docs/03-run-analysis.md)
- [Validation and result-integrity gates](vskipper/docs/gates.md)

## Source map

| Location | Purpose |
|---|---|
| `vskipper/src/vskipper/runtime/` | Routing, execution, K/V handling, design and accounting |
| `vskipper/src/vskipper/kernels/` | Count-adaptive compute, mapped attention and device resources |
| `vskipper/src/vskipper/integration/sglang/` | Model hooks, graph backend and runner integration |
| `vskipper/src/vskipper/experiments/` | Workload preparation, campaign drivers and integrity gates |
| `vskipper/src/vskipper/analysis/` | Frozen-evidence reproduction |
| `vskipper/tests/` | Runtime and reproduction regression tests |
| `vskipper/training/` | Qwen3 checkpoint training and export tooling |
| `vskipper/configs/` | Deployment declarations and host-path configurations |
| `vskipper/docs/` | Project documentation and relocation ledger |
| `python/sglang/` | Mirrored SGLang framework with explicit integration hooks |

`vskipper-ref` is the organized reference release. `vskipper-dev` preserves the
original development line. The [integration map](vskipper/docs/integration.md)
describes the framework extensions and the [cleanup ledger](vskipper/docs/removed.md)
records file ownership and retirement decisions.
