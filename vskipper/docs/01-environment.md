# Stage 1 — environment and assets

## CPU numerical reproduction

Use Linux, Bash, GNU `sed` and CPython 3.12.11 with the pinned analysis
dependencies. This environment reproduces all 120 reference outputs with the
exact verifier. Python 3.11 produces roundoff differences in five JSON records;
use the validated Python version for exact reproduction. On macOS, put GNU
`sed` on `PATH`. The reviewer pack contains the analysis source and inputs;
[stage 3](03-run-analysis.md) gives the complete command.

Create a separate analysis environment, preserving any serving environment:

```bash
python3.12 -m venv analysis-env
. analysis-env/bin/activate
python -m pip install -r vskipper/requirements-analysis.txt
```

## GPU serving

GPU reproduction uses the SGLang fork and vSkipper from the same checkout, a
compatible NVIDIA CUDA environment, and the exact checkpoint/router assets
named by the measurement contract. Reuse a pinned environment when available.
Keep its dependency freeze with the run's provenance.

The reviewer pack contains evidence and CPU analysis code. GPU dependencies,
wheels, conditional-graph binaries and model weights are separate assets.
SGLang's dependency declaration is in `python/pyproject.toml`; vSkipper's package
declaration is in `vskipper/pyproject.toml`.

Bind both source trees when using an existing environment:

```bash
export PYTHONPATH="$PWD/vskipper/src:$PWD/python"
export PYTHONDONTWRITEBYTECODE=1
python -c 'import sglang, vskipper; print(sglang.__file__); print(vskipper.__file__)'
```

For a new environment, install the fork and vSkipper as editable packages after
resolving the serving dependencies. Keep environment creation separate from an
established experimental environment.

```bash
python -m pip install -e python/ -e vskipper/
```

Build the conditional-graph helper on the GPU host using its compatible CUDA
compiler, or use a hash-verified compatible existing binary:

```bash
NVCC=/path/to/cuda/bin/nvcc bash \
  vskipper/src/vskipper/experiments/build_cuda_conditional_graph_helper.sh \
  "$PWD" /path/to/new-helper-directory
```

The script accepts a source root and output directory. It invokes the supplied
compiler; it does not detect the device and select an architecture for you.

Create a host-path JSON from
[`host.example.json`](../configs/host.example.json). Set absolute paths to the
router/projector weights, helper and MoE configuration directory. Record the
checkpoint revision and hashes. The arm's design and routing policy remain in
`vskipper/src/vskipper/runtime/design.py`; host files supply filesystem locations.

Device tile resources are packaged under
`vskipper/src/vskipper/kernels/binary_cohort_configs/`, and the roofline resource
is under `vskipper/src/vskipper/runtime/`. Preserve these files when staging a
source archive or building a package.

Run package/import and numerical checks on the GPU compute host before serving:

```bash
python vskipper/tests/test_package_imports.py
python -m pytest vskipper/tests -q
```

Next: [GPU measurements](02-run-experiments.md).
