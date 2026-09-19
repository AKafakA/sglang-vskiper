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
python -m pip install -r requirements-analysis.txt
```

## GPU serving

GPU reproduction uses the SGLang fork and vSkipper from the same checkout, a
compatible NVIDIA CUDA environment, and the exact checkpoint/router assets
named by the measurement contract. Reuse a pinned environment when available.
Keep its dependency freeze with the run's provenance.

The reviewer pack contains evidence and CPU analysis code. GPU dependencies,
wheels, conditional-graph binaries and model weights are separate assets.
The development branch keeps vSkipper inside the SGLang source tree;
its package dependencies are declared in `python/pyproject.toml`.

Bind the dev checkout when using an existing environment:

```bash
export PYTHONPATH="$PWD/python"
export PYTHONDONTWRITEBYTECODE=1
python -c 'import sglang, sglang.srt.vpipe; print(sglang.__file__); print(sglang.srt.vpipe.__file__)'
```

For a new environment, install the fork as an editable package after
resolving the serving dependencies. Keep environment creation separate from an
established experimental environment.

```bash
python -m pip install -e python/
```

Build the conditional-graph helper on the GPU host using its compatible CUDA
compiler, or use a hash-verified compatible existing binary:

```bash
NVCC=/path/to/cuda/bin/nvcc bash \
  test/vp/build_cuda_conditional_graph_helper.sh \
  "$PWD" /path/to/new-helper-directory
```

The script accepts a source root and output directory. It invokes the supplied
compiler; it does not detect the device and select an architecture for you.

Create a host-path JSON from
[`a100.json.example`](../../deploy/hosts/a100.json.example). Set absolute paths to the
router/projector weights, helper and MoE configuration directory. Record the
checkpoint revision and hashes. The arm's design and routing policy remain in
`python/sglang/srt/vpipe/design.py`; host files supply filesystem locations.

Device tile resources are packaged under
`python/sglang/srt/vpipe/binary_cohort_configs/`, and the roofline resource
is under `python/sglang/srt/vpipe/`. Preserve these files when staging a
source archive or building a package.

Run package/import and numerical checks on the GPU compute host before serving:

```bash
python test/vp/test_package_imports.py
python -m pytest test/vp -q
```

Next: [GPU measurements](02-run-experiments.md).
