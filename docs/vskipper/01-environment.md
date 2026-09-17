# Stage 1 — environment

Two options. Pick (a) if you want the tree built from source on your own hardware; pick (b) if the machine should
not spend an hour compiling and the data pack is available to you.

Both end at the same check.

## What you need either way

- Linux, CUDA 12.x, one NVIDIA GPU. The measured configuration is an A100-80GB (compute capability 8.0, `sm80`).
  An H100 (`sm90`) also works; see the note on the helper below.
- Python 3.10 or newer.
- About 40 GB of disk for model weights, plus room for whatever cells you collect.

The three pinned requirement sets live in the data pack under `critical/`:

| file | what it pins |
|---|---|
| `req-venv.txt` | the serving environment (torch, triton, the SGLang dependency set) |
| `req-lmeval.txt` | the `lm-eval` harness used for every quality number |
| `req-lmeval-fd.txt` | `lm-eval` plus the checkpoint's own dependencies, for the native-PyTorch reference arms |

These are exact pins captured from the machine the paper's cells were measured on. Do not relax them: the paper's
numeric format and kernel behaviour follow from this set.

## (a) Build fresh

```bash
git clone <this-repo> vskipper && cd vskipper
python -m venv .venv && . .venv/bin/activate
pip install -r /path/to/pack/critical/req-venv.txt
pip install -e python/                      # the fork itself, editable

# the CUDA helper for the conditional-graph path, built for your architecture
bash test/vp/build_cuda_conditional_graph_helper.sh
```

The helper build reads your GPU's compute capability and compiles for it. On a machine with no GPU visible the
script stops rather than guessing.

## (b) Reuse the prebuilt wheels

The data pack ships a wheel set plus a prebuilt `sm80` helper, so a machine with the same Python minor version and
CUDA major version can skip compilation:

```bash
python -m venv .venv && . .venv/bin/activate
pip install --no-index --find-links /path/to/pack/wheels -r /path/to/pack/critical/req-venv.txt
pip install -e python/
cp /path/to/pack/wheels/helper-sm80 python/sglang/srt/vpipe/csrc/
```

**`sm90` is build-on-first-use.** The H100 helper cannot be rebuilt without an H100, and the machine that produced
the paper's H100 row is released. On an H100, run the build step from option (a) instead of copying the helper.

## Tile artifacts

The count-GEMM path uses tile configurations tuned offline per device, in
`python/sglang/srt/vpipe/binary_cohort_configs/<device-key>.json` (the key is the CUDA device name with the form-factor
and capacity tokens stripped, so every A100 loads one artifact). The loader fails closed: a device without an artifact
is a configuration error, not a slower fallback. The repository ships the artifacts the paper used: `NVIDIA_A100`,
`NVIDIA_H100_HBM3`, `NVIDIA_RTX_A6000` and the `Quadro_RTX_8000` feasibility box.

**Tune before you carry.** Tiles are per-SM resources. The A100 tiles need 110,592 B of shared memory per block;
Ampere/Ada workstation parts (sm_86, sm_89: 101,376 B) cannot launch them, and a carried smaller-card artifact runs
but costs 25--56 % per GEMM (measured on the RTX A6000 with the Quadro RTX 8000 tiles). For a new device, run the tuner
on the card with the fork's own kernel and the model's real shapes:

```bash
PYTHONPATH=python python test/vp/tune_count_gemm_tiles.py \
  --artifact python/sglang/srt/vpipe/binary_cohort_configs/<seed-artifact>.json \
  --out python/sglang/srt/vpipe/binary_cohort_configs/<device-key>.json
```

It times every candidate tile that fits the device's shared memory for each `<op>@<count>` key, holds `BLOCK_K` at the
seed's value and keeps a candidate only if its output is `torch.equal` to the seed's (so numerics do not move with the
tune), records cuBLAS on the same rows beside it, and replaces a key only for a gain of at least 3 %. Commit the artifact
with the tree revision that serves it; the served attestation carries the artifact's digest. The RTX A6000 artifact was
produced this way on 2026-09-17 (19 of 20 keys replaced; 5--25 % from cuBLAS).

## Check

```bash
python -c "import sglang.srt.vpipe; print('vpipe imports')"
python test/vp/test_package_imports.py
python test/vp/test_module_import_safety.py
```

The import-safety test fails the tree if any module under `vpipe/` reads an environment variable for
configuration. That is deliberate: the served design lives in `design.py`, not in the environment, because a
configuration that can be set from outside the tree is a configuration that can silently fail to be set.

Next: [`02-run-experiments.md`](02-run-experiments.md), or skip to
[`03-run-analysis.md`](03-run-analysis.md) if you only want the paper's tables.
