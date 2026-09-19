# Stage 3 — reproduce numerical outputs

The reviewer pack contains frozen measurement summaries, preserved per-example
quality records, the analysis code used by this recipe, and 120 reference
outputs. Reproduction emits numerical tables/macros and data-driven plots.
The architecture illustration is a separate explanatory figure.

## Run the bundled source

Use the CPU environment described in [stage 1](01-environment.md). Extract the
pack into a path without whitespace, choose a new output directory, and run:

```bash
bash /path/to/unpacked-pack/reproduce_results.sh /path/to/new-output
```

The entrypoint verifies the package manifests, uses its bundled source and
writes `generated/`, `figures/` and isolated scratch files. It checks all 120
reference results: TeX files match byte-for-byte; JSON files match structurally
and numerically after normalizing the unpacked pack's absolute path.

The pack contains reproduction materials only. The command requires no paper
source, paper PDF, review files or TeX installation. Saved measurement summaries
are inputs to the recipe; their original full GPU traces are supplied separately
for reconstructing those summaries from raw traffic.

## Run the matching repository source

Use a verified `vskipper-dev` commit for development. The reviewer release uses
the reference commit recorded in the pack's source-provenance file and its
branch-specific paths; record the dev commit when running this route:

```bash
PACK=/path/to/unpacked-pack bash scripts/reproduce_analysis.sh \
  /path/to/new-repository-output
```

This runs `test/vp/` against the same evidence and checks
the same references. `scripts/reproduce_analysis.sh` is results-only;
it never builds or edits a manuscript.

Generated reference headers retain their original generator labels, preserving
byte-identical numerical artifacts. The current executable paths are those above.

Both routes preserve the input evidence and reject an existing output path.
Keep the pack's checksum, source commit, command, environment freeze and output
verification report with a reproduction run.

For new GPU measurements, follow [stage 2](02-run-experiments.md). Keep new
results in their own artifact root and retain the frozen reference pack.
