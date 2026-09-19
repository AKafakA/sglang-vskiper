# Stage 2 — GPU measurements

A paired cell serves the same frozen request IDs and input/output token counts
through upstream SGLang and the selected vSkipper arm, at the same offered
rate and repetition. Each measurement retains responses, actual token counts,
finish reasons, launch configuration and runtime attestations.

For numerical reproduction from the supplied evidence, use
[stage 3](03-run-analysis.md). This stage describes collecting new GPU cells.
Exact campaign recreation also needs its frozen workload suites, length banks,
checkpoint assets and environment provenance, supplied separately from the
CPU reviewer pack.

The [Qwen3 checkpoint guide](../../training/qwen3-flexidepth/README.md#reviewer-checkpoint-reproduction)
maps all four reported checkpoints to their assets. The served 4B and 8B
checkpoints are full exports; the two additional 8B checkpoints require
reconstruction from the pinned base and supplied router delta. The same guide
separates inference reproduction from the historical training launchers.

## 1. Stage the source and assets

Follow [stage 1](01-environment.md). Stage the treatment checkout and a separate
upstream checkout at `602c8615a1afbb2ad13b80334643c64970884bac`. The `stock` arm
is the fork with skipping disabled; the external baseline uses the upstream
tree. Keep each tree's source archive and digest.

Create a host JSON with absolute asset paths. Build expected-runtime records
from the selected treatment source:

```bash
python test/vp/make_expected_runtime.py \
  --out-dir /path/to/expectations --arm upstream_g1024 --arm vskipper

python test/vp/gates/campaign_preflight.py \
  --tree "$PWD" --upstream /path/to/upstream --workdir /path/to/staging \
  --host-config /path/to/host.json \
  --serving-pythonpath "$PWD/python"
```

The preflight checks staged assets, the named host configuration, source-path
binding and the upstream runtime's content identity. Add `--require-suites`
when the staging root has `serving-fullcells/` and `banks/` populated. Preserve
the workload-specific hashes and contract alongside those directories.

## 2. Freeze the work

Use the upstream-derived natural-generation harvest to construct one length
bank per workload and rate. Apply that same bank to every compared arm:

```bash
python test/vp/bank_from_harvest.py \
  /path/to/upstream-harvest --out /path/to/new-banks

python test/vp/pin_output_lengths_from_bank.py \
  --requests /path/to/requests.jsonl --metadata /path/to/metadata.jsonl \
  --bank /path/to/new-banks/upstream-harvest/bank.json --context-length 8192 \
  --output-requests /path/to/pinned.requests.jsonl \
  --output-metadata /path/to/pinned.metadata.jsonl \
  --output-summary /path/to/pinned.summary.json
```

Use the context length and output policy declared by the target campaign.
Requested maxima and actual generated lengths remain distinct fields. The
cross-arm work gate must pass before combining any arms into a comparison.

For a fresh load calibration, `qstar_int.py` consumes the completed upstream
ladder directory as a positional argument:

```bash
python test/vp/qstar_int.py /path/to/ladder --emit
```

Use the same decode-graph ladder in calibration and paired measurement. The
tool fails when the ladder has not bracketed the knee. Frozen-paper reproduction
uses the recorded rates and banks instead of recalibrating them.

## 3. Declare and run a campaign

Create `campaign.json` with these fields, using absolute paths:

```json
{
  "tree": "/path/to/vskipper-dev",
  "upstream_tree": "/path/to/upstream",
  "python": "/path/to/environment/bin/python",
  "model_path": "/path/to/pinned-model",
  "suites_dir": "/path/to/pinned-suites",
  "staging_root": "/path/to/staging",
  "expect_dir": "/path/to/expectations",
  "host_config": "/path/to/host.json",
  "source_revision": "FULL_DEV_COMMIT",
  "arms": {"baseline": "upstream_g1024", "treatment": "vskipper"},
  "upstream_arms": {"upstream_g1024": ["--cuda-graph-max-bs", "1024"]},
  "upstream_arm_exemptions": {
    "upstream_g1024": {
      "cuda_graph_max_bs": {"value": 1024, "decision": "Matched decode-graph ladder"},
      "cuda_graph_config": {
        "accept_if": {"decode.max_bs": 1024, "decode.bs[max]": 1024},
        "decision": "Matched decode-graph ladder"
      }
    }
  },
  "datasets": {"gsm8k": {"r9p75": 9.75, "r12p35": 12.35, "r16p25": 16.25}}
}
```

`datasets` maps each workload to rate labels and numeric offered rates. The
`upstream_arms` declaration binds the baseline to the upstream checkout and
supplies its matched graph-cap argument; `upstream_arm_exemptions` declares
the corresponding resolved fields to the default-conformance gate.
The example shows the Llama GSM8K grid; supply the complete target campaign's
workloads, rates and repetitions. For another model, include its recorded
`model_revision` and `served_model_name`. Set `launch_profile` when the contract
selects a profile other than the design's default.

```bash
python test/vp/run_paired_campaign.py \
  --spec campaign.json --out-dir /path/to/new-campaign --reps 6 --dry-run

python test/vp/run_paired_campaign.py \
  --spec campaign.json --out-dir /path/to/new-campaign --reps 6
```

`--start-rep N --reps K` requests K repetitions starting at N. Use a new output
directory; retain prior attempts and their failure records.

## 4. Validate and account for the cells

The paired driver applies its input and per-cell gates, then verifies cross-arm
work identity and writes paired analysis. Inspect `campaign_index.json`, the
per-cell `*.command.json`, response artifacts, gate results and before/after
server snapshots together. A `completed` status alone is not a comparison gate.

Verify a live endpoint's served design with:

```bash
python test/vp/gates/verify_served_design.py \
  --url http://127.0.0.1:32051 --arm vskipper --tree "$PWD"
```

Apply `verify_execution_differences.py` to each repetition's baseline/treatment
snapshots using the matching declaration under `deploy/`.
This is an explicit release check in addition to the paired driver's checks;
preserve its command and result. Consult its `--help` for the snapshot layout.

Read [the gate map](gates.md) before accepting cells, then follow
[stage 3](03-run-analysis.md) to regenerate numerical outputs from saved evidence.
