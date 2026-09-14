# Stage 2 — run experiments

From a clean GPU host to measured cells in a directory stage 3 can read.

**You probably do not need this.** The cells the paper reports are already measured and shipped in the data pack;
[`03-run-analysis.md`](03-run-analysis.md) reproduces every table and figure from them with no GPU. This guide is
for re-running the measurement from scratch, or for measuring a new arm.

## What a cell is

A **cell** is one (dataset, offered rate, repetition) served by one arm. A paired cell is the same dataset, rate
and repetition served by **both** arms — the treatment and upstream SGLang — against the same pinned work. The
paired unit is the within-repetition delta, which is why both arms always run inside the same repetition.

The work is pinned in a **bank**: a frozen list of prompts with each request's output length fixed in advance, so
both arms generate exactly the same number of tokens. That is what makes the comparison a runtime measurement
rather than a measurement of how much the two models chose to say.

## 1. Declare the host and the arms

A host file under `deploy/hosts/` names the device and its resolved launch profile. Copy the example and fill in
your own paths:

```bash
cp deploy/hosts/a100.json.example deploy/hosts/my-a100.json
```

A **campaign spec** is a JSON file naming the two trees, the model, the bank directory and the arms. Its fields:

| field | meaning |
|---|---|
| `tree` | the checkout serving the treatment arm |
| `upstream_tree` | the **separate** upstream SGLang checkout serving the baseline |
| `model_path` | the pinned model revision |
| `suites_dir` | where the frozen banks live |
| `host_config` | the host file above |
| `arms` | arm name to `design.py` entry, for both arms |
| `datasets` | dataset to rate list |

`upstream_tree` must be a genuine upstream checkout. This fork with the skipper disabled is **not** the baseline;
it is a different arm with its own name, and the ablation table reports it separately.

## 2. Refuse to start if the inputs are not staged

```bash
python test/vp/gates/campaign_preflight.py --spec my_spec.json
```

This checks the campaign's *inputs*: both trees present and at the revisions the spec names, the model staged, the
banks present and hash-matching. It exists because a re-staged host once silently lacked the upstream baseline
tree and every downstream gate still passed.

## 3. Refuse to measure the wrong system

Once a server is up:

```bash
python test/vp/gates/verify_served_design.py --base-url http://127.0.0.1:32051 --arm <arm-name>
```

This reads the server's own attested design and compares it to the `design.py` entry the arm names. Every other
gate in the project checks the *output* of a measurement; this one checks that the intended system produced it.

## 4. Find the knee

Serving load points are calibrated to **upstream's** capacity, not the treatment's. Walk contiguous integer
offered rates on upstream and take Q\* as the last rate whose achieved output throughput still beats the running
mean of the rates below it:

```bash
python test/vp/qstar_int.py --cells <ladder-root>/cells
```

A ladder that never plateaus yields no Q\*: extend it rather than declaring one. The paper's knees are GSM8K 11,
BBH 25 and CoQA 27 requests per second, and every headline cell is served at 0.75, 0.95 and 1.25 times its own
dataset's knee.

## 5. Pin the banks from a natural-lane harvest

Serve the suite **once** with each stack generating to its own end-of-sequence, then freeze each request's
realised output length into the bank that both arms will then be held to:

```bash
python test/vp/pin_output_lengths_from_bank.py --harvest <harvest-root> --out <suites-dir>/<suite>.bank.json
```

The paper pins from the **base model's** natural lane, so the equal-work comparison asks how fast the same output
work executes. Pinning from the served arm's own lane is a legitimate second bank set and the paper reports it as
a companion table; the two are never mixed inside one campaign.

## 6. Run the paired campaign

```bash
python test/vp/run_paired_campaign.py --spec my_spec.json --out-dir <root> --reps 6
```

Add `--start-rep N` to resume. Each cell writes a `command.json` recording the exact commands, the artifact
hashes, the gates that ran and the status.

## Check

A cell counts as a measurement only if its own record says so:

```bash
python - <<'PY'
import json, glob
for f in sorted(glob.glob("<root>/rep*/*/*/cells/*.command.json")):
    d = json.load(open(f))
    if d["status"] != "completed":
        print(d["status"], d.get("failed_gates"), f)
PY
```

Anything that is not `completed` is excluded from every table, and the paper prints a dash in its place rather
than a number. See [`gates.md`](gates.md) for what each gate refuses and why.

Next: [`03-run-analysis.md`](03-run-analysis.md).
