# Gates

A number is not a result until its cell's gates pass. These are the gates, what each one refuses, and the failure
that put it there. They live in `test/vp/gates/` (drivers) and `test/vp/gate_tests/` (the pytest layer).

## Before a campaign starts

| gate | refuses |
|---|---|
| `campaign_preflight.py` | a campaign whose **inputs** are not staged: either tree missing or at the wrong revision, model absent, banks absent or hash-mismatched |
| `verify_served_design.py` | a server whose **attested resolved design** does not match the `design.py` entry the arm names |

These two exist because both failures have happened and neither was caught by anything downstream. A host was
once re-staged without the upstream baseline tree at all, and the whole campaign ran; separately, an intended
configuration was passed through an environment variable that never reached the server, and every output gate
still passed. Output gates check the answer. These check that the right system was asked the question.

## Per cell

| gate | refuses |
|---|---|
| work identity | a paired cell whose two arms did not generate the same number of output tokens. Equal work is the premise of the comparison, so unequal work is not a result to interpret, it is a void cell |
| accounting | a cell where submitted, started, completed and reported request counts disagree, or where any request is missing from the accounting |
| zero-empty | a cell containing an empty generation that is not on the witnessed known list in `gates/known_empties.json` |
| arrival fidelity | a cell whose realised arrival times drifted from the frozen trace |
| routed-pass observation | a cell where the arm declares a routed leg but the attested counters show that leg never executed |
| cross-arm configuration | a pair of servers whose resolved configurations differ outside a declared allowlist |

The last one deserves emphasis: **a mechanism must be seen executing in the attestation before its effect is
claimed.** A flag that is set is not a treatment that ran. An arm that declares a routed decode leg and attests
zero routed decode passes has not measured that leg, whatever its latency says.

## Tree-level

| gate | proves |
|---|---|
| `gates_box.sh` | the chain: helper build, import gates, ladder default modes, the four-arm posture gate, and route-digest output equality against a frozen tree |
| `box_vpcov_arm.sh <arm>` | one arm serves, routes and attests, on a 32-request greedy probe. Never a performance number |
| `route_digest_compare.py` | generated-text and route-digest equality between two trees |
| `verify_tree_equivalence.py` | that two checkouts are the same tree where the comparison requires it |

**A known instrument artifact:** `tape.digest_sum_u64` differs across boots on an identical tree. It is not
signal; the other digest fields are.

## What a failed gate does

It excludes the cell. The paper prints a dash where a gated-out cell would have been, rather than a number with a
footnote. One cell in the reported campaign failed this way — an ablation arm below the knee on GSM8K, which
failed the routed-pass observation — and both of its columns are dashes in the ablation table.
