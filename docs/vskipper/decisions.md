# The rulings the code refers to

Comments in this tree explain *why* a thing is the way it is, and several of those reasons are decisions taken
once and then binding. This page restates them so a reader does not need the development history.

## Measurement

**The baseline is upstream SGLang, served from its own tree.** Not this fork with the skipper disabled. That
configuration exists, is a legitimate arm, has its own name, and is reported separately in the ablation table —
where it shows the fork's own fixed cost. It is not the baseline, because a fork measured against itself cannot
show what the fork costs. Load points are calibrated to upstream's capacity for the same reason: a knee taken on
the treatment recalibrates the entire grid in the treatment's favour.

**The served design lives in the tree, never in the environment.** No runtime module reads an environment
variable for configuration, and a test fails the tree if one does. The reason is specific: an intended
configuration was once passed through an environment variable that never reached the server, roughly eighteen
hours of GPU time measured a design that had been rejected, and every output gate passed. A configuration that
can be set from outside the tree is a configuration that can silently fail to be set.

**A mechanism must be seen executing before its effect is claimed.** The attestation counters are the evidence. A
zero where the design says a leg ran is a refusal, not a rounding.

**Equal work, or no comparison.** Both arms of a paired cell are held to the same pinned output lengths. Unequal
work makes the cell void rather than interesting.

**Open-loop load.** Offered request rate is the control variable, never a client-side concurrency cap. Every
submitted request finishes and stays in the accounting; no tail is filtered.

**Quality numbers come from a third-party harness.** `lm-eval`, with the filters and metrics the task defines,
applied to the served generations. The project's own scorer is used for serving behaviour, never for a quality
claim in the paper.

**Non-inferiority at one percentage point.** The quality gate asks whether the served implementation reproduces
the checkpoint's own gap to its base model, within one point, on a paired per-document difference of differences.
A point estimate inside the margin whose interval is not is reported as *unresolved*, not as a failure: a failure
would assert the served arm is worse by more than the margin, which a wide interval does not establish.

## Design

**Compaction means count-bounded cohort execution.** Routed rows are gathered into dense cohorts under a bound on
the cohort *count*. A separate capacity-bounded packing policy exists in the design space and was never active in
any measured run; it is not a lane, not an ablation, and not a source of any reported gain.

**The engagement rule is a sufficient criterion, not an optimum.** It estimates where the K/V traffic the skipper
removes exceeds the routed body's fixed per-step cost, and engages only above that. It is deliberately
conservative: it gives up gain near the crossover rather than risk a loss below it.

**The interface refuses what it cannot express.** A skipper whose actions are not whole-layer run-or-project —
independent attention and MLP decisions, for instance — is refused at startup rather than silently lowered into
something the runtime can execute. A lowered policy is a different policy.

**Naming.** Cells, suites and banks are named by protocol and cap, never by an internal decision identifier.
