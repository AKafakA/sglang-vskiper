"""Assert that state shared between vpipe modules is ONE object, not two.

Updated after the async K/V removal: the K/V containers are asserted ABSENT
rather than shared, so this file fails loudly if that subsystem is reimplemented
without restoring a single-owner check.

Three split-mutable-state defects were found in this refactor: a module global
defined in two places so the reader saw a counter nothing incremented; an
attestation module holding pre-reset objects; and K/V containers whose producer
and drain lived on different dictionaries. In every case the code looked correct
and the four-arm gate stayed green, because the affected path either never ran
or degraded silently.

Identity is the only check that catches this class. Run as a script; exits
non-zero on failure.
"""
import sys

failures = []


def same(a, b, label):
    if a is b:
        print(f"  OK    {label}")
    else:
        print(f"  SPLIT {label}: {id(a)} != {id(b)}")
        failures.append(label)


from sglang.srt.vpipe import cohort, kv_commit, coverage, attestation

# The K/V readiness tracker and batched queues were REMOVED with the async K/V
# subsystem (see the removed-feature register). Assert they are gone rather than
# leaving a check that would silently pass on nothing.
for mod, nm in ((cohort, "_KV_READINESS_TRACKERS"), (kv_commit, "_KV_READINESS_TRACKERS"),
                (cohort, "_BATCHED_KV_QUEUES"), (kv_commit, "_BATCHED_KV_QUEUES")):
    if hasattr(mod, nm):
        print(f"  RESURRECTED {mod.__name__}.{nm} -- if async K/V is reimplemented, "
              "put the container in ONE module and restore the identity assertion")
        failures.append(f"{mod.__name__}.{nm} came back without a shared-state check")
    else:
        print(f"  OK    {mod.__name__}.{nm} absent (removed with async K/V)")

# F2: attestation must observe a reset, i.e. read through accessors rather than
# holding the pre-reset objects.
before_ctr = coverage.coverage_dense_counters()
coverage.reset_coverage_dense_state()
after_ctr = coverage.coverage_dense_counters()
if after_ctr is before_ctr:
    print("  NOTE  reset did not rebind; the accessor test is vacuous here")
else:
    print("  OK    reset rebinds the coverage objects (so stale reads are possible)")

# recapture_events lives on CoverageDenseCounters, not the ladder record.
coverage.record_recapture_event(4242)
seen = list(coverage.coverage_dense_counters().recapture_events)
if 4242 in seen:
    print("  OK    accessor observes a post-reset write")
else:
    print(f"  FAIL  post-reset write not observed: {seen[:5]}")
    failures.append("accessor observes post-reset write")

# The end-to-end assertion: build the real attestation block AFTER the reset and
# confirm it carries the post-reset write. This is what F2 actually broke.
blk = attestation.coverage_dense_runtime_attestation(None)
if blk is None:
    print("  NOTE  attestation block is None in this posture (not armed); "
          "accessor identity above still covers the fix")
else:
    events = (blk.get("ladder") or {}).get("recapture_events")
    if events is None:
        events = ((blk.get("counters") or {}).get("recapture_events"))
    if events is not None and 4242 in list(events):
        print("  OK    attestation block reflects the post-reset write")
    else:
        print(f"  FAIL  attestation block stale after reset: {events}")
        failures.append("attestation block reflects post-reset write")

# The single-definition invariant, stated directly.
same(coverage._c3_counters, coverage.coverage_dense_counters(),
     "coverage._c3_counters is what the accessor returns")

print("SHARED STATE:", "PASS" if not failures else f"FAIL {failures}")

if __name__ == "__main__":
    sys.exit(0 if not failures else 1)
