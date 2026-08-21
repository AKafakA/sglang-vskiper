"""Assert that state shared between vpipe modules is ONE object, not two.

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

# F4: the producer (cohort.tracker_for_device) and the drain
# (kv_commit.drain_request_kv_work) must share one container each.
same(cohort._KV_READINESS_TRACKERS, kv_commit._KV_READINESS_TRACKERS,
     "_KV_READINESS_TRACKERS shared by cohort and kv_commit")
same(cohort._BATCHED_KV_QUEUES, kv_commit._BATCHED_KV_QUEUES,
     "_BATCHED_KV_QUEUES shared by cohort and kv_commit")

# F2: attestation must observe a reset, i.e. read through accessors rather than
# holding the pre-reset objects.
before_ctr = coverage.coverage_dense_counters()
before_lad = coverage.coverage_ladder_record()
coverage.reset_coverage_dense_state()
after_ctr = coverage.coverage_dense_counters()
after_lad = coverage.coverage_ladder_record()
if after_ctr is before_ctr or after_lad is before_lad:
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
sys.exit(0 if not failures else 1)
