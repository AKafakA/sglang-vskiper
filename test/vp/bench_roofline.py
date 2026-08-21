"""Empirically validate the crossover-model roofline (I-54): where does A30 decode become
compute-bound? Sweep vanilla decode throughput vs concurrency; the aggregate should keep
rising then SATURATE around batch ≈ R (A30 FLOP:byte ≈ 177). VP r=0 span-full should track
vanilla (tax removed) — confirming §5.5's premise before the cross-cohort win experiment.
Qwen2.5-1.5B, eager. Reports aggregate tok/s + per-req tok/s (the saturation tell).
"""

import os
import time

MODEL = os.environ.get("VP_AGREE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
CONCS = [32, 64, 128, 256, 512]
NTOK = 96
SPAN_FULL = 7  # Qwen2.5-1.5B = 7 blocks → span 7 = 1 forward/token (tax-free r=0)


def eng(vp, conc):
    for k in ("SGLANG_VP_SCHED", "SGLANG_VP_HIT_RATE", "SGLANG_VP_SPAN"):
        os.environ.pop(k, None)
    if vp:
        os.environ["SGLANG_VP_SCHED"] = "1"
        os.environ["SGLANG_VP_HIT_RATE"] = "0.0"
        os.environ["SGLANG_VP_SPAN"] = str(SPAN_FULL)
    import sglang as sgl
    return sgl.Engine(model_path=MODEL, mem_fraction_static=0.8, disable_cuda_graph=True,
                      disable_radix_cache=True, disable_overlap_schedule=True,
                      max_running_requests=conc, tp_size=1)


def bench(llm, conc):
    prompts = [f"Write a technical paragraph number {i} about distributed consensus and fault tolerance:"
               for i in range(conc)]
    sp = {"temperature": 0.0, "max_new_tokens": NTOK}
    llm.generate(prompts[:4], {"temperature": 0.0, "max_new_tokens": 8})
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t0
    toks = sum(o["meta_info"]["completion_tokens"] for o in outs)
    return toks / dt, toks / dt / conc


def main():
    print(f"=== A30 decode roofline (vanilla vs VP r=0 span-full), {MODEL} ===", flush=True)
    print(f"{'conc':>5} {'van_agg':>9} {'van/req':>8} {'vp_agg':>9} {'vp/req':>8} {'vp/van':>7}", flush=True)
    for c in CONCS:
        try:
            e = eng(False, c); va, vr = bench(e, c); e.shutdown()
            e = eng(True, c); pa, pr = bench(e, c); e.shutdown()
            print(f"{c:>5} {va:>9.1f} {vr:>8.2f} {pa:>9.1f} {pr:>8.2f} {100*pa/va:>6.0f}%", flush=True)
        except Exception as ex:
            print(f"{c:>5} FAILED: {type(ex).__name__}: {str(ex)[:80]}", flush=True)
    print("ROOFLINE DONE", flush=True)


if __name__ == "__main__":
    main()
