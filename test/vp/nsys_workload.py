"""Minimal VP+bank decode workload for nsys/torch profiling — a few decode steps at a fixed
batch so the profiler captures the per-forward VP overhead (launches, router, ledger, syncs)."""
import os
os.environ["SGLANG_VP_SCHED"]="1"; os.environ["SGLANG_VP_SPAN"]=os.environ.get("SPAN","3")
os.environ["SGLANG_VP_BANK_DIR"]=os.environ.get("BANK_DIR","/mydata/banks/bank_4b_mp")
os.environ["SGLANG_VP_HIT_RATE"]="0.0"
MODEL=os.environ.get("VP_AGREE_MODEL","Qwen/Qwen3-4B"); CONC=int(os.environ.get("CONC","128"))
def main():
    import sglang as sgl
    e=sgl.Engine(model_path=MODEL, mem_fraction_static=0.85, disable_cuda_graph=True,
                 disable_radix_cache=True, disable_overlap_schedule=True, max_running_requests=CONC, tp_size=1)
    p=[f"Explain topic {i} in detail with reasoning and examples:" for i in range(CONC)]
    e.generate(p, {"temperature":0.0,"max_new_tokens":32,"frequency_penalty":1.2,"ignore_eos":True})
    e.shutdown(); print("NSYS-WORKLOAD DONE")
if __name__=="__main__": main()
