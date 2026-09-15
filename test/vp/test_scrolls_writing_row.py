import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from transformers import AutoTokenizer
import labeled_workload as L

tok = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B-Instruct")
print("composite:", L.WORKLOAD_DATASETS["longctx_writing"])
print("phase:", L.WORKLOAD_PHASE["longctx_writing"])
tot = 0
for ds in L.WORKLOAD_DATASETS["longctx_writing"]:
    items = L.load_dataset_items(ds, 10**6, tok, False, context_length=8192)
    tot += len(items)
    pt = sorted(i.evaluator_data["prompt_tokens"] for i in items)
    out = items[0].reference_output_len
    print("  %-24s survivors=%-4d prompt p50=%-5d max=%-5d out=%-5d metric=%s"
          % (ds, len(items), pt[len(pt)//2], pt[-1], out, items[0].metric))
    assert pt[-1] + out <= 8192, "window violated"
    assert items[0].gold and items[0].gold[0], "no reference summary"
print("  TOTAL survivors:", tot)
print("  weights:", {k: v for k, v in L.DEFAULT_DATASET_WEIGHTS.items() if k.startswith("scrolls_")})
