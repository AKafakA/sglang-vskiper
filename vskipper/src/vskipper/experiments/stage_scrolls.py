"""Stage SCROLLS validation splits to local jsonl for offline (CSD3) loading.
CSD3 has no outbound SSL, so the archived hf_hub_download path cannot run there."""
import datasets, json, hashlib, os, sys
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
manifest = {}
for cfg in ("gov_report", "summ_screen_fd", "qmsum"):
    d = datasets.load_dataset("tau/scrolls", cfg, split="validation", trust_remote_code=True)
    path = os.path.join(OUT, f"{cfg}.validation.jsonl")
    h = hashlib.sha256()
    with open(path, "w") as f:
        for r in d:
            # normalise to the load_records contract: {context, input, answers}
            line = json.dumps({"id": r["id"], "context": r["input"], "input": "",
                               "answers": [r["output"]]}, sort_keys=True) + "\n"
            f.write(line); h.update(line.encode())
    manifest[cfg] = {"rows": len(d), "sha256": h.hexdigest(), "file": os.path.basename(path)}
    print("%-16s rows=%-5d sha256=%s" % (cfg, len(d), h.hexdigest()[:16]))
mp = os.path.join(OUT, "MANIFEST.json")
json.dump({"source": "tau/scrolls", "split": "validation",
           "note": "context=SCROLLS 'input' (the document); answers=[SCROLLS 'output']",
           "datasets_version": datasets.__version__, "tasks": manifest}, open(mp, "w"), indent=1, sort_keys=True)
print("manifest:", mp)
