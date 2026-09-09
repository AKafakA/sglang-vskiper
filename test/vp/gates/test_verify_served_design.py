"""Prove the input gate FAILS CLOSED.

A gate that has only ever been seen to pass is not evidence of anything -- that is precisely
how the output gates behaved on 2026-09-09, all green while the wrong system was measured.
So verify_served_design.py is exercised against a synthetic /server_info for each way it must
REFUSE, including the actual defect that cost ~18 h of A100 time (low_row full_dual reaching
the server while the campaign believed it had set off).

Runs anywhere, no GPU, no model:  python3 test/vp/gates/test_verify_served_design.py
"""
import json, subprocess, sys, threading, http.server, socketserver
from pathlib import Path

# the tree this test lives in -- never a hard-coded checkout path
TREE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(TREE / "python"))
import importlib.util
spec = importlib.util.spec_from_file_location("d", str(TREE / "python/sglang/srt/vpipe/design.py"))
d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)
GOOD = d.design_attestation()

def serve(payload):
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(payload).encode()
            self.send_response(200); self.send_header("Content-Type","application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers()
            self.wfile.write(body)
        def log_message(self, *a): pass
    srv = socketserver.TCPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]

CASES = [
    ("matching design + arm", {"internal_states":[{"vp_runtime":{"arm":"integrated_it4","design":GOOD}}]}, "integrated_it4", 0),
    ("WRONG ARM served",      {"internal_states":[{"vp_runtime":{"arm":"vdec_fd","design":GOOD}}]}, "integrated_it4", 1),
    ("low_row full_dual (the $15 defect)",
        {"internal_states":[{"vp_runtime":{"arm":"integrated_it4","design":{**GOOD,"low_row_policy":"full_dual"}}}]}, "integrated_it4", 1),
    ("regime switch OFF",
        {"internal_states":[{"vp_runtime":{"arm":"integrated_it4","design":{**GOOD,"regime_switch":None}}}]}, "integrated_it4", 1),
    ("zero routed layers",
        {"internal_states":[{"vp_runtime":{"arm":"integrated_it4","design":{**GOOD,"routed_layers":[]}}}]}, "integrated_it4", 1),
    ("attestation not wired (old server)",
        {"internal_states":[{"vp_runtime":{}}]}, "integrated_it4", 1),
    ("no vp_runtime at all", {"internal_states":[{}]}, "integrated_it4", 1),
]
ok = True
for name, payload, arm, want in CASES:
    srv, port = serve(payload)
    r = subprocess.run([sys.executable, str(TREE / "test/vp/gates/verify_served_design.py"),
                        "--url", f"http://127.0.0.1:{port}", "--arm", arm,
                        "--tree", str(TREE)], capture_output=True, text=True)
    srv.shutdown()
    got = r.returncode
    mark = "PASS" if got == want else "**WRONG**"
    if got != want: ok = False
    print(f"  {mark:9s} exit={got} (want {want})  {name}")
print("\nGATE SELF-TEST:", "all cases behave correctly" if ok else "DEFECTIVE")
sys.exit(0 if ok else 1)
