#!/usr/bin/env python3
"""Prove the input gate FAILS CLOSED.

A gate that has only ever been seen to pass is not evidence of anything -- that is exactly how
the output gates behaved on 2026-09-09, all green while the wrong system was measured. So the
gate is driven against a synthetic /server_info for each way it must REFUSE, including:

  * the actual defect that cost ~18 h of A100 time (low_row full_dual reaching the server while
    the campaign believed it had set `off`), and
  * an EXTRA served knob (prefill.max_tokens=6144 -- the one D-596 deleted), which the first
    version of the comparator silently ignored (Codex F11).

The intended side is computed by the gate itself from the deployed tree, so each case starts
from the real resolved design and perturbs one field. That means this must run somewhere the
tree is importable -- i.e. a serving host, not the code box.

    /opt/vpipe/venv/bin/python test/vp/gates/test_verify_served_design.py
"""
from __future__ import annotations

import copy
import http.server
import json
import socketserver
import subprocess
import sys
import threading
from pathlib import Path

TREE = Path(__file__).resolve().parents[3]
GATE = TREE / "test/vp/gates/verify_served_design.py"
ARM = "integrated_it4"


def serve(payload):
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = socketserver.TCPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def wrap(served):
    return {"internal_states": [{"vp_runtime": {"served_design": served}}]}


def main() -> int:
    sys.path.insert(0, str(TREE / "python"))
    from sglang.srt.vpipe import design as d
    from sglang.srt.vpipe.common import resolved_design_attestation

    d._ARM_CACHE.clear()
    d._ARM_CACHE["name"] = ARM
    good = resolved_design_attestation()

    def perturb(fn):
        s = copy.deepcopy(good)
        fn(s)
        return s

    def set_max_tokens(s):
        s["regime_switch"]["prefill"]["max_tokens"] = 6144

    cases = [
        ("matching resolved design", wrap(good), 0),
        ("WRONG ARM served", wrap(perturb(lambda s: s.__setitem__("arm", "vdec_fd"))), 1),
        ("low_row full_dual (the $15 defect)",
         wrap(perturb(lambda s: s.__setitem__("low_row_policy", "full_dual"))), 1),
        ("regime switch OFF", wrap(perturb(lambda s: s.__setitem__("regime_switch", None))), 1),
        ("EXTRA knob: prefill.max_tokens=6144 (Codex F11)", wrap(perturb(set_max_tokens)), 1),
        ("skipper swapped to the mock",
         wrap(perturb(lambda s: s.__setitem__("skipper", "deterministic_mock"))), 1),
        ("compact phases narrowed",
         wrap(perturb(lambda s: s.__setitem__("compact_phases", ["decode"]))), 1),
        ("served_design absent (old server)", {"internal_states": [{"vp_runtime": {}}]}, 1),
        ("no vp_runtime at all", {"internal_states": [{}]}, 1),
    ]

    ok = True
    for name, payload, want in cases:
        srv, port = serve(payload)
        r = subprocess.run(
            [sys.executable, str(GATE), "--url", f"http://127.0.0.1:{port}",
             "--arm", ARM, "--tree", str(TREE)],
            capture_output=True, text=True,
        )
        srv.shutdown()
        mark = "PASS" if r.returncode == want else "**WRONG**"
        if r.returncode != want:
            ok = False
            print(r.stdout[-400:])
        print(f"  {mark:10s} exit={r.returncode} (want {want})  {name}")

    print("\nGATE SELF-TEST:", "all cases behave correctly" if ok else "DEFECTIVE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
