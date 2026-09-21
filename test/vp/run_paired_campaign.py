#!/usr/bin/env python3
"""Drive the paired performance campaign: N reps x datasets x rates x two arms.

Why this is a committed file. Every performance table this project has produced came from
hand-written shell on an ephemeral box, and every one of those scripts is now gone. That is
not only a reproducibility complaint -- it is how the v1.3 table came to be measured on a
design that no longer exists (73 commits and four deleted mechanisms later), and how eleven
of twelve drivers invoked gates through a pipe that discarded their exit status. A table
that decides the paper is not a thing to re-improvise each time.

What this enforces, structurally rather than by remembering:

  * PAIRING. A cell is (dataset, rate, rep). Both arms measure the SAME pinned equal-work
    suite at the SAME rate in the same rep, so the paired unit is a within-rep delta and the
    CI is over reps. Cross-arm work identity is a code gate elsewhere (GR-1a); here the
    guarantee is that neither arm can be handed different work, because both are handed one
    suite name.
  * ARM ORDER ALTERNATES BY REP PARITY. Odd reps run stock first, even reps run vSkipper
    first. Arm order is worth ~3% on a shared node -- unalternated, that bias sits entirely
    on one arm and is indistinguishable from the effect.
  * EVERY PINNED SUITE EXISTS BEFORE ANYTHING BOOTS. A campaign that discovers a missing
    suite in rep 4 has burned hours to find out what one stat() call knew at the start.
  * THE SERVED ARM IS READ BACK, NOT ASSUMED. `deploy/active_arm` is what we asked for;
    /server_info is what runs. A set flag is not an active treatment -- a whole night's
    quality numbers once described the no-skip body while every input gate passed.
  * GATES. Per-cell enforcement lives inside run_qps_evaluation and refuses a bad cell
    there; this driver does not re-implement it and does not paper over its exit status.

Usage:
    run_paired_campaign.py --spec campaign.json --out-dir OUT [--reps 6] [--start-rep 1]

The spec is data, not code (QPS grids are contract data):

    {"tree": "/opt/vpipe/trees/tree-318bc23929",
     "python": "/opt/vpipe/venv/bin/python",
     "model_path": "/dev/shm/vpipe/models/Meta-Llama-3-8B-Instruct-53346005",
     "suites_dir": "/dev/shm/vpipe/suites",
     "staging_root": "/dev/shm/vpipe",
     "host_config": "deploy/hosts/a100.json",
     "expect_dir": "/opt/vpipe/campaign",
     "arms": {"baseline": "upstream", "treatment": "vskipper"},
     "datasets": {"gsm8k": {"r11p25": 11.25, "r14p25": 14.25, "r18p75": 18.75}}}
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

# The Llama-3-8B-Instruct defaults; a spec may name another model with "model_revision" and
# "served_model_name" (v1.5: the Qwen3-4B feasibility row uses the same driver, same gates).
MODEL_REVISION = "53346005fb0ef11d3b6a83b12c895cca40156b6c"
SERVED_MODEL_NAME = "NousResearch/Meta-Llama-3-8B-Instruct"


def model_revision(spec: dict[str, Any]) -> str:
    return str(spec.get("model_revision", MODEL_REVISION))


def served_model_name(spec: dict[str, Any]) -> str:
    return str(spec.get("served_model_name", SERVED_MODEL_NAME))


_PROFILE_CACHE: dict[str, dict[str, Any]] = {}


def launch_profile_name(spec: dict[str, Any]) -> str | None:
    return spec.get("launch_profile")


def launch_profile(spec: dict[str, Any]) -> dict[str, Any]:
    """The launch profile's arguments, read from the TREE's design table (never from here)."""
    name = launch_profile_name(spec)
    key = f"{spec['tree']}::{name}"
    if key not in _PROFILE_CACHE:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(spec["tree"]) / "python")
        code = ("import json, sys; from sglang.srt.vpipe import design; "
                "profiles = getattr(design, 'SERVED_LAUNCH_PROFILES', None); "
                "name = sys.argv[1] or getattr(design, 'SERVED_LAUNCH_PROFILE_DEFAULT', 'paper'); "
                "print(json.dumps(None if profiles is None else profiles[name]))")
        done = subprocess.run([spec["python"], "-c", code, name or ""], env=env,
                              capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(f"launch profile {name!r} could not be read from the tree: "
                               f"{done.stderr.strip()[-400:]}")
        profile = json.loads(done.stdout.strip().splitlines()[-1])
        if profile is None:
            # A tree from before the profiles existed (e.g. the measured 735d451c89, booted by
            # this driver for tree equivalence) hard-coded the paper protocol in its OWN driver;
            # booting it under any other launch would compare the wrong thing. Say so.
            if name not in (None, "paper"):
                raise RuntimeError(f"tree {spec['tree']} declares no launch profiles; only the "
                                   f"paper protocol it hard-coded can be reproduced, not {name!r}")
            log(f"    launch profile: tree declares none -- paper protocol (fp16, 0.8) as that "
                f"tree's own driver hard-coded it")
            profile = {"dtype": "float16", "mem_fraction_static": 0.8}
        _PROFILE_CACHE[key] = profile
    return _PROFILE_CACHE[key]


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {message}", flush=True)


def suite_name(dataset: str, label: str) -> str:
    """The pinned equal-work suite for one (dataset, rate). Built by the harvest."""
    return f"{dataset}_eqw_{label}"


def preflight(spec: dict[str, Any]) -> list[str]:
    """Everything checkable before a GPU is touched. Returns complaints."""
    problems: list[str] = []
    required = ("tree", "python", "model_path", "suites_dir", "staging_root", "expect_dir",
                "host_config", "arms", "datasets", "source_revision")
    missing = [key for key in required if key not in spec]
    if missing:
        return [f"spec is missing required keys: {', '.join(missing)}"]
    tree = Path(spec["tree"])
    suites = Path(spec["suites_dir"])
    for key in ("tree", "python", "model_path", "suites_dir", "staging_root", "expect_dir"):
        if not Path(spec[key]).exists():
            problems.append(f"{key} does not exist: {spec[key]}")
    if not (tree / spec["host_config"]).is_file():
        problems.append(f"host config missing: {tree / spec['host_config']}")
    for role, arm in campaign_arms(spec):
        expect = Path(spec["expect_dir"]) / f"expect_{arm}.json"
        if not expect.is_file():
            problems.append(f"{role} arm {arm}: no expected-runtime file {expect}")
    for dataset, rates in spec["datasets"].items():
        for label in rates:
            name = suite_name(dataset, label)
            for suffix in ("requests.jsonl", "metadata.jsonl"):
                path = suites / f"{name}.{suffix}"
                if not path.is_file():
                    problems.append(f"pinned suite missing: {path}")
    return problems


def server_info(port: int) -> Any:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/server_info", timeout=30) as r:
        return json.load(r)


def served_arm(port: int) -> str:
    """What the live server says it is serving -- not what we asked it to serve."""
    info = server_info(port)
    try:
        return str(info["internal_states"][0]["vp_runtime"]["served_design"]["arm"])
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(
            "live /server_info carries no vp_runtime.served_design.arm; refusing to "
            "attribute a measurement to an arm the server will not name"
        ) from error


def upstream_arms(spec: dict[str, Any]) -> dict[str, list[str]]:
    """The upstream-served arms of this campaign: arm name -> EXTRA launch arguments.

    Default = the single literal `upstream` arm with no extra arguments. The ladder-control
    campaign (2026-09-16, incident `incident-graph-ladder-confound-20260916`) adds a second
    upstream-served arm, `upstream_g1024` = the SAME upstream checkout launched with
    `--cuda-graph-max-bs 1024`, so the baseline captures the decode ladder the fork captures.
    Declared in the campaign spec as `"upstream_arms": {"upstream": [], "upstream_g1024":
    ["--cuda-graph-max-bs", "1024"]}`; every extra argument must also be declared to Gate E
    through `upstream_arm_exemptions` (below), or the boot is refused.
    """
    declared = spec.get("upstream_arms")
    if declared is None:
        return {spec.get("upstream_arm_name", "upstream"): []}
    return {name: list(extra) for name, extra in declared.items()}


def arm_is_upstream(spec: dict[str, Any], arm: str) -> bool:
    """Is this arm GENUINE upstream SGLang, served from its own tree?

    The paper's baseline is a freshly-cloned upstream checkout with no vpipe/ package
    (owner order), NOT ARMS["stock"] -- which is this fork with the skipper off and
    whose own comment falsely claimed to be upstream for months.
    """
    return arm in upstream_arms(spec)


def serving_pythonpath(spec: dict[str, Any], arm: str) -> Path:
    """The `python/` directory this arm will actually serve from."""
    tree = Path(spec["upstream_tree"]) if arm_is_upstream(spec, arm) else Path(spec["tree"])
    return tree / "python"


def campaign_preflight_gate(spec: dict[str, Any],
                            arm: str) -> subprocess.CompletedProcess[str]:
    """G1a — campaign_preflight, bound to the PYTHONPATH this arm is about to export.

    Extracted from main() so the QUALITY lane runs the same gate as the perf lane. It had
    never run against a genuine upstream tree at all: its manifest claimed 2319 files when
    the commit yields 2079, so no correct tree could pass it. Hashing a directory
    does not bind it to the process that serves, which is why --serving-pythonpath exists
    (audit).
    """
    serving = serving_pythonpath(spec, arm)
    command = [sys.executable,
               str(Path(spec["tree"]) / "test/vp/gates/campaign_preflight.py"),
               "--tree", spec["tree"], "--upstream", spec["upstream_tree"],
               # The STAGING ROOT, which holds models/ beside suites/. Passing suites_dir
               # made the gate look for models/ inside it and report "served model weights
               # 0 shards" on a box with the model staged one level up .
               "--workdir", spec["staging_root"],
               # The campaign's own host config, NAMED. The gate used to pick the
               # alphabetically first deploy/hosts/*.json for itself and validated the HPC cluster
               # paths on a Vast box .
               "--host-config", spec["host_config"],
               "--serving-pythonpath", str(serving)]
    if arm_is_upstream(spec, arm):
        command.append("--serving-is-upstream")
    return subprocess.run(command, capture_output=True, text=True)


class Server:
    """One booted arm. Boot, attest, tear down -- and never leave a process behind."""

    def __init__(self, spec: dict[str, Any], arm: str, port: int, log_path: Path) -> None:
        self.spec, self.arm, self.port, self.log_path = spec, arm, port, log_path
        self.upstream = arm_is_upstream(spec, arm)
        self.process: subprocess.Popen[bytes] | None = None

    def command(self) -> list[str]:
        # Beyond the substrate, ONLY the launch profile's arguments are passed
        # (design.SERVED_LAUNCH_PROFILES, declared by name; the spec picks one). `--dtype=float16`
        # and `--mem-fraction-static 0.8` used to be hard-coded here, carried from a Turing dev
        # host and undeclared in every headline cell since v1.3 ; they are now the
        # "paper" profile, declared in SERVED_LAUNCH_EXEMPTIONS and checked by Gate E at boot.
        profile_args: list[str] = []
        for key, value in launch_profile(self.spec).items():
            profile_args += [f"--{key.replace('_', '-')}", str(value)]
        return [
            self.spec["python"], "-m", "sglang.launch_server",
            "--model-path", self.spec["model_path"],
            "--revision", model_revision(self.spec),
            "--served-model-name", served_model_name(self.spec),
            "--host", "127.0.0.1", "--port", str(self.port),
            *profile_args,
            "--attention-backend=triton",
            "--prefill-attention-backend=triton",
            "--decode-attention-backend=triton",
            # per-arm extra launch arguments exist ONLY for upstream-served arms (the same-ladder
            # control); fork arms are selected in the tree and take no per-arm flags.
            *(upstream_arms(self.spec).get(self.arm, []) if self.upstream else []),
        ]

    def __enter__(self) -> Server:
        tree = Path(self.spec["tree"])
        environment = dict(os.environ)
        environment["SGLANG_IS_FLASHINFER_AVAILABLE"] = "false"
        if self.upstream:
            # Served from the upstream tree, with NO vpipe environment at all. There is no
            # active_arm to write and no host config to point at -- that tree has neither.
            environment["PYTHONPATH"] = str(Path(self.spec["upstream_tree"]) / "python")
            for key in ("SGLANG_VP_HOST_CONFIG", "SGLANG_FD_WEIGHTS"):
                environment.pop(key, None)
        else:
            # The arm is selected IN THE TREE, never through the environment .
            (tree / "deploy/active_arm").write_text(self.arm + "\n")
            environment["PYTHONPATH"] = str(tree / "python")
            environment["SGLANG_VP_HOST_CONFIG"] = str(tree / self.spec["host_config"])
        handle = self.log_path.open("wb")
        self.process = subprocess.Popen(
            self.command(), stdout=handle, stderr=subprocess.STDOUT,
            env=environment, start_new_session=True,
        )
        for _ in range(180):
            if self.process.poll() is not None:
                raise RuntimeError(f"{self.arm} server exited during boot; see {self.log_path}")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=3).read()
                break
            except Exception:
                time.sleep(5)
        else:
            raise RuntimeError(f"{self.arm} server never became healthy; see {self.log_path}")

        if self.upstream:
            # For upstream the ABSENCE of the block is the attestation, and its PRESENCE
            # means the fork was launched by mistake -- the exact confusion that made every
            # prior campaign measure ARMS["stock"] while calling it upstream.
            info = server_info(self.port)
            try:
                design = info["internal_states"][0]["vp_runtime"]["served_design"]
            except (KeyError, IndexError, TypeError):
                design = None
            if design is not None:
                raise RuntimeError(
                    f"arm {self.arm!r} is declared upstream but the live server carries "
                    f"vp_runtime.served_design ({design.get('arm')!r}) — this is our fork. "
                    "Refusing to record it as upstream."
                )
            log(f"    {self.arm} up on :{self.port}, attested UPSTREAM (no vp_runtime)")
            default_conformance_gate(self.spec, self.arm, self.port)
            return self
        # G1b — INTENDED vs RESOLVED design, on the live server, before the cell runs.
        # verify_served_design's only invocation sites were cell_gates.py (which nothing
        # calls) and a box smoke script, so no campaign has ever run it. It is a PRE gate
        # because a wrong configuration cannot be repaired once the GPU is spent.
        gate = Path(self.spec["tree"]) / "test/vp/gates/verify_served_design.py"
        done = subprocess.run(
            [sys.executable, str(gate), "--url", f"http://127.0.0.1:{self.port}",
             "--arm", self.arm, "--tree", self.spec["tree"]],
            capture_output=True, text=True,
        )
        if done.returncode != 0:
            tail = (done.stdout + done.stderr).strip().splitlines()
            raise RuntimeError(
                f"G1 served-design gate REFUSED arm {self.arm!r}: "
                f"{tail[-1][:200] if tail else 'no output'}"
            )
        log(f"    G1 served design == intended design ({self.arm})")
        default_conformance_gate(self.spec, self.arm, self.port)

        live = served_arm(self.port)
        if live != self.arm:
            raise RuntimeError(
                f"asked for arm {self.arm!r} but the server is serving {live!r} — refusing "
                "to record a measurement under the wrong arm name"
            )
        log(f"    {self.arm} up on :{self.port}, attested as {live!r}")
        return self

    def __exit__(self, *_: object) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            self.process.wait(timeout=30)


def default_conformance_gate(spec: dict[str, Any], arm: str, port: int) -> None:
    """Gate E (restored by): this server runs at SGLang's DEFAULTS for its model
    on this device, except the knobs design.SERVED_LAUNCH_EXEMPTIONS names. Both arms, at
    boot, before any GPU time is spent. The cross-arm gate cannot see what both arms share;
    this one compares each arm against the defaults the BASELINE tree computes."""
    snapshot = Path(spec["staging_root"]) / f"server_info.conformance.{arm}.{port}.json"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(json.dumps(server_info(port), indent=2, sort_keys=True) + "\n")
    # The gate is THIS driver's (a legacy tree booted for equivalence has neither the gate nor
    # the declarations); the design tree is the spec's, and the gate falls back to this
    # driver's declarations when that tree declares none (the paper protocol it hard-coded).
    gate = Path(__file__).resolve().parent / "gates/verify_default_conformance.py"
    done = subprocess.run(
        [sys.executable, str(gate), "--server-info", str(snapshot),
         "--model-path", spec["model_path"], "--defaults-tree", spec["upstream_tree"],
         "--design-tree", spec["tree"], "--python", spec["python"],
         "--fallback-design-tree", str(Path(__file__).resolve().parents[2]),
         *(["--launch-profile", launch_profile_name(spec)] if launch_profile_name(spec) else []),
         # an upstream-served arm launched with extra arguments must declare EVERY resolved
         # field those arguments change (spec["upstream_arm_exemptions"][arm]); otherwise the
         # gate holds it to the defaults and refuses -- the ladder-control lesson made a gate.
         *(["--arm-exemptions", json.dumps(spec["upstream_arm_exemptions"][arm])]
           if arm in spec.get("upstream_arm_exemptions", {}) else [])],
        capture_output=True, text=True,
    )
    if upstream_arms(spec).get(arm) and arm not in spec.get("upstream_arm_exemptions", {}):
        raise RuntimeError(
            f"arm {arm!r} is launched with extra arguments {upstream_arms(spec)[arm]} but the spec "
            "declares no upstream_arm_exemptions for it; refusing (undeclared execution difference)")
    for line in done.stdout.splitlines():
        log(f"      {line}")
    if done.returncode != 0:
        raise RuntimeError(
            f"Gate E default conformance REFUSED arm {arm!r}: an undeclared non-default "
            f"launch value (see lines marked '!'); {done.stderr.strip()[-300:]}"
        )
    log(f"    Gate E default conformance ({arm}): PASS")


def runner_command(spec: dict[str, Any], arm: str, manifest: Path, qps_config: Path,
                   cells_dir: Path, port: int, resume: bool = False) -> list[str]:
    """The run_qps_evaluation.py invocation for ONE booted arm.

    Shared by the perf lane (run_arm_cells) and the natural-lane harvest (run_natural_harvest.py)
    so the two clients of the quality table are driven by one argv, not two hand-copied ones.
    """
    tree = Path(spec["tree"])
    return ([spec["python"], str(tree / "test/vp/run_qps_evaluation.py"),
         # THE EXPERIMENT NAME MUST BE THE ARM. run_qps_evaluation asserts
         # `deployment["system_id"] == args.experiment` (:1545), and this driver sets
         # --system-id to the arm (:252). Passing a composite label here raised
         #   ValueError: deployment system 'upstream' does not match experiment
         #               'paired-gsm8k-upstream-rep1'
         # on BOTH arms of the smoke -- 100 % of cells, one minute in, before any request
         # was served. Nothing is lost by matching: the dataset, rate and rep are already
         # carried by the output path (rep<N>/<dataset>/<arm>/) and by the cell filename
         # (<suite>_qps<rate>_rep<N>.jsonl).
         "--experiment", arm,
         "--deployment-manifest", str(manifest),
         "--model", served_model_name(spec), "--workload-dir", spec["suites_dir"],
         "--qps-config", str(qps_config), "--output-dir", str(cells_dir),
         "--evidence-class", spec.get("evidence_class", "development"),
         "--host", "127.0.0.1", "--port", str(port), "--reps", "1",
         "--runner-source-revision", spec["source_revision"],
         # ONE REFUSED RATE MUST NOT COST THE OTHER TWO.
         #
         # An arm-run covers every rate of its dataset in one invocation. Without this,
         # the FIRST refused cell aborts the whole invocation and the remaining rates are
         # never attempted. Measured 2026-09-11: gsm8k r8p25 was refused on
         # `skipping_executed` (peak occupancy 96 against the design's enter_rows=176, so
         # the load-aware switch correctly never left prod_allrun) -- and that refusal
         # took r10p45 and r13p75 with it, despite both engaging cleanly at peak 389/386.
         # The arm produced nothing, so GR-1a failed all three rates and rep 1 gsm8k
         # yielded ZERO usable pairs.
         #
         # The ladder has always passed this for the same reason: a rung that fails its
         # gates must not abandon the band. A refused cell is still refused -- its
         # artifacts are renamed INVALID.* and GR-1a fails that rate -- it simply no
         # longer takes its siblings down.
         "--continue-after-accounting-rejection"]
        # A re-measure boot writes into the SAME cells directory (every downstream reader --
        # cell_artifact, paired_analysis, the node's bundle scripts -- looks there); the
        # runner only accepts an existing directory in resume mode.
        + (["--resume-completed"] if resume else [])
        + (["--upstream-baseline"] if arm_is_upstream(spec, arm) else []))


# A REFUSED CELL IS RE-MEASURED, NEVER ACCEPTED AND NEVER ABANDONED (owner rule 3 of the
# cell-measurement rules, restated 2026-09-21 after the H100 upstream 0.75x cell was refused
# on 8 empty generations -- 3 "Duplicate request ID" + 5 stream aborts, server-side and
# sporadic, not the policy under test). Without this the chain ended with rc=1 and a human
# had to notice, build a clean root and re-run by hand. Now the driver re-measures ONLY the
# refused rates of that arm on a FRESH boot, up to this many attempts in total. Every
# rejected attempt's artifacts stay on disk under cells/rejected/attempt<k>/ (the rejection
# is evidence too); the boot-level files of a superseded boot go to boots/attempt<k>/.
MAX_CELL_ATTEMPTS = 3

BOOT_FILES = ("server_info.before.json", "launch_command.json", "deployment_manifest.json",
              "qps.json", "runner.log", "server_info.after.json")


def refused_cells(cells_dir: Path, dataset: str, rates: dict[str, float]) -> dict[str, float]:
    """The rates of this arm-run with NO valid result artifact (refused, crashed or never run)."""
    refused: dict[str, float] = {}
    for label, rate in rates.items():
        suite = suite_name(dataset, label)
        valid = [p for p in cells_dir.glob(f"{suite}_qps*_rep*.jsonl")
                 if "arrival" not in p.name and "load" not in p.name]
        if not valid:
            refused[label] = rate
    return refused


def shelve_rejected(cells_dir: Path, dataset: str, refused: dict[str, float], attempt: int) -> None:
    """Move every file of the refused cells (INVALID.* included) out of the runner's way."""
    shelf = cells_dir / "rejected" / f"attempt{attempt}"
    shelf.mkdir(parents=True, exist_ok=True)
    for label in refused:
        suite = suite_name(dataset, label)
        for path in list(cells_dir.glob(f"*{suite}_qps*")):
            if path.is_file():
                path.rename(shelf / path.name)
    manifest = cells_dir / "run_manifest.json"
    if manifest.is_file():
        (shelf / "run_manifest.json").write_bytes(manifest.read_bytes())


def run_arm_boot(spec: dict[str, Any], dataset: str, rates: dict[str, float],
                 arm: str, rep: int, cell_root: Path, port: int, attempt: int) -> int:
    """The given rates for one (dataset, arm, rep) against ONE boot. Returns the runner's rc."""
    tree = Path(spec["tree"])
    cells_dir = cell_root / "cells"
    if attempt > 1:
        previous = cell_root / "boots" / f"attempt{attempt - 1}"
        previous.mkdir(parents=True, exist_ok=True)
        for name in BOOT_FILES:
            if (cell_root / name).is_file():
                (cell_root / name).rename(previous / name)
    server_log = cell_root / ("server.log" if attempt == 1 else f"server.attempt{attempt}.log")

    with Server(spec, arm, port, server_log) as server:
        (cell_root / "server_info.before.json").write_text(
            json.dumps(server_info(port), indent=2, sort_keys=True) + "\n")

        launch = cell_root / "launch_command.json"
        launch.write_text(json.dumps(server.command()) + "\n")
        manifest = cell_root / "deployment_manifest.json"
        subprocess.run(
            [spec["python"], str(tree / "test/vp/make_deployment_manifest.py"),
             "--deployment-id", f"paired-{dataset}-{arm}-rep{rep}",
             "--system-id", arm, "--model", served_model_name(spec),
             "--model-revision", model_revision(spec),
             "--client-tokenizer-path", spec["model_path"],
             "--source-revision", spec["source_revision"],
             "--launch-command-file", str(launch),
             "--expected-runtime-json", str(Path(spec["expect_dir"]) / f"expect_{arm}.json"),
             "--host", "127.0.0.1", "--port", str(port), "--out", str(manifest)],
            check=True, capture_output=True,
        )

        qps_config = cell_root / "qps.json"
        qps_config.write_text(json.dumps(
            {"workloads": {suite_name(dataset, label): [rate] for label, rate in rates.items()}}
        ) + "\n")

        # DURATION AND MIN-PROMPTS ARE THE RUNNER'S, NOT OURS (owner rule 2 /).
        #
        # This block used to compute `rows // slowest` and pass --min-prompts/--duration-s.
        # Two things were wrong with it. The runner now REFUSES those flags outright unless
        # --partial-suite-diagnostic is given, so every cell died one minute in with
        #   "--duration-s / --min-prompts are DIAGNOSTIC overrides"
        # -- the same defect that killed the ladder driver , in a second caller.
        # And the arithmetic was itself the banned pattern: one duration derived from the
        # SLOWEST rate, applied to every rate in the cell, makes the faster rates submit
        # fewer than their whole suite. The runner derives duration = rows/qps per cell,
        # which is the only form that keeps work identical across rates.
        completed = subprocess.run(
            runner_command(spec, arm, manifest, qps_config, cells_dir, port,
                           resume=cells_dir.is_dir()),
            check=False, stdout=(cell_root / "runner.log").open("wb"),
            stderr=subprocess.STDOUT,
        )
        (cell_root / "server_info.after.json").write_text(
            json.dumps(server_info(port), indent=2, sort_keys=True) + "\n")
    return completed.returncode


def run_arm_cells(spec: dict[str, Any], dataset: str, rates: dict[str, float],
                  arm: str, rep: int, out_dir: Path, port: int) -> int:
    """All rates for one (dataset, arm, rep); refused rates re-measured on fresh boots.

    Returns the last runner's rc, or 1 if any rate is still refused after MAX_CELL_ATTEMPTS.
    """
    cell_root = out_dir / f"rep{rep}" / dataset / arm
    cell_root.mkdir(parents=True, exist_ok=True)
    cells_dir = cell_root / "cells"
    rc = run_arm_boot(spec, dataset, rates, arm, rep, cell_root, port, attempt=1)
    for attempt in range(2, MAX_CELL_ATTEMPTS + 1):
        refused = refused_cells(cells_dir, dataset, rates)
        if not refused:
            break
        log(f"    {arm}: refused cell(s) {sorted(refused)} -> re-measuring on a fresh boot, "
            f"attempt {attempt}/{MAX_CELL_ATTEMPTS} (rejected artifacts kept under "
            f"cells/rejected/attempt{attempt - 1})")
        shelve_rejected(cells_dir, dataset, refused, attempt - 1)
        rc = run_arm_boot(spec, dataset, refused, arm, rep, cell_root, port, attempt=attempt)
    refused = refused_cells(cells_dir, dataset, rates)
    if refused:
        log(f"    {arm}: STILL REFUSED after {MAX_CELL_ATTEMPTS} attempts: {sorted(refused)} "
            f"-- unquotable, needs a human")
        rc = rc or 1
    return rc


def cell_artifact(cell_root: Path, suite: str) -> Path | None:
    """The one results .jsonl for a suite, excluding the sidecar arrival/load traces."""
    matches = [p for p in (cell_root / "cells").rglob(f"{suite}_qps*_rep*.jsonl")
               if "arrival" not in p.name and "load" not in p.name]
    return matches[0] if len(matches) == 1 else None


# THE FROZEN CROSS-ARM ALLOWLIST (plan F1: run --report ONCE, classify, then freeze).
#
# Each entry is a CLAIM: "this field differs between the arms, and here is why it cannot
# flatter the treatment." Anything not listed is refused, and the gate exits 1.
#
# Measured 2026-09-11 01:3xZ by running --report on two REAL manifests -- an upstream ladder
# boot and an integrated_it4 bank harvest, both on this box. Of 33 compared fields, exactly
# TWO differ:
#
#   vp_runtime                    THE TREATMENT. Present on the fork, `[]` on upstream. This
#                                 difference IS the experiment.
#
#   server.max_total_num_tokens   upstream 393319 vs integrated_it4 389479 = -0.976 %.
#                                 A DERIVED consequence: the router and projector weights
#                                 occupy HBM, leaving the fork a smaller KV pool. It takes
#                                 capacity AWAY FROM THE TREATMENT, so it can only make our
#                                 result look worse, never better -- which is the condition
#                                 for declaring rather than fixing it. The plan predicted
#                                 this field and this direction before it was measured.
#
# Frozen in CODE, not in each spec, because every spec that hard-coded ["vp_runtime"] would
# otherwise have had G1c refuse 100 % of its cells -- the headline and all twelve sweep
# points -- on a field we had already classified. A spec may still ADD to this set; it cannot
# silently replace it.
CROSS_ARM_DECLARED = frozenset({"vp_runtime", "server.max_total_num_tokens"})


def campaign_arms(spec: dict[str, Any]) -> list[tuple[str, str]]:
    """(role, arm) for every arm the campaign boots: the baseline anchor and each treatment."""
    return [("baseline", spec["arms"]["baseline"])] + [
        ("treatment", arm) for arm in treatments_of(spec)]


def treatments_of(spec: dict[str, Any]) -> list[str]:
    """`arms.treatment` (one, the headline) or `arms.treatments` (a list: a sweep whose points
    share ONE baseline anchor inside the same session -- the plan's 12-arms-plus-anchor design,
   /N1). Exactly one of the two keys is accepted."""
    arms = spec["arms"]
    if ("treatment" in arms) == ("treatments" in arms):
        raise SystemExit("spec.arms must carry exactly one of 'treatment' or 'treatments'")
    return [arms["treatment"]] if "treatment" in arms else list(arms["treatments"])


def cross_arm_config_gate(spec: dict[str, Any], dataset: str, rep: int,
                          out_dir: Path, treatment: str) -> bool:
    """Are the two arms the same engine apart from the treatment? Returns True on PASS.

    Every other config gate checks ONE arm against its own intent. None of them can see
    whether the two arms are comparable TO EACH OTHER -- which is the question a paired table
    rests on, and which stopped being nearly free the moment the baseline became a separate
    upstream tree 193 commits away. Identical CLI flags no longer imply identical
    resolved configuration.

    The allowlist comes from `spec["cross_arm_allow"]` and is a set of CLAIMS: each entry
    says "this field differs and cannot flatter the treatment". It is built once from
    `--report` on the smoke, then frozen.
    """
    baseline = spec["arms"]["baseline"]
    manifests = {
        arm: out_dir / f"rep{rep}" / dataset / arm / "deployment_manifest.json"
        for arm in (baseline, treatment)
    }
    missing = [a for a, m in manifests.items() if not m.is_file()]
    if missing:
        log(f"    G1c cross-arm config: FAIL — no manifest for {', '.join(missing)}")
        return False
    command = [sys.executable,
               str(Path(spec["tree"]) / "test/vp/gates/verify_cross_arm_config.py")]
    for arm, m in manifests.items():
        command += ["--arm", f"{arm}={m}"]
    for field in sorted(CROSS_ARM_DECLARED | set(spec.get("cross_arm_allow", []))):
        command += ["--allow", field]
    done = subprocess.run(command, capture_output=True, text=True)
    (out_dir / f"rep{rep}" / dataset / "cross_arm_config.txt").write_text(
        done.stdout + done.stderr)
    if done.returncode != 0:
        tail = [l for l in done.stdout.splitlines() if l.startswith("  ! ")]
        log(f"    G1c cross-arm config: FAIL — undeclared: {', '.join(t.strip('! ') for t in tail[:4])}")
        return False
    log("    G1c cross-arm config: PASS (arms differ only in declared fields)")
    return True


def cross_arm_work_gate(spec: dict[str, Any], dataset: str, rates: dict[str, float],
                        rep: int, out_dir: Path, treatment: str) -> list[str]:
    """GR-1a — did the two arms do the SAME WORK? Returns the rate labels that FAILED.

    This gate cannot live in run_qps_evaluation's per-cell gates: it compares two arms, and
    a cell knows only its own. So it has had NO CALLER anywhere in the repo -- `cell_gates.py`,
    the file that runs it, is invoked by nothing -- and cross-arm claims on unmatched work is
    the failure that already cost this project ~$100 and two weeks, with a final warning
    attached. A pairing driver is the only place it can be enforced, so it is enforced here.

    An UNCHECKED pair is a FAILED pair, never a passing one: a missing artifact fails the
    rate rather than skipping it.
    """
    baseline = spec["arms"]["baseline"]
    failed: list[str] = []
    for label in rates:
        suite = suite_name(dataset, label)
        specs, missing = [], []
        for role, arm in (("baseline", baseline), ("treatment", treatment)):
            artifact = cell_artifact(out_dir / f"rep{rep}" / dataset / arm, suite)
            if artifact is None:
                missing.append(f"{role}({arm})")
            else:
                specs.append(f"{arm}={artifact}")
        if missing:
            log(f"    GR-1a {suite}: FAIL — no unique artifact for {', '.join(missing)}; "
                "an unchecked pair is a failed pair")
            failed.append(label)
            continue
        # --out is REQUIRED by the gate's own CLI. Omitting it made every invocation exit 2
        # on argparse before reading a single artifact -- which this wrapper would have
        # recorded as a work-identity FAILURE, marking all 54 paired cells unquotable after
        # a ten-hour run. The verdict is written beside the cells so it is evidence, not
        # just an exit code.
        verdict = out_dir / f"rep{rep}" / dataset / f"workgate_{suite}.json"
        # --reference names the arm whose per-request lengths ARE the identity table. Its
        # default is "V-dec-rs", a legacy arm this campaign does not have, so leaving it
        # unset raised `reference arm 'V-dec-rs' not among arms` on every call -- the second
        # of two independent ways this gate would have failed 100% of cells while looking
        # like a genuine work-identity refusal in the log.
        #
        # The BASELINE is the reference: the equal-work suite is pinned from banked baseline
        # lengths, so the treatment is what must match it, not the other way round.
        command = [sys.executable, str(Path(spec["tree"]) / "test/vp/cross_arm_work_gate.py"),
                   "--cell", f"{suite}_rep{rep}", "--out", str(verdict),
                   "--reference", baseline]
        for item in specs:
            command += ["--arm", item]
        done = subprocess.run(command, cwd=Path(spec["tree"]), capture_output=True, text=True)
        if done.returncode != 0:
            tail = (done.stdout + done.stderr).strip().splitlines()
            log(f"    GR-1a {suite}: FAIL — {tail[-1][:150] if tail else 'no output'}")
            failed.append(label)
        else:
            log(f"    GR-1a {suite}: PASS (equal work across arms)")
    return failed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--start-rep", type=int, default=1)
    ap.add_argument("--port", type=int, default=32051)
    ap.add_argument("--dry-run", action="store_true",
                    help="preflight only: prove every suite and arm exists, boot nothing")
    args = ap.parse_args()

    spec = json.loads(args.spec.read_text())
    problems = preflight(spec)
    if problems:
        print("PREFLIGHT FAILED — nothing has booted:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2
    log(f"preflight OK: {len(spec['datasets'])} datasets, "
        f"{sum(len(r) for r in spec['datasets'].values())} rates, arms {spec['arms']}")

    # G1a — campaign_preflight, whose ONLY caller was the dead cell_gates.py. It is the gate
    # whose literal job is "upstream baseline staged and genuinely stock", and it had never
    # been run against a genuine upstream tree: its manifest claimed 2319 files when the
    # commit yields 2079, so no correct tree could pass it . Run once per arm here,
    # binding the PYTHONPATH about to be exported to the tree just content-verified --
    # hashing a directory does not bind it to the process that serves (audit).
    if "upstream_tree" in spec:
        for role, arm in campaign_arms(spec):
            serving = serving_pythonpath(spec, arm)
            done = campaign_preflight_gate(spec, arm)
            if done.returncode != 0:
                print(f"G1a campaign preflight REFUSED for {role} arm {arm!r} — "
                      "nothing has booted:", file=sys.stderr)
                sys.stderr.write(done.stdout[-2500:] + done.stderr[-1000:])
                return 2
            log(f"  G1a campaign preflight PASS ({role}={arm}, serving {serving})")
    if args.dry_run:
        for dataset, rates in spec["datasets"].items():
            for label, rate in sorted(rates.items(), key=lambda kv: kv[1]):
                log(f"  would run {dataset} {suite_name(dataset, label)} @ {rate}")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    baseline, treatments = spec["arms"]["baseline"], treatments_of(spec)
    index: list[dict[str, Any]] = []
    failures = 0

    for rep in range(args.start_rep, args.start_rep + args.reps):
        # Alternate, so arm order cannot masquerade as the effect. A sweep (several treatments,
        # one anchor) keeps the anchor at one end of the session; with a single rep the order
        # effect is not alternated away and the caption must say so.
        order = [baseline] + treatments if rep % 2 == 1 else treatments + [baseline]
        for dataset, rates in spec["datasets"].items():
            log(f"=== rep {rep} · {dataset} · order {' -> '.join(order)} ===")
            for arm in order:
                started = time.time()
                try:
                    rc = run_arm_cells(spec, dataset, rates, arm, rep, args.out_dir, args.port)
                except Exception as error:  # boot / attestation / manifest
                    log(f"    {arm}: FAILED — {error}")
                    rc, failures = 1, failures + 1
                else:
                    if rc != 0:
                        failures += 1
                        log(f"    {arm}: runner rc={rc} (a refused cell is a refused cell)")
                log(f"    {arm}: {int(time.time() - started) // 60} min, rc={rc}")
                index.append({"rep": rep, "dataset": dataset, "arm": arm,
                              "order": order, "rc": rc,
                              "rates": rates,
                              "path": str(args.out_dir / f"rep{rep}" / dataset / arm)})
                (args.out_dir / "campaign_index.json").write_text(
                    json.dumps({"spec": str(args.spec), "cells": index}, indent=2) + "\n")

            # Both arms of this (rep, dataset) are now on disk, which is the earliest moment
            # GR-1a can be asked. A failure here does NOT stop the campaign -- it marks these
            # rates unquotable, which is what the contract says a failed equal-work gate means.
            for treatment in treatments:
                config_ok = cross_arm_config_gate(spec, dataset, rep, args.out_dir, treatment)
                if not config_ok:
                    failures += 1
                work_failures = cross_arm_work_gate(spec, dataset, rates, rep, args.out_dir,
                                                    treatment)
                if work_failures:
                    failures += len(work_failures)
                index.append({"rep": rep, "dataset": dataset, "gate": "GR-1a",
                              "treatment": treatment,
                              "failed_rates": work_failures,
                              "cross_arm_config_ok": config_ok,
                              "quotable": bool(config_ok) and not work_failures})
            (args.out_dir / "campaign_index.json").write_text(
                json.dumps({"spec": str(args.spec), "cells": index}, indent=2) + "\n")

    log(f"--- campaign done: {len(index)} arm-runs, {failures} with a non-zero rc ---")

    # PAIRED ANALYSIS IS PART OF THE RUN, NOT A TOOL SOMEONE REMEMBERS .
    #
    # The plan asks for this explicitly, and it was not true: this driver wrote 54 cells and
    # computed no delta -- "the paired unit is a within-rep delta" existed only in a docstring.
    # The one tool with the right method was written for the v1.3 layout and could not read
    # these artifacts at all, and hardcoded two knees that are now void. So the analysis runs
    # HERE, on the cells just written, and its report lands beside them.
    #
    # It is deliberately NOT allowed to change the campaign's exit status: a paired table that
    # reports parity is a valid result, not a failure, and a campaign must not appear to have
    # failed because its answer was "no difference".
    log("--- paired analysis (within-rep deltas, t-based 95 % CI, straddle rule) ---")
    for treatment in treatments:
        suffix = "" if len(treatments) == 1 else f".{treatment}"
        analysis = subprocess.run(
            [sys.executable, str(Path(spec["tree"]) / "test/vp/paired_analysis.py"),
             str(args.out_dir),
             "--baseline", baseline, "--treatment", treatment,
             "--json", str(args.out_dir / f"paired_report{suffix}.json")],
            capture_output=True, text=True,
        )
        (args.out_dir / f"paired_table{suffix}.txt").write_text(analysis.stdout + analysis.stderr)
        for line in analysis.stdout.splitlines():
            log(f"  {line}")
        if analysis.returncode != 0:
            log(f"  paired analysis for {treatment} produced no gated pair -- the cells are on "
                "disk, the table is not")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
