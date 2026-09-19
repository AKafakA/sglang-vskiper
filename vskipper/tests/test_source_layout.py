"""Reference-release source binding and resource ownership, run on the GPU host."""
from pathlib import Path
import importlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "vskipper/src/vskipper"


def test_serving_modules_have_one_canonical_identity():
    for directory in ("runtime", "kernels", "integration"):
        for path in sorted((PACKAGE / directory).rglob("*.py")):
            parts = path.relative_to(PACKAGE).with_suffix("").parts
            name = ".".join(("vskipper", *[p for p in parts if p != "__init__"]))
            module = importlib.import_module(name)
            assert Path(module.__file__).resolve() == path.resolve(), name
            aliases = [key for key, value in list(sys.modules.items())
                       if getattr(value, "__file__", None)
                       and Path(value.__file__).resolve() == path.resolve()]
            assert aliases == [name], (name, aliases)
    assert not any(n.startswith("sglang.srt.vpipe") for n in sys.modules)


def test_runtime_resources_are_in_their_own_package():
    from vskipper.runtime import design, roofline
    from vskipper.kernels import kernel
    assert design._repo_root() == ROOT
    assert (ROOT / design._ACTIVE_ARM_FILE).is_file()
    resources = Path(kernel.__file__).parent / "binary_cohort_configs"
    assert len(list(resources.glob("*.json"))) == 6
    for path in resources.glob("*.json"):
        assert json.loads(path.read_text())
    assert json.loads((Path(roofline.__file__).parent / "device_roofline.json").read_text())


def test_campaign_binds_both_fork_packages_and_only_upstream_baseline():
    from run_paired_campaign import serving_pythonpath
    spec = {"tree": "/candidate", "upstream_tree": "/upstream",
            "upstream_arms": {"upstream": [], "upstream_g1024": ["--cuda-graph-max-bs", "1024"]},
            "arms": {"baseline": "upstream_g1024", "treatment": "vskipper"}}
    assert serving_pythonpath(spec, "vskipper") == "/candidate/vskipper/src:/candidate/python"
    assert serving_pythonpath(spec, "upstream_g1024") == "/upstream/python"
    assert serving_pythonpath(spec, "upstream") == "/upstream/python"
    assert serving_pythonpath(spec, "stock") == "/candidate/vskipper/src:/candidate/python"


def test_documented_campaign_binds_upstream_baseline():
    from run_paired_campaign import serving_pythonpath, upstream_arms
    text = (ROOT / "vskipper/docs/02-run-experiments.md").read_text()
    spec = json.loads(text.split("```json\n", 1)[1].split("```", 1)[0])
    baseline = spec["arms"]["baseline"]
    assert serving_pythonpath(spec, baseline) == spec["upstream_tree"] + "/python"
    assert upstream_arms(spec)[baseline] == ["--cuda-graph-max-bs", "1024"]
    exemptions = spec["upstream_arm_exemptions"][baseline]
    assert exemptions["cuda_graph_max_bs"]["value"] == 1024
    assert exemptions["cuda_graph_config"]["accept_if"] == {
        "decode.max_bs": 1024, "decode.bs[max]": 1024}
