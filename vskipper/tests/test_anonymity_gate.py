"""The release scanner includes itself and requires private, external rules."""
from pathlib import Path
import shutil
import subprocess

import pytest


GATE = Path(__file__).resolve().parents[1] / "scripts" / "check_anonymity.sh"


@pytest.fixture
def scan_tree(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    gate = tree / "check_anonymity.sh"
    shutil.copyfile(GATE, gate)
    (tree / "README.md").write_text("A reproducible anonymous fixture.\n")
    subprocess.run(["git", "-c", "init.defaultBranch=fixture", "init", "-q", str(tree)],
                   check=True)
    subprocess.run(["git", "-C", str(tree), "add", "."], check=True)
    terms = tmp_path / "private.tsv"
    terms.write_text("fixture identity\towner-marker\n")
    return tree, gate, terms


def run_gate(tree, *args):
    return subprocess.run(["bash", "check_anonymity.sh", *map(str, args)],
                          cwd=tree, capture_output=True, text=True)


def test_clean_tree(scan_tree):
    tree, _, terms = scan_tree
    result = run_gate(tree, "--terms-file", terms)
    assert result.returncode == 0, result.stderr
    assert "CLEAN:" in result.stdout


@pytest.mark.parametrize("kind", ["checker", "content", "filename"])
def test_identifying_matches(scan_tree, kind):
    tree, gate, terms = scan_tree
    if kind == "checker":
        gate.write_text(gate.read_text() + "\n# owner-marker\n")
    elif kind == "content":
        (tree / "README.md").write_text("owner-marker\n")
    else:
        (tree / "owner-marker.txt").write_text("neutral content\n")
        subprocess.run(["git", "-C", str(tree), "add", "."], check=True)
    result = run_gate(tree, "--terms-file", terms, "--list")
    assert result.returncode == 1, result.stderr
    assert "BLOCKED:" in result.stdout
    assert "CLEAN:" not in result.stdout


@pytest.mark.parametrize("content", ["", "# no rules\n", "missing-pattern\n",
                                    "bad regex\t[\n"])
def test_invalid_rule_file(scan_tree, content):
    tree, _, terms = scan_tree
    terms.write_text(content)
    assert run_gate(tree, "--terms-file", terms).returncode == 2


def test_rules_cannot_be_inside_repository(scan_tree):
    tree, _, terms = scan_tree
    internal = tree / "private.tsv"
    shutil.copyfile(terms, internal)
    assert run_gate(tree, "--terms-file", internal).returncode == 2


def test_missing_rules(scan_tree):
    tree, _, _ = scan_tree
    assert run_gate(tree).returncode == 2
    assert run_gate(tree, "--terms-file").returncode == 2
    assert run_gate(tree, "--terms-file", tree / "absent").returncode == 2
