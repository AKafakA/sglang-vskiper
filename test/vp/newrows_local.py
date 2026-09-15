"""Locally staged rows for the balanced long-output rows (owner, 2026-09-06).

Two datasets, read from a staged directory (CSD3 has no outbound network; the
staging happens on a box with network and the files travel with a MANIFEST that
binds the frozen suite to its input data, exactly like `scrolls_local.py`):

* ``longbench_write`` — LongBench-Write (LongWriter, THUDM 2024): 120 long-form
  writing prompts with an explicit required length in words (``length``). The
  model's own chat template renders the prompt (LongWriter's ``pred.py`` does the
  same). Quality: the authors' length score S_l (offline formula) and their
  GPT-4o quality score S_q (judge; only with the owner's API approval).
* ``lca_libgen`` — Long Code Arena, library-based code generation (JetBrains
  Research 2024): 150 instructions, each asking for a complete program built on
  one open-source library; the reference program and the library's API
  identifiers are given. The prompt follows the baseline's "with API list"
  setting: instruction + the library's defined elements. Quality: the
  benchmark's ChrF and API-recall metrics (offline).

Row selection is a WINDOW FILTER (ruling A1, never truncation): a row is kept
when its chat-templated prompt plus its reference output budget fits the
context. For LongBench-Write the budget is the required length in words
converted to tokens with the measured English ratio; for LCA it is the
reference program's token count.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterator, Optional

_STAGE_DIR_ENV = "VP_NEWROWS_STAGE_DIR"
FILES = {
    "longbench_write": "longbench_write.jsonl",
    "lca_libgen": "lca_libgen.jsonl",
    "lca_libgen_noapi": "lca_libgen.jsonl",  # same rows, instruction-only prompt (D-526)
    "lca_libgen_official": "lca_libgen.jsonl",  # same rows, benchmark protocol (D-531)
    "longwriter6k": "longwriter6k.jsonl",  # serving-scale writing row (owner 2026-09-06)
    "livecodebench": "livecodebench.jsonl",  # serving-scale code row (owner 2026-09-06)
}
# LiveCodeBench's own prompt (lcb_runner/prompts/code_generation.py, verbatim; the
# LLaMa3 style = system + user through the model's chat template).
LCB_SYSTEM_MESSAGE = (
    "You are an expert Python programmer. You will be given a question (problem "
    "specification) and will generate a correct Python program that matches the "
    "specification and passes all tests."
)
LCB_FORMAT_WITH_STARTER = (
    "You will use the following starter code to write the solution to the problem "
    "and enclose your code within delimiters."
)
LCB_FORMAT_WITHOUT_STARTER = (
    "Read the inputs from stdin solve the problem and write the answer to stdout "
    "(do not directly test on the sample inputs). Enclose your code within "
    "delimiters as follows. Ensure that when the python program runs, it reads "
    "the inputs, runs the algorithm and writes output to STDOUT."
)


def lcb_user_prompt(record: dict[str, Any]) -> str:
    """`get_generic_question_template_answer`, character for character."""

    prompt = f"### Question:\n{record['question_content']}\n\n"
    starter = str(record.get("starter_code") or "")
    if starter:
        prompt += f"### Format: {LCB_FORMAT_WITH_STARTER}\n"
        prompt += f"```python\n{starter}\n```\n\n"
    else:
        prompt += f"### Format: {LCB_FORMAT_WITHOUT_STARTER}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


_WORDS_RE = None


def longwriter_required_words(record: dict[str, Any]) -> Optional[int]:
    """Required length stated in a LongWriter-6k prompt ("... 3000 words ...",
    "2,000-word ..."), for the S_l length score; None when the prompt states
    no length. Only the first explicit word count is used."""

    global _WORDS_RE
    if _WORDS_RE is None:
        import re

        _WORDS_RE = re.compile(r"(\d{1,3}(?:,\d{3})+|\d{3,6})\s*(?:-|\s)?\s*words?\b", re.IGNORECASE)
    match = _WORDS_RE.search(str(record["prompt"]))
    return int(match.group(1).replace(",", "")) if match else None


def render_messages(dataset: str, record: dict[str, Any]) -> list[dict[str, str]]:
    """Chat messages for the row: every staged dataset is a single user turn
    except LiveCodeBench, whose benchmark prompt carries a system message."""

    if dataset == "livecodebench":
        return [
            {"role": "system", "content": LCB_SYSTEM_MESSAGE},
            {"role": "user", "content": lcb_user_prompt(record)},
        ]
    return [{"role": "user", "content": render_prompt(dataset, record)}]
# The benchmark's own generation limit (lca-baselines src/models/*_model.py:
# `max_tokens=2048`, `temperature=0.0`). Applied as the per-request cap ONLY on
# `lca_libgen_official` (owner: the only capped long-output row, because
# Llama-3-8B-Instruct loops on 14-15 % of these prompts in both arms).
LCA_OFFICIAL_MAX_TOKENS = 2048
# Verbatim from lca-baselines src/models/example_generation_model.py
# (`get_prompt`), the benchmark's no-context setting.
LCA_OFFICIAL_PROMPT = (
    "Generate Python code based on the following instruction. Output ONLY code. "
    "DO NOT include explanations or other textual content.\nInstruction: {instruction}"
)
# Measured on the staged prompts with the Llama-3 tokenizer ( survey):
# ~1.35 tokens per English word. Used ONLY for the window filter budget of
# LongBench-Write rows; the served generation is natural (uncapped).
LONGWRITE_TOKENS_PER_WORD = 1.35


def stage_dir() -> Path:
    raw = os.environ.get(_STAGE_DIR_ENV, "").strip()
    if not raw:
        raise RuntimeError(
            f"{_STAGE_DIR_ENV} is unset. LongBench-Write / LCA rows are read from "
            "locally staged jsonl (CSD3 has no outbound network). Stage them on a "
            f"networked box and point {_STAGE_DIR_ENV} at that directory."
        )
    path = Path(raw)
    if not path.is_dir():
        raise RuntimeError(f"{_STAGE_DIR_ENV}={raw} is not a directory")
    return path


def manifest() -> dict[str, Any]:
    path = stage_dir() / "MANIFEST.json"
    if not path.is_file():
        raise RuntimeError(
            f"staged directory {stage_dir()} has no MANIFEST.json -- re-stage; "
            "the manifest carries the row counts and sha256 that bind a frozen "
            "suite to its input data"
        )
    return json.loads(path.read_text())


def _path(dataset: str) -> Path:
    if dataset not in FILES:
        raise ValueError(f"unknown staged dataset {dataset!r}")
    path = stage_dir() / FILES[dataset]
    if not path.is_file():
        raise RuntimeError(f"staged file {path} is missing")
    return path


def verify_manifest(dataset: str) -> None:
    """Fail closed if the staged file does not match the manifest's sha256."""

    entry = manifest().get("files", {}).get(FILES[dataset])
    if entry is None:
        raise RuntimeError(f"MANIFEST.json has no entry for {FILES[dataset]}")
    digest = hashlib.sha256(_path(dataset).read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        raise RuntimeError(
            f"{FILES[dataset]} sha256 {digest} != manifest {entry['sha256']}"
        )


def iter_records(dataset: str) -> Iterator[dict[str, Any]]:
    """Yield staged rows in file order (the staging order is the seal)."""

    verify_manifest(dataset)
    with _path(dataset).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def render_prompt(dataset: str, record: dict[str, Any]) -> str:
    if dataset in ("longbench_write", "longwriter6k"):
        # LongWriter pred.py: the prompt is the user message, verbatim.
        return str(record["prompt"])
    if dataset == "lca_libgen_official":
        # Declared protocol `lca-libgen-v1:official-nocontext-chat:cap2048`: the
        # benchmark's own prompt template, verbatim, as the user message.
        return LCA_OFFICIAL_PROMPT.format(instruction=str(record["instruction"]))
    if dataset == "lca_libgen_noapi":
        # Declared protocol `lca-libgen-v1:instruction-only-chat` (our own
        # instruction-only wording;). Superseded by `lca_libgen_official`.
        return (
            f"{str(record['instruction']).strip()}\n\n"
            "Write the complete Python program. Output only the code."
        )
    if dataset == "lca_libgen":
        # Declared protocol `lca-libgen-v1:instruction+api-list-chat`: the
        # instruction, then the library's defined elements (the benchmark's
        # "with API list" context), then the completion request.
        elements = ", ".join(str(e) for e in record["project_defined_elements"])
        return (
            f"{str(record['instruction']).strip()}\n\n"
            f"You may use the following functions and classes from the "
            f"{record['repo_name']} library:\n{elements}\n\n"
            "Write the complete Python program. Output only the code."
        )
    raise ValueError(f"unknown staged dataset {dataset!r}")


def reference_tokens(record: dict[str, Any], tokenizer: Any) -> int:
    """Token count of an LCA reference program (information / window budget)."""

    return len(tokenizer(str(record["clean_reference"]), add_special_tokens=False)["input_ids"])


def output_budget_tokens(dataset: str, record: dict[str, Any], tokenizer: Any) -> int:
    """Output budget used by the window filter. For the natural rows it is the
    reference length (never a cap); for `lca_libgen_official` it is the
    benchmark's generation limit, which the suite also applies as the
    per-request cap (`task_reference_limit` policy)."""

    if dataset == "longbench_write":
        words = int(record["length"])
        return int(math.ceil(words * LONGWRITE_TOKENS_PER_WORD))
    if dataset in ("lca_libgen", "lca_libgen_noapi"):
        return reference_tokens(record, tokenizer)
    if dataset == "lca_libgen_official":
        return LCA_OFFICIAL_MAX_TOKENS
    if dataset == "longwriter6k":
        # the staged reference response (owner 2026-09-06: filter on prompt +
        # provided reference response <= context; the served generation is
        # natural and uncapped)
        return len(tokenizer(str(record["reference"]), add_special_tokens=False)["input_ids"])
    if dataset == "livecodebench":
        # no reference response exists (tests only): prompt-only window filter
        return 0
    raise ValueError(f"unknown staged dataset {dataset!r}")


def fits_window(prompt_tokens: int, budget_tokens: int, context_length: int) -> bool:
    return prompt_tokens + budget_tokens <= context_length
