"""SCROLLS long-context summarisation, loaded from LOCALLY STAGED jsonl.

Why local rather than a hub fetch: the HPC cluster has no outbound SSL, so the archived
``longbench_eval.load_records`` pattern (``hf_hub_download`` + ``zipfile``)
cannot run there. And ``tau/scrolls`` is a script-based dataset, which
``datasets`` >= 3.0 refuses outright -- the cluster's lm-eval venv has 3.6.0 and the
box has 4.8.4. The staging step therefore runs on the box under a venv pinned
to ``datasets<3``, writes one jsonl per task, and the serving harness reads
those files with nothing but the stdlib.

Staged row shape, normalised to the ``longbench_eval.load_records`` contract so
the two paths stay interchangeable::

    {"id": ..., "context": <the document>, "input": "", "answers": [<reference>]}

SCROLLS' own field names are ``input`` (the document) and ``output`` (the
reference summary); the staging script maps them, which is why ``context``
holds the document and ``input`` is empty for these tasks.

Window policy (owner ruling A1, 2026-09-04): FILTER, never truncate. The model
is Llama-3-8B with ``max_position_embeddings = 8192``, while SCROLLS documents
average 12.6k (gov_report) to 14.2k (qmsum) tokens. Truncating to fit would
score ROUGE against a summary of a document the model only half saw, making the
number incomparable to any published SCROLLS result -- which is the whole reason
the axis uses SCROLLS validation splits. So documents that do not fit are
DROPPED, and the surviving counts are recorded. Measured survivors at
``output=1024``: gov_report 330/972 (34.0 %), summ_screen_fd 145/338 (42.9 %),
qmsum 38/272 (14.0 %) = 513 requests / 3.12 M tokens.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

# SCROLLS' own recommendation for the summarisation tasks (``scrolls/task.py``
# notes 1024 for GovReport); LongBench's subset uses 512. The axis takes 1024
# because the longer output raises decode's share of the cell, which is what
# the mechanism acts on.
GENLEN: dict[str, int] = {
    "scrolls_gov_report": 1024,
    "scrolls_summ_screen_fd": 1024,
    "scrolls_qmsum": 1024,
}

# Prompt templates. gov_report and summ_screen_fd are pure summarisation;
# qmsum is QUERY-BASED, and its query is embedded at the head of the document by
# SCROLLS itself, so the same template serves all three.
_PROMPT: dict[str, str] = {
    "scrolls_gov_report": (
        "You are given a report by a government agency. Write a one-page "
        "summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page "
        "summary of the report.\n\nSummary:"
    ),
    "scrolls_summ_screen_fd": (
        "You are given a script of a TV episode. Summarise the episode in a "
        "paragraph.\n\nEpisode script:\n{context}\n\nNow, summarise the "
        "episode.\n\nSummary:"
    ),
    "scrolls_qmsum": (
        "You are given a meeting transcript and a query. Answer the query "
        "based on the transcript.\n\nTranscript:\n{context}\n\nNow, answer the "
        "query.\n\nAnswer:"
    ),
}

_STAGE_DIR_ENV = "VP_SCROLLS_STAGE_DIR"


def stage_dir() -> Path:
    """Resolve the staged-jsonl directory, failing closed if unset or missing.

    Deliberately NOT defaulted to a hub fetch: a silent fallback to the network
    would work on the box and fail on the cluster, i.e. it would break exactly where it
    matters and pass exactly where it does not.
    """

    raw = os.environ.get(_STAGE_DIR_ENV, "").strip()
    if not raw:
        raise RuntimeError(
            f"{_STAGE_DIR_ENV} is unset. SCROLLS rows are read from locally "
            "staged jsonl (the cluster has no outbound SSL and `datasets`>=3 refuses "
            "the script-based tau/scrolls loader). Stage with "
            "`stage_scrolls.py <dir>` under a `datasets<3` venv, then point "
            f"{_STAGE_DIR_ENV} at that directory."
        )
    path = Path(raw)
    if not path.is_dir():
        raise RuntimeError(f"{_STAGE_DIR_ENV}={raw} is not a directory")
    return path


def manifest() -> dict[str, Any]:
    """Return the staging manifest (source, split, per-task rows and sha256)."""

    path = stage_dir() / "MANIFEST.json"
    if not path.is_file():
        raise RuntimeError(
            f"staged SCROLLS directory {stage_dir()} has no MANIFEST.json -- "
            "re-stage; the manifest carries the row counts and sha256 that "
            "bind a frozen suite to its input data"
        )
    return json.loads(path.read_text())


def _task_suffix(dataset: str) -> str:
    if dataset not in GENLEN:
        raise ValueError(f"unknown SCROLLS dataset {dataset!r}")
    return dataset[len("scrolls_"):]


def iter_records(dataset: str) -> Iterator[dict[str, Any]]:
    """Yield staged rows in file order (the staging order is the seal)."""

    path = stage_dir() / f"{_task_suffix(dataset)}.validation.jsonl"
    if not path.is_file():
        raise RuntimeError(
            f"staged file {path} is missing; MANIFEST.json lists "
            f"{sorted(manifest().get('tasks', {}))}"
        )
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def render_prompt(dataset: str, record: dict[str, Any]) -> str:
    return _PROMPT[dataset].format(context=record["context"])


def fits_window(
    prompt_tokens: int, dataset: str, context_length: int
) -> bool:
    """Whether this row can run WITHOUT truncation.

    The generated tokens share the window with the prompt, so the test is
    ``prompt + output <= context_length``. Rows that fail are dropped, never
    truncated (ruling A1).

    ``prompt_tokens`` MUST be the length the suite will record, i.e. the
    chat-templated length with the generation prompt -- not the bare text.
    Passing the text length lets rows through whose templated form no longer
    leaves room for the reference output, which then silently lose quality
    eligibility. ``load_scrolls_summary`` gets this right by calling the very
    function that records ``prompt_len``.
    """

    return prompt_tokens + GENLEN[dataset] <= context_length
