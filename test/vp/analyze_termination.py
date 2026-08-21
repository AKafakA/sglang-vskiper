#!/usr/bin/env python3
"""Classify natural-generation repetition and termination per request."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from labeled_workload import read_jsonl, score_prediction
from validate_qps_artifact import read_last_record


def _normalized_lines(text: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", line).strip().lower()
        for line in text.splitlines()
        if line.strip()
    ]


def _line_loop(text: str) -> dict[str, Any]:
    lines = _normalized_lines(text)
    if not lines:
        return {
            "detected": False,
            "line": None,
            "count": 0,
            "character_fraction": 0.0,
        }
    counts = Counter(lines)
    line, count = max(counts.items(), key=lambda item: (item[1], len(item[0])))
    total_characters = sum(len(value) for value in lines)
    covered = count * len(line)
    fraction = covered / total_characters if total_characters else 0.0
    return {
        "detected": count >= 4 and fraction >= 0.5,
        "line": line if count >= 4 else None,
        "count": count,
        "character_fraction": fraction,
    }


def _answer_marker_loop(text: str) -> dict[str, Any]:
    markers = [
        match.strip().lower()
        for match in re.findall(r"####\s*([^\n\r]+)", text)
        if match.strip()
    ]
    if not markers:
        return {"detected": False, "answer": None, "count": 0}
    counts = Counter(markers)
    answer, count = max(counts.items(), key=lambda item: item[1])
    return {
        "detected": count >= 3,
        "answer": answer if count >= 3 else None,
        "count": count,
    }


def _periodic_suffix(output_ids: list[int]) -> dict[str, Any]:
    tail = output_ids[-256:]
    best_period = None
    best_covered = 0
    for period in range(8, min(64, len(tail) // 3) + 1):
        matched = 0
        index = len(tail) - 1
        while index - period >= 0 and tail[index] == tail[index - period]:
            matched += 1
            index -= 1
        covered = matched + period if matched else 0
        if covered >= 3 * period and covered > best_covered:
            best_period = period
            best_covered = covered
    fraction = best_covered / len(tail) if tail else 0.0
    detected = best_period is not None and fraction >= 0.5
    return {
        "detected": detected,
        "period_tokens": best_period if detected else None,
        "covered_tokens": best_covered,
        "tail_fraction": fraction,
        "suffix_start_token": len(output_ids) - best_covered if detected else None,
    }


def classify_repetition(text: str, output_ids: list[int]) -> dict[str, Any]:
    line = _line_loop(text)
    answer = _answer_marker_loop(text)
    periodic = _periodic_suffix(output_ids)
    return {
        "loop_any": bool(
            line["detected"] or answer["detected"] or periodic["detected"]
        ),
        "line_loop": line,
        "answer_marker_loop": answer,
        "periodic_token_loop": periodic,
    }


def analyze_record(
    record: dict[str, Any],
    metadata: list[dict[str, Any]],
    *,
    allow_code_execution: bool = False,
) -> dict[str, Any]:
    request_ids = record.get("request_ids") or []
    texts = record.get("generated_texts") or []
    output_ids = record.get("raw_output_ids") or []
    finish_reasons = record.get("finish_reasons") or []
    successes = record.get("successes") or []
    arrays = (request_ids, texts, output_ids, finish_reasons, successes)
    if not all(isinstance(value, list) for value in arrays):
        raise ValueError("benchmark record lacks termination arrays")
    if len({len(value) for value in arrays}) != 1:
        raise ValueError("benchmark termination arrays do not align")
    metadata_by_id = {str(row["request_id"]): row for row in metadata}
    if len(metadata_by_id) != len(metadata):
        raise ValueError("metadata contains duplicate request IDs")

    rows = []
    aggregate: dict[str, Counter] = defaultdict(Counter)
    quality: dict[str, list[float]] = defaultdict(list)
    for index, request_id_value in enumerate(request_ids):
        request_id = str(request_id_value)
        if request_id not in metadata_by_id:
            raise ValueError(f"unknown request ID {request_id}")
        row = metadata_by_id[request_id]
        ids = output_ids[index]
        if not isinstance(ids, list) or any(not isinstance(value, int) for value in ids):
            raise ValueError(f"{request_id} has invalid raw output IDs")
        text = str(texts[index] or "")
        repetition = classify_repetition(text, ids)
        score, score_status = (
            score_prediction(row, text, allow_code_execution)
            if successes[index]
            else (None, "request_error")
        )
        dataset = str(row["dataset"])
        finish_reason = finish_reasons[index]
        context_hit = (
            finish_reason == "length"
            and row.get("output_policy") == "remaining_model_context"
        )
        no_answer = bool(
            dataset == "gsm8k" and re.search(r"####\s*[^\n\r]+", text) is None
        )
        result = {
            "index": index,
            "request_id": request_id,
            "source_request_id": row.get("source_request_id"),
            "dataset": dataset,
            "success": bool(successes[index]),
            "finish_reason": finish_reason,
            "context_limit_hit": context_hit,
            "output_tokens_raw": len(ids),
            "quality_score": score,
            "quality_status": score_status,
            "no_answer": no_answer,
            **repetition,
        }
        rows.append(result)
        counters = aggregate[dataset]
        counters["requests"] += 1
        counters["successes"] += int(bool(successes[index]))
        counters["loops"] += int(repetition["loop_any"])
        counters["line_loops"] += int(repetition["line_loop"]["detected"])
        counters["answer_marker_loops"] += int(
            repetition["answer_marker_loop"]["detected"]
        )
        counters["periodic_token_loops"] += int(
            repetition["periodic_token_loop"]["detected"]
        )
        counters["context_limit_hits"] += int(context_hit)
        counters["no_answer"] += int(no_answer)
        if score is not None:
            quality[dataset].append(float(score))

    summary = {}
    for dataset, counters in aggregate.items():
        requests = counters["requests"]
        scores = quality[dataset]
        summary[dataset] = {
            **dict(counters),
            "loop_rate": counters["loops"] / requests if requests else 0.0,
            "context_limit_hit_rate": (
                counters["context_limit_hits"] / requests if requests else 0.0
            ),
            "no_answer_rate": (
                counters["no_answer"] / requests if requests else 0.0
            ),
            "scored_requests": len(scores),
            "mean_quality": sum(scores) / len(scores) if scores else None,
        }
    return {
        "schema_version": 1,
        "requests": len(rows),
        "rows": rows,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-jsonl", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-code-execution", action="store_true")
    args = parser.parse_args()
    result = analyze_record(
        read_last_record(args.bench_jsonl),
        read_jsonl(args.metadata),
        allow_code_execution=args.allow_code_execution,
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
