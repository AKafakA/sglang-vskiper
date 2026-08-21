"""AdaSkip calibration profile — load and validate the banked skip profile.

Hash-gated: the profile's own parquet hash anchors the calibration, so a
deployment cannot silently run against a different profile than the one its
evidence was produced with. A missing or mismatched artifact is a configuration
error, never a heuristic fallback.
"""

from __future__ import annotations

import math
import hashlib
import json
import os
from pathlib import Path
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Mapping, Optional
from sglang.srt.vpipe.skipper import (
    ADASKIP_OFFICIAL_SOURCE_REVISION,
    ADASKIP_OFFICIAL_WINDOW,
    _cosine_float,
    _nonempty_string,
    _positive_float,
    _positive_int,
    _require_keys,
)


ADASKIP_CALIBRATION_SCHEMA = "vpipe-adaskip-calibration-v1"
@dataclass(frozen=True, slots=True)
class AdaSkipCalibration:
    model_id: str
    model_revision: str
    source_revision: str
    dataset_id: str
    dataset_revision: str
    request_count: int
    num_hidden_layers: int
    attention_similarity: tuple[float, ...]
    mlp_similarity: tuple[float, ...]
    attention_scale: tuple[float, ...]
    mlp_scale: tuple[float, ...]
    input_sha256: str

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        input_sha256: str,
    ) -> "AdaSkipCalibration":
        _require_keys(
            value,
            required=frozenset(
                (
                    "schema",
                    "model",
                    "source",
                    "calibration",
                    "measurements",
                )
            ),
            context="AdaSkip calibration",
        )
        if value["schema"] != ADASKIP_CALIBRATION_SCHEMA:
            raise ValueError(
                f"AdaSkip calibration schema must be {ADASKIP_CALIBRATION_SCHEMA}"
            )
        model = value["model"]
        source = value["source"]
        calibration = value["calibration"]
        measurements = value["measurements"]
        for name, item in (
            ("model", model),
            ("source", source),
            ("calibration", calibration),
            ("measurements", measurements),
        ):
            if not isinstance(item, dict):
                raise ValueError(f"AdaSkip {name} must be an object")
        _require_keys(
            model,
            required=frozenset(("id", "revision", "num_hidden_layers")),
            context="AdaSkip model",
        )
        _require_keys(
            source,
            required=frozenset(("repository", "revision")),
            context="AdaSkip source",
        )
        if source["repository"] != "https://github.com/ASISys/AdaSkip":
            raise ValueError("AdaSkip source repository is not the official repository")
        source_revision = _nonempty_string(
            source["revision"], name="AdaSkip source revision"
        )
        if source_revision != ADASKIP_OFFICIAL_SOURCE_REVISION:
            raise ValueError(
                "AdaSkip source revision does not match the audited official source"
            )
        _require_keys(
            calibration,
            required=frozenset(
                ("dataset_id", "dataset_revision", "request_count")
            ),
            context="AdaSkip calibration provenance",
        )
        _require_keys(
            measurements,
            required=frozenset(
                (
                    "attention_similarity",
                    "mlp_similarity",
                    "attention_scale",
                    "mlp_scale",
                )
            ),
            context="AdaSkip measurements",
        )
        num_layers = _positive_int(
            model["num_hidden_layers"], name="AdaSkip num_hidden_layers"
        )

        def vector(key: str, converter: Any) -> tuple[float, ...]:
            raw = measurements[key]
            if not isinstance(raw, list) or len(raw) != num_layers:
                raise ValueError(
                    f"AdaSkip {key} must contain exactly {num_layers} values"
                )
            return tuple(
                converter(item, name=f"AdaSkip {key}[{index}]")
                for index, item in enumerate(raw)
            )

        request_count = _positive_int(
            calibration["request_count"], name="AdaSkip calibration request_count"
        )
        if request_count != ADASKIP_OFFICIAL_WINDOW:
            raise ValueError(
                "audited AdaSkip fixed-profile calibration requires exactly "
                f"{ADASKIP_OFFICIAL_WINDOW} requests"
            )
        return cls(
            model_id=_nonempty_string(model["id"], name="AdaSkip model id"),
            model_revision=_nonempty_string(
                model["revision"], name="AdaSkip model revision"
            ),
            source_revision=source_revision,
            dataset_id=_nonempty_string(
                calibration["dataset_id"], name="AdaSkip calibration dataset id"
            ),
            dataset_revision=_nonempty_string(
                calibration["dataset_revision"],
                name="AdaSkip calibration dataset revision",
            ),
            request_count=request_count,
            num_hidden_layers=num_layers,
            attention_similarity=vector("attention_similarity", _cosine_float),
            mlp_similarity=vector("mlp_similarity", _cosine_float),
            attention_scale=vector("attention_scale", _positive_float),
            mlp_scale=vector("mlp_scale", _positive_float),
            input_sha256=input_sha256,
        )
