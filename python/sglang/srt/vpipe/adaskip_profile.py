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
    ADASKIP_OFFICIAL_EXTRA_MLP_MINIMUM,
    ADASKIP_OFFICIAL_SOURCE_REVISION,
    ADASKIP_OFFICIAL_WINDOW,
    ADASKIP_PROFILE_SCHEMA,
    AdaSkipProfile,
    _cosine_float,
    _load_json_bytes,
    _nonempty_string,
    _positive_float,
    _positive_int,
    _require_keys,
    _sha256_bytes,
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


def build_adaskip_profile_mapping(
    calibration: AdaSkipCalibration,
    *,
    skip_sublayer_count: int,
    online_decode_extra_mlp: bool = False,
) -> dict[str, Any]:
    if isinstance(skip_sublayer_count, bool) or not isinstance(
        skip_sublayer_count, int
    ):
        raise ValueError("AdaSkip skip_sublayer_count must be an integer")
    if not 1 <= skip_sublayer_count <= 2 * calibration.num_hidden_layers:
        raise ValueError("AdaSkip skip_sublayer_count is outside model sublayers")
    if not isinstance(online_decode_extra_mlp, bool):
        raise ValueError("AdaSkip online_decode_extra_mlp must be boolean")
    combined = calibration.attention_similarity + calibration.mlp_similarity
    selected = frozenset(
        sorted(range(len(combined)), key=lambda index: (-combined[index], index))[
            :skip_sublayer_count
        ]
    )
    layers = []
    for layer_id in range(calibration.num_hidden_layers):
        layers.append(
            {
                "layer_id": layer_id,
                "attention_similarity": calibration.attention_similarity[layer_id],
                "mlp_similarity": calibration.mlp_similarity[layer_id],
                "attention_scale": calibration.attention_scale[layer_id],
                "mlp_scale": calibration.mlp_scale[layer_id],
                "skip_attention": layer_id in selected,
                "skip_mlp": calibration.num_hidden_layers + layer_id in selected,
            }
        )
    return {
        "schema": ADASKIP_PROFILE_SCHEMA,
        "model": {
            "id": calibration.model_id,
            "revision": calibration.model_revision,
            "num_hidden_layers": calibration.num_hidden_layers,
        },
        "source": {
            "repository": "https://github.com/ASISys/AdaSkip",
            "revision": calibration.source_revision,
        },
        "calibration": {
            "dataset_id": calibration.dataset_id,
            "dataset_revision": calibration.dataset_revision,
            "request_count": calibration.request_count,
            "sha256": calibration.input_sha256,
        },
        "policy": {
            "skip_sublayer_count": skip_sublayer_count,
            "selection": "topk_concat_attention_then_mlp_similarity",
            "compensation": "mean_output_norm_over_input_norm",
            "online_decode_window": ADASKIP_OFFICIAL_WINDOW,
            "online_decode_extra_mlp": online_decode_extra_mlp,
            "online_extra_mlp_minimum": ADASKIP_OFFICIAL_EXTRA_MLP_MINIMUM,
            "native_kv": "omit_on_skip_attention",
            "vp_kv": "materialize_own_layer_projection",
        },
        "layers": layers,
    }

def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")

def load_adaskip_calibration(path: str | Path) -> AdaSkipCalibration:
    resolved = Path(path).expanduser().resolve()
    value, payload = _load_json_bytes(resolved)
    return AdaSkipCalibration.from_mapping(
        value,
        input_sha256=_sha256_bytes(payload),
    )

def profile_from_calibration(
    calibration: AdaSkipCalibration,
    *,
    skip_sublayer_count: int,
    online_decode_extra_mlp: bool = False,
) -> tuple[AdaSkipProfile, bytes]:
    mapping = build_adaskip_profile_mapping(
        calibration,
        skip_sublayer_count=skip_sublayer_count,
        online_decode_extra_mlp=online_decode_extra_mlp,
    )
    payload = canonical_json_bytes(mapping)
    return (
        AdaSkipProfile.from_mapping(
            mapping,
            input_sha256=_sha256_bytes(payload),
        ),
        payload,
    )

def selected_sublayer_ids(profile: AdaSkipProfile) -> tuple[str, ...]:
    result: list[str] = []
    for layer in profile.layers:
        if layer.skip_attention:
            result.append(f"attention:{layer.layer_id}")
        if layer.skip_mlp:
            result.append(f"mlp:{layer.layer_id}")
    return tuple(result)
