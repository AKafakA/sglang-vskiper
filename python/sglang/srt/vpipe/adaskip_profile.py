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
from pathlib import Path
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Mapping, Optional


ADASKIP_OFFICIAL_SOURCE_REVISION = (
    "4ef686d6fd7b94756e4959f3f80191708dcec9e3"
)

ADASKIP_OFFICIAL_WINDOW = 20

ADASKIP_PROFILE_SCHEMA = "vpipe-adaskip-fixed-sublayer-profile-v2"

ADASKIP_OFFICIAL_EXTRA_MLP_MINIMUM = 0

def _require_keys(
    value: Mapping[str, Any],
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    context: str,
) -> None:
    keys = frozenset(value)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing:
        raise ValueError(f"{context} is missing: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{context} has unknown fields: {', '.join(unknown)}")

def _nonempty_string(value: Any, *, name: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{name} must be a nonempty string")
    return result

def _positive_float(value: Any, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result

def _cosine_float(value: Any, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if not -1.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [-1, 1]")
    return result

def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if result <= 0 or result != value:
        raise ValueError(f"{name} must be a positive integer")
    return result

def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

def _load_json_bytes(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read AdaSkip JSON {path}: {error}") from error
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid AdaSkip JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("AdaSkip JSON root must be an object")
    return value, payload

@dataclass(frozen=True, slots=True)
class AdaSkipLayerProfile:
    layer_id: int
    attention_similarity: float
    mlp_similarity: float
    attention_scale: float
    mlp_scale: float
    skip_attention: bool
    skip_mlp: bool

    @property
    def is_routed(self) -> bool:
        return self.skip_attention or self.skip_mlp

@dataclass(frozen=True, slots=True)
class AdaSkipProfile:
    model_id: str
    model_revision: str
    source_revision: str
    calibration_dataset_id: str
    calibration_dataset_revision: str
    calibration_request_count: int
    calibration_sha256: str
    skip_sublayer_count: int
    online_decode_window: int
    online_decode_extra_mlp: bool
    online_extra_mlp_minimum: int
    layers: tuple[AdaSkipLayerProfile, ...]
    input_sha256: str

    @property
    def num_hidden_layers(self) -> int:
        return len(self.layers)

    @property
    def routed_layer_ids(self) -> tuple[int, ...]:
        return tuple(layer.layer_id for layer in self.layers if layer.is_routed)

    def layer(self, layer_id: int) -> AdaSkipLayerProfile:
        if layer_id < 0 or layer_id >= len(self.layers):
            raise ValueError(f"AdaSkip layer {layer_id} is outside the profile")
        result = self.layers[layer_id]
        if result.layer_id != layer_id:
            raise RuntimeError("AdaSkip profile layer ordering is corrupt")
        return result

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        input_sha256: str,
    ) -> "AdaSkipProfile":
        _require_keys(
            value,
            required=frozenset(
                (
                    "schema",
                    "model",
                    "source",
                    "calibration",
                    "policy",
                    "layers",
                )
            ),
            context="AdaSkip profile",
        )
        if value["schema"] != ADASKIP_PROFILE_SCHEMA:
            raise ValueError(
                f"AdaSkip profile schema must be {ADASKIP_PROFILE_SCHEMA}"
            )
        model = value["model"]
        source = value["source"]
        calibration = value["calibration"]
        policy = value["policy"]
        layers = value["layers"]
        for name, item in (
            ("model", model),
            ("source", source),
            ("calibration", calibration),
            ("policy", policy),
        ):
            if not isinstance(item, dict):
                raise ValueError(f"AdaSkip profile {name} must be an object")
        if not isinstance(layers, list) or not layers:
            raise ValueError("AdaSkip profile layers must be a nonempty list")
        _require_keys(
            model,
            required=frozenset(("id", "revision", "num_hidden_layers")),
            context="AdaSkip profile model",
        )
        num_layers = _positive_int(
            model["num_hidden_layers"], name="AdaSkip profile num_hidden_layers"
        )
        if len(layers) != num_layers:
            raise ValueError(
                f"AdaSkip profile requires {num_layers} ordered layer records"
            )
        _require_keys(
            source,
            required=frozenset(("repository", "revision")),
            context="AdaSkip profile source",
        )
        if source["repository"] != "https://github.com/ASISys/AdaSkip":
            raise ValueError("AdaSkip profile source is not the official repository")
        source_revision = _nonempty_string(
            source["revision"], name="AdaSkip profile source revision"
        )
        if source_revision != ADASKIP_OFFICIAL_SOURCE_REVISION:
            raise ValueError(
                "AdaSkip profile source revision does not match the audited source"
            )
        _require_keys(
            calibration,
            required=frozenset(
                (
                    "dataset_id",
                    "dataset_revision",
                    "request_count",
                    "sha256",
                )
            ),
            context="AdaSkip profile calibration",
        )
        request_count = _positive_int(
            calibration["request_count"],
            name="AdaSkip profile calibration request_count",
        )
        if request_count != ADASKIP_OFFICIAL_WINDOW:
            raise ValueError(
                "AdaSkip profile requires the audited 20-request calibration"
            )
        calibration_sha256 = _nonempty_string(
            calibration["sha256"], name="AdaSkip calibration sha256"
        )
        if len(calibration_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in calibration_sha256
        ):
            raise ValueError("AdaSkip calibration sha256 must be lowercase hex")
        _require_keys(
            policy,
            required=frozenset(
                (
                    "skip_sublayer_count",
                    "selection",
                    "compensation",
                    "online_decode_window",
                    "online_decode_extra_mlp",
                    "online_extra_mlp_minimum",
                    "native_kv",
                    "vp_kv",
                )
            ),
            context="AdaSkip profile policy",
        )
        skip_count = _positive_int(
            policy["skip_sublayer_count"],
            name="AdaSkip skip_sublayer_count",
        )
        if skip_count > 2 * num_layers:
            raise ValueError("AdaSkip skip_sublayer_count exceeds model sublayers")
        if policy["selection"] != "topk_concat_attention_then_mlp_similarity":
            raise ValueError("AdaSkip profile selection semantics changed")
        if policy["compensation"] != "mean_output_norm_over_input_norm":
            raise ValueError("AdaSkip profile compensation semantics changed")
        online_window = _positive_int(
            policy["online_decode_window"],
            name="AdaSkip online_decode_window",
        )
        if online_window != ADASKIP_OFFICIAL_WINDOW:
            raise ValueError("AdaSkip online decode window must be 20 tokens")
        online_extra = policy["online_decode_extra_mlp"]
        if not isinstance(online_extra, bool):
            raise ValueError("AdaSkip online_decode_extra_mlp must be boolean")
        online_extra_minimum = policy["online_extra_mlp_minimum"]
        if (
            isinstance(online_extra_minimum, bool)
            or not isinstance(online_extra_minimum, int)
            or online_extra_minimum != ADASKIP_OFFICIAL_EXTRA_MLP_MINIMUM
        ):
            raise ValueError(
                "AdaSkip online extra MLP minimum must match released value 0"
            )
        if policy["native_kv"] != "omit_on_skip_attention":
            raise ValueError("AdaSkip native K/V semantics changed")
        if policy["vp_kv"] != "materialize_own_layer_projection":
            raise ValueError("AdaSkip VP K/V semantics changed")

        parsed_layers: list[AdaSkipLayerProfile] = []
        for expected_layer, item in enumerate(layers):
            if not isinstance(item, dict):
                raise ValueError("AdaSkip layer records must be objects")
            _require_keys(
                item,
                required=frozenset(
                    (
                        "layer_id",
                        "attention_similarity",
                        "mlp_similarity",
                        "attention_scale",
                        "mlp_scale",
                        "skip_attention",
                        "skip_mlp",
                    )
                ),
                context=f"AdaSkip layer {expected_layer}",
            )
            if item["layer_id"] != expected_layer:
                raise ValueError("AdaSkip layer records must be ordered and complete")
            if not isinstance(item["skip_attention"], bool) or not isinstance(
                item["skip_mlp"], bool
            ):
                raise ValueError("AdaSkip skip actions must be boolean")
            parsed_layers.append(
                AdaSkipLayerProfile(
                    layer_id=expected_layer,
                    attention_similarity=_cosine_float(
                        item["attention_similarity"],
                        name=f"AdaSkip attention_similarity[{expected_layer}]",
                    ),
                    mlp_similarity=_cosine_float(
                        item["mlp_similarity"],
                        name=f"AdaSkip mlp_similarity[{expected_layer}]",
                    ),
                    attention_scale=_positive_float(
                        item["attention_scale"],
                        name=f"AdaSkip attention_scale[{expected_layer}]",
                    ),
                    mlp_scale=_positive_float(
                        item["mlp_scale"],
                        name=f"AdaSkip mlp_scale[{expected_layer}]",
                    ),
                    skip_attention=item["skip_attention"],
                    skip_mlp=item["skip_mlp"],
                )
            )

        observed = sum(
            int(layer.skip_attention) + int(layer.skip_mlp)
            for layer in parsed_layers
        )
        if observed != skip_count:
            raise ValueError(
                "AdaSkip selected sublayer count does not match the policy"
            )
        ranked = sorted(
            range(2 * num_layers),
            key=lambda index: (
                -(
                    parsed_layers[index].attention_similarity
                    if index < num_layers
                    else parsed_layers[index - num_layers].mlp_similarity
                ),
                index,
            ),
        )
        expected = frozenset(ranked[:skip_count])
        selected = frozenset(
            (
                layer.layer_id
                if layer.skip_attention
                else -1
                for layer in parsed_layers
            )
        ) | frozenset(
            (
                num_layers + layer.layer_id
                if layer.skip_mlp
                else -1
                for layer in parsed_layers
            )
        )
        selected = selected - {-1}
        if selected != expected:
            raise ValueError(
                "AdaSkip selected sublayers do not match concatenated top-k similarity"
            )
        return cls(
            model_id=_nonempty_string(model["id"], name="AdaSkip profile model id"),
            model_revision=_nonempty_string(
                model["revision"], name="AdaSkip profile model revision"
            ),
            source_revision=source_revision,
            calibration_dataset_id=_nonempty_string(
                calibration["dataset_id"],
                name="AdaSkip profile calibration dataset id",
            ),
            calibration_dataset_revision=_nonempty_string(
                calibration["dataset_revision"],
                name="AdaSkip profile calibration dataset revision",
            ),
            calibration_request_count=request_count,
            calibration_sha256=calibration_sha256,
            skip_sublayer_count=skip_count,
            online_decode_window=online_window,
            online_decode_extra_mlp=online_extra,
            online_extra_mlp_minimum=online_extra_minimum,
            layers=tuple(parsed_layers),
            input_sha256=input_sha256,
        )

@lru_cache(maxsize=None)
def load_adaskip_profile(path: str | Path) -> AdaSkipProfile:
    resolved = Path(path).expanduser().resolve()
    value, payload = _load_json_bytes(resolved)
    return AdaSkipProfile.from_mapping(
        value,
        input_sha256=_sha256_bytes(payload),
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


def _finite_float(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result

def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result
