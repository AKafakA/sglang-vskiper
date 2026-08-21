"""AdaSkip — sublayer attention/MLP skipping, as a pluggable policy.

Separate module because it is the largest adapter and because its
action-semantics declaration is ⛔ owner-gated for PROMOTION (F7 evidence is
banked: sublayer skip 0.5000 exact at 42.77M rows, both phases).

The contract that must not be violated: AdaSkip's independent attention and MLP
actions are NEVER lowered into the binary RUN/PROJECT executor. An unsupported
action fails closed. Same-layers-for-every-token would be pruning, not the
dynamic skipping this system is about.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from functools import lru_cache
from math import ceil
from typing import Any, Mapping, Optional
import torch
from sglang.srt.vpipe.env import (
    ADASKIP_MAX_GRAPH_ROWS_ENV,
    ADASKIP_MAX_REQUEST_SLOTS_ENV,
)
from sglang.srt.vpipe.env import (
    SUBLAYER_EXECUTION,
)
from sglang.srt.vpipe.types import (
    FullGraphActionBatch,
    FullGraphSkipperAdapter,
    LogicalAction,
)
from sglang.srt.vpipe.types import (
    FULL_GRAPH_ACTION_CONTRACT,
    _LOGICAL_ACTION_CODES,
)


