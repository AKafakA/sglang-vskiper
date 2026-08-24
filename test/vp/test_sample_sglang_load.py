from __future__ import annotations

import json
from io import BytesIO

import sample_sglang_load


class _Response:
    def __init__(self, payload: object):
        self._body = BytesIO(json.dumps(payload).encode())

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        pass

    def read(self, *args: object) -> bytes:
        return self._body.read(*args)


def test_sampler_records_running_and_waiting_requests(monkeypatch) -> None:
    payload = {
        "loads": [
            {
                "num_running_reqs": 7,
                "num_waiting_reqs": 4,
                "num_total_tokens": 1234,
                "num_used_tokens": 913,
            }
        ]
    }
    monkeypatch.setattr(
        sample_sglang_load.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(payload),
    )
    sample = sample_sglang_load.fetch_sample("http://unused/v1/loads", 1.0)
    assert sample["running_requests"] == 7
    assert sample["waiting_requests"] == 4
    assert sample["resident_tokens"] == 1234
    assert sample["pending_tokens"] == 321
    summary = sample_sglang_load.summarize([sample])
    assert summary["max_running_requests"] == 7
    assert summary["max_waiting_requests"] == 4
