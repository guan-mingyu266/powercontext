# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rerank Memory candidates through an offline HTTP scoring service.

The service caps how many candidates one request may carry, so the pool is split
into batches. Every batch keeps the global one-based rank of each candidate, and
the final selection is made once over the whole pool rather than per batch.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import httpx
from pydantic import BaseModel, ConfigDict

from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision, _validate_bounds
from powercontext.builtin.inference import InferenceUsage

if TYPE_CHECKING:
    from powercontext.builtin.artifacts.memory.models import MemoryHit

__all__ = ["HTTPMemoryReranker"]


class _ScoredCandidate(BaseModel):
    """One score reported for the global rank it was sent with."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int
    score: float


class _ScoreResponse(BaseModel):
    """The exact response body the scoring service may return."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scores: tuple[_ScoredCandidate, ...]


class HTTPMemoryReranker:
    """Select Memory hits from one coarse pool using service-provided scores."""

    policy_id: str

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = "http-cross-encoder-v1",
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least one")  # noqa: TRY003
        self.policy_id = policy_id
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Score every candidate exactly once, then select over the whole pool."""

        _validate_bounds(len(candidates), limit)

        scores: dict[int, float] = {}
        requests = 0
        for start in range(0, len(candidates), self._batch_size):
            batch = tuple(range(start + 1, min(start + self._batch_size, len(candidates)) + 1))
            response = await self._client.post(
                self._endpoint,
                json={
                    "query": query,
                    "candidates": [{"rank": rank, "text": candidates[rank - 1].text} for rank in batch],
                },
            )
            requests += 1
            response.raise_for_status()
            scores.update(_parse_scores(response, batch))

        ordered = sorted(scores, key=lambda rank: (-scores[rank], rank))
        return MemoryRerankDecision(
            selected_ranks=tuple(ordered[:limit]),
            usage=InferenceUsage(requests=requests),
        )


def _parse_scores(response: httpx.Response, batch: tuple[int, ...]) -> dict[int, float]:
    """Reject any response that does not cover exactly this batch once."""

    try:
        payload = _ScoreResponse.model_validate(response.json())
    except ValueError as error:
        raise ValueError(f"reranker returned an invalid response: {error}") from error  # noqa: TRY003

    expected = set(batch)
    observed: dict[int, float] = {}
    for item in payload.scores:
        if item.rank not in expected:
            raise ValueError(f"reranker scored rank {item.rank} outside its own batch")  # noqa: TRY003
        if item.rank in observed:
            raise ValueError(f"reranker scored rank {item.rank} more than once")  # noqa: TRY003
        if not math.isfinite(item.score):
            raise ValueError(f"reranker score for rank {item.rank} must be finite")  # noqa: TRY003
        observed[item.rank] = item.score

    if len(observed) != len(expected):
        missing = sorted(expected - set(observed))
        raise ValueError(f"reranker did not score ranks {missing}")  # noqa: TRY003
    return observed
