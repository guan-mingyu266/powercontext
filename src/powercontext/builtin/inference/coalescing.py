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

"""Share embedding work across concurrent callers and cache successful vectors.

One wrapper coordinates every caller in the same event loop: identical texts wait
on the same in-flight future, the queue is drained in bounded batches, and only a
successful batch is written to the bounded LRU cache. Each caller still reports
the vectors in its own input order and only the provider usage it actually waited
for.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from powercontext.builtin.inference.models import EmbeddingResult, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel

if TYPE_CHECKING:
    from powercontext.builtin.artifacts.memory.models import EmbeddingProfile
    from powercontext.builtin.inference.models import EmbeddingVector

__all__ = ["CoalescingEmbeddingModel"]


@dataclass(frozen=True, slots=True)
class _Resolved:
    """One vector plus the provider call that produced it, if any."""

    vector: EmbeddingVector
    usage: InferenceUsage | None


class CoalescingEmbeddingModel:
    """Coalesce concurrent embeddings and reuse successful vectors."""

    profile: EmbeddingProfile

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least one")  # noqa: TRY003
        if coalesce_ms < 0:
            raise ValueError("coalesce_ms must not be negative")  # noqa: TRY003
        if cache_size < 0:
            raise ValueError("cache_size must not be negative")  # noqa: TRY003

        self.profile = model.profile
        self._model = model
        self._max_batch_size = max_batch_size
        self._coalesce_ms = coalesce_ms
        self._cache_size = cache_size
        self._cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()
        # Texts that are waiting for a vector, whether queued or already dispatched.
        self._inflight: dict[str, asyncio.Future[_Resolved]] = {}
        # Texts that no batch has picked up yet.
        self._queued: list[str] = []
        self._flusher: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        """Return one ordered vector per input text."""

        if not texts:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))

        futures = [self._acquire(text) for text in texts]
        resolved = await asyncio.gather(*(asyncio.shield(future) for future in futures))

        batches: dict[int, InferenceUsage] = {}
        for item in resolved:
            if item.usage is not None:
                batches[id(item.usage)] = item.usage
        usage = InferenceUsage(requests=0)
        for batch_usage in batches.values():
            usage = _merge_usage(usage, batch_usage)

        return EmbeddingResult(vectors=tuple(item.vector for item in resolved), usage=usage)

    def _acquire(self, text: str) -> asyncio.Future[_Resolved]:
        cached = self._cache.get(text)
        if cached is not None:
            self._cache.move_to_end(text)
            done: asyncio.Future[_Resolved] = asyncio.get_running_loop().create_future()
            done.set_result(_Resolved(vector=cached, usage=None))
            return done

        existing = self._inflight.get(text)
        if existing is not None:
            return existing

        future: asyncio.Future[_Resolved] = asyncio.get_running_loop().create_future()
        self._inflight[text] = future
        self._queued.append(text)
        self._ensure_flusher()
        return future

    def _ensure_flusher(self) -> None:
        if self._flusher is None or self._flusher.done():
            self._flusher = asyncio.create_task(self._flush())

    async def _flush(self) -> None:
        """Drain queued texts into bounded batches until nothing is waiting."""

        try:
            while self._queued:
                if self._coalesce_ms > 0:
                    await asyncio.sleep(self._coalesce_ms / 1_000)
                else:
                    # Yield once so every caller in this tick registers first.
                    await asyncio.sleep(0)
                pending = self._queued
                self._queued = []
                for start in range(0, len(pending), self._max_batch_size):
                    batch = [(text, self._inflight[text]) for text in pending[start : start + self._max_batch_size]]
                    task = asyncio.create_task(self._dispatch(batch))
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)
        finally:
            self._flusher = None

    async def _dispatch(self, batch: list[tuple[str, asyncio.Future[_Resolved]]]) -> None:
        texts = tuple(text for text, _ in batch)
        try:
            result = await self._model.embed(texts)
        except BaseException as error:
            self._fail(batch, error)
            return

        if len(result.vectors) != len(texts):
            self._fail(batch, ValueError("provider returned a different number of vectors than texts"))
            return

        for (text, future), vector in zip(batch, result.vectors, strict=True):
            self._remember(text, vector)
            self._inflight.pop(text, None)
            if not future.done():
                future.set_result(_Resolved(vector=vector, usage=result.usage))

    def _fail(self, batch: list[tuple[str, asyncio.Future[_Resolved]]], error: BaseException) -> None:
        for text, future in batch:
            self._inflight.pop(text, None)
            if not future.done():
                future.set_exception(error)

    def _remember(self, text: str, vector: tuple[float, ...]) -> None:
        if self._cache_size == 0:
            return
        self._cache[text] = vector
        self._cache.move_to_end(text)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)


def _merge_usage(total: InferenceUsage, batch: InferenceUsage) -> InferenceUsage:
    return InferenceUsage(
        requests=total.requests + batch.requests,
        input_tokens=_add_optional(total.input_tokens, batch.input_tokens),
        output_tokens=_add_optional(total.output_tokens, batch.output_tokens),
    )


def _add_optional(left: int | None, right: int | None) -> int | None:
    if left is None and right is None:
        return None
    return (left or 0) + (right or 0)
