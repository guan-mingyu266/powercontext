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

"""Propagate Source lifecycle state to the Memory and Topic Memory that depend on it.

Each Scope holds an explicit dependency graph: Sources are the roots, Memory entries and
Topic Memory artifacts are derived nodes with an ``all`` or ``any`` support policy. A
withdrawal event deactivates the dependent entries natively and a restore event brings
back the ones that still have support, while audit reads keep pointing at the exact
registered references.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Annotated, Any

import rfc8785
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import insert, select, update

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.builtin.persistence.database import AsyncDatabase
from powercontext.builtin.persistence.tables import SOURCE_LIFECYCLE_NODES_TABLE
from powercontext.builtin.runtime.models import GetMemoryEntryRequest
from powercontext.builtin.sources import validate_scope_id
from powercontext.builtin.statistics import ModelUsagePurpose
from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import SourceRef

if TYPE_CHECKING:
    from powercontext.builtin.artifacts.topic_memory import TopicMemorySearchResult
    from powercontext.builtin.runtime.application import BuiltinRuntime

SOURCE_KIND = "source"
MEMORY_KIND = "memory"
TOPIC_KIND = "topic"
POLICIES = ("all", "any")
PURPOSES = ("generation", "audit")
LIFECYCLE_REASON = "source lifecycle"
_TABLE = SOURCE_LIFECYCLE_NODES_TABLE

__all__ = [
    "ScopedSourceLifecycleApplication",
    "SourceLifecycleApplication",
    "SourceLifecycleError",
    "SourceLifecycleEvent",
]


class SourceLifecycleError(ValueError):
    """Report an invalid lifecycle graph, event, or reference."""


class SourceLifecycleEvent(BaseModel):
    """One ordered availability change of a registered Source node."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    revision: Annotated[int, Field(strict=True, gt=0)]
    sequence: Annotated[int, Field(strict=True, gt=0)]
    available: Annotated[bool, Field(strict=True)]
    reason: str

    @field_validator("node_id", "reason")
    @classmethod
    def require_text(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("lifecycle event text must be non-empty and not only whitespace")  # noqa: TRY003
        return value


@dataclass(frozen=True, slots=True)
class _Node:
    """One persisted node of a Scope's evidence dependency graph."""

    node_id: str
    kind: str
    reference: dict[str, Any]
    parents: tuple[str, ...]
    policy: str
    revision: int | None
    available: bool
    fingerprint: str
    sequence: int | None
    event_digest: str | None
    ordinal: int


@dataclass(frozen=True, slots=True)
class _NodeDraft:
    """One registration request before it is validated against the stored graph."""

    node_id: str
    kind: str
    reference: dict[str, Any]
    parents: tuple[str, ...]
    policy: str
    revision: int | None = None
    citation: MemoryCitation | None = None


class SourceLifecycleApplication:
    """Select the lifecycle service serving one Scope's dependency graph."""

    def __init__(self, runtime: BuiltinRuntime, database: AsyncDatabase | None = None) -> None:
        self._runtime = runtime
        self._database = database

    def for_scope(self, scope_id: str, /) -> ScopedSourceLifecycleApplication:
        if self._database is None:
            raise SourceLifecycleError("lifecycle state requires the built-in relational Runtime")  # noqa: TRY003
        return ScopedSourceLifecycleApplication(self._runtime, self._database, scope_id)

    async def filter_topic_result(self, scope_id: str, result: TopicMemorySearchResult, /) -> TopicMemorySearchResult:
        """Hide unsupported Topic Memory from native search, keeping the graph optional."""

        if self._database is None:
            return result
        return await self.for_scope(scope_id).filter_topic_result(result)


class ScopedSourceLifecycleApplication:
    """Register one Scope's evidence graph and propagate its Source availability."""

    def __init__(self, runtime: BuiltinRuntime, database: AsyncDatabase, scope_id: str) -> None:
        self._runtime = runtime
        self._database = database
        self.scope_id = validate_scope_id(scope_id)

    async def register_source(self, node_id: str, reference: SourceRef, /, *, revision: int) -> None:
        """Register one captured Source as a lifecycle root at an exact lifecycle revision."""

        _require_node_id(node_id)
        _require_positive(revision, "revision")
        if not isinstance(reference, SourceRef):
            raise SourceLifecycleError("a Source node needs a SourceRef")  # noqa: TRY003
        await self._require_source(reference)
        await self._register(
            _NodeDraft(
                node_id=node_id,
                kind=SOURCE_KIND,
                reference=reference.model_dump(mode="json"),
                parents=(),
                policy="",
                revision=revision,
            )
        )

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        /,
        *,
        parents: Sequence[str],
        policy: str,
    ) -> None:
        """Register one Memory entry version with the nodes that must support it."""

        _require_node_id(node_id)
        _require_policy(policy)
        resolved = _require_parents(parents)
        if not isinstance(citation, MemoryCitation):
            raise SourceLifecycleError("a Memory node needs a MemoryCitation")  # noqa: TRY003
        await self._require_memory(citation)
        await self._register(
            _NodeDraft(
                node_id=node_id,
                kind=MEMORY_KIND,
                reference=citation.model_dump(mode="json"),
                parents=resolved,
                policy=policy,
                citation=citation,
            )
        )

    async def register_topic(
        self,
        node_id: str,
        reference: ArtifactRef,
        /,
        *,
        parents: Sequence[str],
        policy: str,
    ) -> None:
        """Register one Topic Memory artifact with the nodes that must support it."""

        _require_node_id(node_id)
        _require_policy(policy)
        resolved = _require_parents(parents)
        if not isinstance(reference, ArtifactRef):
            raise SourceLifecycleError("a Topic Memory node needs an ArtifactRef")  # noqa: TRY003
        await self._require_topic(reference)
        await self._register(
            _NodeDraft(
                node_id=node_id,
                kind=TOPIC_KIND,
                reference=reference.model_dump(mode="json"),
                parents=resolved,
                policy=policy,
            )
        )

    async def apply(self, event: SourceLifecycleEvent, /) -> list[str]:
        """Apply one Source event and return every currently available node id, sorted."""

        if not isinstance(event, SourceLifecycleEvent):
            raise SourceLifecycleError("apply needs a SourceLifecycleEvent")  # noqa: TRY003
        digest = _event_digest(event)
        async with self._runtime._scope_operation(self.scope_id), self._runtime._locked(self.scope_id):
            nodes = await self._load_nodes()
            node = _require_event_target(nodes, event)
            if _is_replay(node, event, digest):
                return _available_ids(nodes)
            updated = dict(nodes)
            updated[event.node_id] = replace(
                node,
                available=event.available,
                sequence=event.sequence,
                event_digest=digest,
            )
            active = _propagate(updated)
            changed = tuple(item for item in updated.values() if active[item.node_id] != item.available)
            for item in changed:
                if item.kind == MEMORY_KIND:
                    await self._set_memory_available(
                        MemoryCitation.model_validate(item.reference),
                        active[item.node_id],
                    )
            await self._persist(event, active, changed, digest)
        return sorted(node_id for node_id, available in active.items() if available)

    async def _persist(
        self,
        event: SourceLifecycleEvent,
        active: Mapping[str, bool],
        changed: Sequence[_Node],
        digest: str,
    ) -> None:
        async with self._database.transaction() as connection:
            await connection.execute(
                update(_TABLE)
                .where(_TABLE.c.scope_id == self.scope_id, _TABLE.c.node_id == event.node_id)
                .values(
                    available=event.available,
                    sequence=event.sequence,
                    event_digest=digest,
                )
            )
            for item in changed:
                if item.node_id == event.node_id:
                    continue
                await connection.execute(
                    update(_TABLE)
                    .where(_TABLE.c.scope_id == self.scope_id, _TABLE.c.node_id == item.node_id)
                    .values(available=active[item.node_id])
                )

    async def resolve(
        self, node_id: str, /, *, purpose: str = "generation"
    ) -> SourceRef | MemoryCitation | ArtifactRef:
        """Return the registered reference for one node under the requested purpose."""

        if purpose not in PURPOSES:
            raise SourceLifecycleError(f"unknown resolution purpose {purpose!r}")  # noqa: TRY003
        nodes = await self._load_nodes()
        node = nodes.get(node_id)
        if node is None:
            raise SourceLifecycleError(f"node {node_id!r} is not registered")  # noqa: TRY003
        if purpose == "generation" and not node.available:
            raise SourceLifecycleError(f"node {node_id!r} is not available for generation")  # noqa: TRY003
        return _reference(node)

    async def filter_topic_result(self, result: TopicMemorySearchResult, /) -> TopicMemorySearchResult:
        """Hide Topic Memory artifacts that the registered graph no longer supports."""

        hidden = await self._hidden_topic_ids()
        if not hidden:
            return result
        hits = tuple(hit for hit in result.hits if hit.artifact_ref.artifact_id not in hidden)
        if len(hits) == len(result.hits):
            return result
        return result.model_copy(update={"hits": hits})

    # --- registration ------------------------------------------------------------------

    async def _register(self, draft: _NodeDraft) -> None:
        fingerprint = _fingerprint(draft)
        async with self._runtime._scope_operation(self.scope_id), self._runtime._locked(self.scope_id):
            nodes = await self._load_nodes()
            existing = nodes.get(draft.node_id)
            if existing is not None:
                if existing.fingerprint != fingerprint:
                    raise SourceLifecycleError(  # noqa: TRY003
                        f"node {draft.node_id!r} is already registered with a different binding",
                    )
                return
            for parent_id in draft.parents:
                if parent_id not in nodes:
                    raise SourceLifecycleError(  # noqa: TRY003
                        f"node {draft.node_id!r} names parent {parent_id!r} before it is registered",
                    )
            available = _derive(
                draft,
                {node_id: item.available for node_id, item in nodes.items()},
            )
            if draft.kind == MEMORY_KIND and draft.citation is not None:
                await self._set_memory_available(draft.citation, available)
            async with self._database.transaction() as connection:
                await connection.execute(
                    insert(_TABLE).values(
                        scope_id=self.scope_id,
                        node_id=draft.node_id,
                        kind=draft.kind,
                        reference=_payload(draft.reference),
                        parents=_payload(list(draft.parents)),
                        policy=draft.policy,
                        revision=draft.revision,
                        available=available,
                        fingerprint=fingerprint,
                        sequence=None,
                        event_digest=None,
                        ordinal=len(nodes),
                    )
                )

    async def _require_source(self, reference: SourceRef) -> None:
        try:
            await self._runtime.records.for_scope(self.scope_id).get_source(
                reference.source_type,
                reference.source_id,
            )
        except SourceLifecycleError:
            raise
        except Exception as error:
            raise SourceLifecycleError("registered Source is not readable") from error  # noqa: TRY003

    async def _require_memory(self, citation: MemoryCitation) -> None:
        try:
            await self._runtime.memory.for_scope(self.scope_id).get(GetMemoryEntryRequest(citation=citation))
        except SourceLifecycleError:
            raise
        except Exception as error:
            raise SourceLifecycleError("registered Memory entry is not readable") from error  # noqa: TRY003

    async def _require_topic(self, reference: ArtifactRef) -> None:
        try:
            await self._runtime.records.for_scope(self.scope_id).get_artifact_revision(
                reference.family,
                reference.artifact_id,
                reference.revision,
            )
        except SourceLifecycleError:
            raise
        except Exception as error:
            raise SourceLifecycleError("registered Topic Memory artifact is not readable") from error  # noqa: TRY003

    # --- native Memory state ------------------------------------------------------------

    async def _set_memory_available(self, citation: MemoryCitation, available: bool) -> None:
        """Move one registered Memory entry into the state its support currently implies.

        The Memory service compares the manifest state before writing, so repeating a
        propagation that already succeeded never creates another revision.
        """

        async with self._runtime._context(
            self.scope_id,
            embedding_purpose=ModelUsagePurpose.MEMORY_INDEXING,
        ) as context:
            service = context.artifacts.memory
            artifact_id = context.artifacts.memory_artifact_id
            if citation.memory_ref.artifact_id != artifact_id:
                raise SourceLifecycleError("Memory citation belongs to another artifact")  # noqa: TRY003
            memory, entries = await service.head_entries(artifact_id)
            manifest = {item.entry_id: item for item in memory.content.manifest.entries}
            item = manifest.get(citation.entry_id)
            version = next((entry for entry in entries if entry.entry_id == citation.entry_id), None)
            if item is None or version is None:
                raise SourceLifecycleError("registered Memory entry is no longer in the head")  # noqa: TRY003
            if (item.state == "active") == available:
                return
            if available:
                await service.reactivate(memory, entries=(version,), reason=LIFECYCLE_REASON)
            else:
                await service.forget(memory, entries=(version,), reason=LIFECYCLE_REASON)

    # --- persistence --------------------------------------------------------------------

    async def _load_nodes(self) -> dict[str, _Node]:
        async with self._database.transaction() as connection:
            rows = (
                await connection.execute(
                    select(
                        _TABLE.c.node_id,
                        _TABLE.c.kind,
                        _TABLE.c.reference,
                        _TABLE.c.parents,
                        _TABLE.c.policy,
                        _TABLE.c.revision,
                        _TABLE.c.available,
                        _TABLE.c.fingerprint,
                        _TABLE.c.sequence,
                        _TABLE.c.event_digest,
                        _TABLE.c.ordinal,
                    )
                    .where(_TABLE.c.scope_id == self.scope_id)
                    .order_by(_TABLE.c.ordinal)
                )
            ).all()
        return {
            str(row.node_id): _Node(
                node_id=str(row.node_id),
                kind=str(row.kind),
                reference=_decode(row.reference),
                parents=tuple(str(parent) for parent in _decode(row.parents)),
                policy=str(row.policy),
                revision=None if row.revision is None else int(row.revision),
                available=bool(row.available),
                fingerprint=str(row.fingerprint),
                sequence=None if row.sequence is None else int(row.sequence),
                event_digest=None if row.event_digest is None else str(row.event_digest),
                ordinal=int(row.ordinal),
            )
            for row in rows
        }

    async def _hidden_topic_ids(self) -> set[str]:
        async with self._database.transaction() as connection:
            rows = (
                await connection.execute(
                    select(_TABLE.c.reference).where(
                        _TABLE.c.scope_id == self.scope_id,
                        _TABLE.c.kind == TOPIC_KIND,
                        _TABLE.c.available.is_(False),
                    )
                )
            ).all()
        return {str(_decode(row.reference).get("artifact_id")) for row in rows}


# --- graph rules -----------------------------------------------------------------------


def _derive(draft: _NodeDraft, active: Mapping[str, bool]) -> bool:
    if draft.kind == SOURCE_KIND:
        return True
    if not draft.parents:
        raise SourceLifecycleError("a derived node needs at least one registered parent")  # noqa: TRY003
    support = [active[parent] for parent in draft.parents]
    return all(support) if draft.policy == "all" else any(support)


def _propagate(nodes: Mapping[str, _Node]) -> dict[str, bool]:
    active: dict[str, bool] = {}
    for node in sorted(nodes.values(), key=lambda item: item.ordinal):
        if node.kind == SOURCE_KIND:
            active[node.node_id] = node.available
            continue
        support = [active[parent] for parent in node.parents]
        active[node.node_id] = all(support) if node.policy == "all" else any(support)
    return active


def _require_event_target(nodes: Mapping[str, _Node], event: SourceLifecycleEvent) -> _Node:
    node = nodes.get(event.node_id)
    if node is None or node.kind != SOURCE_KIND:
        raise SourceLifecycleError(f"source node {event.node_id!r} is not registered")  # noqa: TRY003
    if node.revision != event.revision:
        raise SourceLifecycleError(  # noqa: TRY003
            f"event revision {event.revision} does not match registered revision {node.revision}",
        )
    return node


def _is_replay(node: _Node, event: SourceLifecycleEvent, digest: str) -> bool:
    """Report whether this exact event already produced the stored sequence."""

    if node.sequence is None:
        return False
    if event.sequence < node.sequence:
        raise SourceLifecycleError(  # noqa: TRY003
            f"event sequence {event.sequence} is behind the applied sequence {node.sequence}",
        )
    if event.sequence > node.sequence:
        return False
    if node.event_digest != digest:
        raise SourceLifecycleError(  # noqa: TRY003
            f"event sequence {event.sequence} was already used for a different event",
        )
    return True


def _available_ids(nodes: Mapping[str, _Node]) -> list[str]:
    return sorted(node_id for node_id, available in _propagate(nodes).items() if available)


def _reference(node: _Node) -> SourceRef | MemoryCitation | ArtifactRef:
    if node.kind == SOURCE_KIND:
        return SourceRef.model_validate(node.reference)
    if node.kind == MEMORY_KIND:
        return MemoryCitation.model_validate(node.reference)
    return ArtifactRef.model_validate(node.reference)


def _require_node_id(node_id: str) -> None:
    if not isinstance(node_id, str) or not node_id.strip() or node_id != node_id.strip():
        raise SourceLifecycleError("node_id must be a non-empty trimmed string")  # noqa: TRY003
    if len(node_id) > MAX_SOURCE_ID_LENGTH:
        raise SourceLifecycleError(f"node_id must not exceed {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003


def _require_positive(value: int, field: str) -> None:
    if type(value) is not int or value < 1:
        raise SourceLifecycleError(f"{field} must be a positive integer")  # noqa: TRY003


def _require_policy(policy: str) -> None:
    if policy not in POLICIES:
        raise SourceLifecycleError(f"policy must be one of {POLICIES}")  # noqa: TRY003


def _require_parents(parents: Sequence[str]) -> tuple[str, ...]:
    if isinstance(parents, str) or not parents:
        raise SourceLifecycleError("a derived node needs at least one registered parent")  # noqa: TRY003
    resolved = tuple(parents)
    for parent in resolved:
        if not isinstance(parent, str) or not parent.strip():
            raise SourceLifecycleError("parent ids must be non-empty strings")  # noqa: TRY003
    if len(set(resolved)) != len(resolved):
        raise SourceLifecycleError("parent ids must not repeat")  # noqa: TRY003
    return resolved


# --- durable encoding ------------------------------------------------------------------


def _payload(value: Any) -> bytes:
    return rfc8785.dumps(value)


def _decode(value: Any) -> Any:
    return json.loads(bytes(value))


def _fingerprint(draft: _NodeDraft) -> str:
    declaration = {
        "kind": draft.kind,
        "reference": draft.reference,
        "parents": list(draft.parents),
        "policy": draft.policy,
        "revision": draft.revision,
    }
    return "sha256:" + hashlib.sha256(_payload(declaration)).hexdigest()


def _event_digest(event: SourceLifecycleEvent) -> str:
    declaration = {
        "node_id": event.node_id,
        "revision": event.revision,
        "sequence": event.sequence,
        "available": event.available,
        "reason": event.reason,
    }
    return "sha256:" + hashlib.sha256(_payload(declaration)).hexdigest()
