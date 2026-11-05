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

"""Capture the selected branch of a Pi session as immutable event Sources.

A Pi session file stores a tree: every record names its parent, and one leaf
selects the history path the user actually saw. This module parses that path
from root to leaf, keeps one Source per visible event, and refuses partial
results so a later citation always resolves to the exact original record.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel, JsonValue

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.connectors import (
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
)
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

SOURCE_TYPE = "pi-event"
CONNECTOR_NAME = "pi-session-history"
DEFINITION_VERSION = "1"
SESSION_TYPE = "session"
SESSION_VERSION = 3
CHECKPOINT_SCHEMA = "powercontext.pi-session-checkpoint.v1"

_UNKNOWN_TOOL = "unknown"

__all__ = [
    "PI_EVENT_SOURCE_DEFINITION",
    "PiEventInput",
    "PiEventSource",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_id",
]


class PiSessionInput(BaseModel):
    """One complete Pi session file and the leaf that selects the watched path."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible event on the selected branch, with its complete input record."""

    session_id: str
    event_id: str
    parent_id: str | None = None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiEventSource(Source):
    """One captured session event whose identity is stable across branch switches."""

    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None = None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Parse the whole file, then return only the selected branch's visible events."""

    records = _records(_decode(value.content))
    session_id = _session_id(records[0])
    nodes = _nodes(records[1:])
    leaf_id = value.leaf_id
    if leaf_id is None:
        leaf_id = next(reversed(nodes), None)
        if leaf_id is None:
            return ()
    elif not isinstance(leaf_id, str) or leaf_id not in nodes:
        raise ValueError(f"leaf_id {leaf_id!r} is not a session node")  # noqa: TRY003
    events: list[PiEventInput] = []
    for event_id in _ancestry(nodes, leaf_id):
        event = _visible_event(session_id, event_id, nodes[event_id])
        if event is not None:
            events.append(event)
    return tuple(events)


def pi_event_source_id(session_id: str, event_id: str) -> str:
    """Build the delimiter-safe identity of one session event."""

    identity = f"pi:{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        # ``quote`` never emits ``!``, so the two forms cannot collide.
        identity = "pi!" + hashlib.sha256(f"{session_id}\x00{event_id}".encode()).hexdigest()
    return identity


# --- session tree ----------------------------------------------------------------------


def _decode(content: bytes) -> str:
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"session content must be valid UTF-8: {error}") from error  # noqa: TRY003


def _records(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"line {number} is not valid JSON: {error}") from error  # noqa: TRY003
        if not isinstance(value, dict):
            raise ValueError(f"line {number} is not a JSON object")  # noqa: TRY003, TRY004
        records.append(value)
    if not records:
        raise ValueError("session file is empty")  # noqa: TRY003
    return records


def _session_id(header: dict[str, Any]) -> str:
    if header.get("type") != SESSION_TYPE:
        raise ValueError("first record must be the session header")  # noqa: TRY003
    version = header.get("version")
    if type(version) is not int or version != SESSION_VERSION:
        raise ValueError(f"session version must be {SESSION_VERSION}")  # noqa: TRY003
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session id must be a non-empty string")  # noqa: TRY003
    if session_id != session_id.strip():
        raise ValueError("session id must not contain surrounding whitespace")  # noqa: TRY003
    return session_id


def _nodes(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    nodes: dict[str, dict[str, Any]] = {}
    for record in records:
        event_id = record.get("id")
        if not isinstance(event_id, str) or not event_id.strip():
            raise ValueError("every session record needs a non-empty string id")  # noqa: TRY003
        if event_id != event_id.strip():
            raise ValueError("record ids must not contain surrounding whitespace")  # noqa: TRY003
        if event_id in nodes:
            raise ValueError(f"duplicate session record id {event_id!r}")  # noqa: TRY003
        parent_id = record.get("parentId")
        if parent_id is not None:
            if not isinstance(parent_id, str) or not parent_id.strip():
                raise ValueError(f"record {event_id!r} declares an invalid parentId")  # noqa: TRY003
            # Declaring the parent before the child also rules out cycles.
            if parent_id not in nodes:
                raise ValueError(f"record {event_id!r} names a parent that is not present yet")  # noqa: TRY003
        nodes[event_id] = record
    return nodes


def _ancestry(nodes: dict[str, dict[str, Any]], leaf_id: str) -> list[str]:
    chain: list[str] = []
    current: str | None = leaf_id
    while current is not None:
        chain.append(current)
        current = nodes[current].get("parentId")
    chain.reverse()
    return chain


# --- visible event text ----------------------------------------------------------------


def _visible_event(
    session_id: str,
    event_id: str,
    record: dict[str, Any],
) -> PiEventInput | None:
    event_type = record.get("type")
    parent_id = record.get("parentId")
    if event_type == "message":
        role, body = _message_body(record.get("message"))
    elif event_type in {"compaction", "branch_summary"}:
        summary = record.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            return None
        role, body = str(event_type), summary
    elif event_type == "custom_message":
        body = _blocks_text(record.get("content"))
        if body is None:
            return None
        role = "custom"
    else:
        return None
    if body is None:
        return None
    return PiEventInput(
        session_id=session_id,
        event_id=event_id,
        parent_id=parent_id if isinstance(parent_id, str) else None,
        event_type=str(event_type),
        role=role,
        content=f"{role}: {body}",
        record=record,
    )


def _message_body(message: object) -> tuple[str, str | None]:
    if not isinstance(message, dict):
        return "", None
    role = message.get("role")
    if not isinstance(role, str) or not role.strip():
        return "", None
    if role == "toolResult":
        name = message.get("toolName")
        call_id = message.get("toolCallId")
        text = _blocks_text(message.get("content"))
        marker = f"[tool_result {_text(name) or _UNKNOWN_TOOL} {_text(call_id)}]"
        return role, marker if text is None else f"{marker} {text}"
    if role == "bashExecution":
        command = _text(message.get("command"))
        output = _text(message.get("output"))
        if output is None:
            output = _blocks_text(message.get("content"))
        body = f"$ {command}"
        if output is not None:
            body = f"{body}\n{output}"
        return role, body
    return role, _blocks_text(message.get("content"))


def _blocks_text(blocks: object) -> str | None:
    if isinstance(blocks, str):
        return blocks or None
    if not isinstance(blocks, list):
        return None
    pieces: list[str] = []
    for block in blocks:
        text = _block_text(block)
        if text is not None:
            pieces.append(text)
    if not pieces:
        return None
    return "\n".join(pieces)


def _block_text(block: object) -> str | None:
    if not isinstance(block, dict):
        return None
    kind = block.get("type")
    if kind == "text":
        text = block.get("text")
        return text if isinstance(text, str) and text else None
    if kind == "toolCall":
        name = _text(block.get("name")) or _UNKNOWN_TOOL
        call_id = _text(block.get("id"))
        arguments = block.get("arguments")
        return f"[tool_call {name} {call_id}] {_compact_json(arguments)}"
    return None


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _compact_json(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"tool call arguments are not JSON: {error}") from error  # noqa: TRY003


# --- source definition -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _PiEventAdapter:
    """Resolve a parsed event into its Source and read it back exactly."""

    name: str = SOURCE_TYPE
    input_class: type[PiEventInput] = PiEventInput
    source_class: type[PiEventSource] = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        return PiEventSource(
            name=pi_event_source_id(value.session_id, value.event_id),
            definition_version=DEFINITION_VERSION,
            materialization=SourceMaterialization.CAPTURED,
            source_id=pi_event_source_id(value.session_id, value.event_id),
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


@dataclass(frozen=True, slots=True)
class _PiEventTextEvidence:
    """Project an event into the standard textual Artifact evidence shape."""

    name: str = TEXT_EVIDENCE_PROJECTION_KEY.name
    version: str = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class: type[PiEventSource] = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=SOURCE_TYPE,
            source_id=source.source_id,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_DEFINITION: AdapterSourceDefinition[PiEventInput, PiEventSource, PiEventInput] = (
    AdapterSourceDefinition(
        adapter=_PiEventAdapter(),
        version=DEFINITION_VERSION,
        projections=(_PiEventTextEvidence(),),
    )
)


# --- incremental connector -------------------------------------------------------------


class PiSessionConnector:
    """Import the newly visible events of one Pi session branch, once each."""

    name = CONNECTOR_NAME
    version = DEFINITION_VERSION
    source_definitions = frozenset({SOURCE_TYPE})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self.path = Path(path)
        self.leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        events = parse_pi_session(PiSessionInput(content=self.path.read_bytes(), leaf_id=self.leaf_id))
        imported = _imported_digests(session.checkpoint, events)
        digests: dict[str, str] = {}
        for event in events:
            item_id = pi_event_source_id(event.session_id, event.event_id)
            digest = _record_digest(event.record)
            if imported.get(item_id, digest) != digest:
                raise ValueError(  # noqa: TRY003
                    f"already imported session event {event.event_id!r} was rewritten",
                )
            digests[item_id] = digest
        for event in events:
            item_id = pi_event_source_id(event.session_id, event.event_id)
            if item_id in imported:
                continue
            await session.submit(item_id, SOURCE_TYPE, event)
        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE,
            checkpoint=_checkpoint(session.checkpoint, digests, events),
        )


def _imported_digests(checkpoint: JsonValue | None, events: tuple[PiEventInput, ...]) -> dict[str, str]:
    if checkpoint is None:
        return {}
    if not isinstance(checkpoint, dict) or checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("connector checkpoint is not a Pi session checkpoint")  # noqa: TRY003
    sessions = checkpoint.get("sessions")
    if not isinstance(sessions, dict):
        raise ValueError("connector checkpoint is missing its imported sessions")  # noqa: TRY003, TRY004
    session_id = events[0].session_id if events else None
    imported = sessions.get(session_id) if session_id is not None else {}
    if imported is None:
        return {}
    if not isinstance(imported, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in imported.items()
    ):
        raise ValueError("connector checkpoint imported events are malformed")  # noqa: TRY003
    return dict(imported)


def _checkpoint(
    previous: JsonValue | None,
    digests: dict[str, str],
    events: tuple[PiEventInput, ...],
) -> dict[str, JsonValue]:
    sessions: dict[str, JsonValue] = {}
    if isinstance(previous, dict) and isinstance(previous.get("sessions"), dict):
        sessions.update(previous["sessions"])
    if events:
        existing = sessions.get(events[0].session_id)
        merged = dict(existing) if isinstance(existing, dict) else {}
        merged.update(digests)
        sessions[events[0].session_id] = merged
    return {"schema": CHECKPOINT_SCHEMA, "sessions": sessions}


def _record_digest(record: dict[str, JsonValue]) -> str:
    canonical = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
