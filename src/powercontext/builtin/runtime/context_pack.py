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

"""Encode prepared Context as a chunked pack that later rounds can extend.

The first request ships the complete pack. A later request ships a delta that
names how many leading chunks of the previous pack the receiver may keep and the
suffix that replaces the rest, so a receiver rebuilds byte-identical context
without sharing process state with the sender.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

PACK_SCHEMA = "powercontext.context-pack.v1"
DELTA_SCHEMA = "powercontext.context-pack-delta.v1"
DIGEST_PREFIX = "sha256:"
DEFAULT_CHUNK_BYTES = 512

_PACK_FIELDS = frozenset({"schema", "items", "digest", "byte_count"})
_DELTA_FIELDS = frozenset({"schema", "base_digest", "target_digest", "prefix_items", "appended"})

__all__ = [
    "DEFAULT_CHUNK_BYTES",
    "DELTA_SCHEMA",
    "PACK_SCHEMA",
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
]


class ContextPackError(ValueError):
    """Report a protocol or chunking violation on the pack wire format."""


@dataclass(frozen=True, slots=True)
class ContextPack:
    """One complete Context body split into ordered, non-empty byte chunks."""

    items: tuple[bytes, ...] = ()
    digest: str = field(default="", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "digest", _pack_digest(self.items))

    @classmethod
    def build(cls, items: Iterable[bytes], /) -> ContextPack:
        """Build a complete pack from already chunked bytes."""

        return cls(items=_chunks(items))

    @classmethod
    def from_content(cls, content: str, /, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> ContextPack:
        """Split text into deterministic chunks that stay valid UTF-8 and bounded."""

        if not isinstance(content, str):
            raise ContextPackError("content must be a string")  # noqa: TRY003
        if type(chunk_bytes) is not int or chunk_bytes < 1:
            raise ContextPackError("chunk_bytes must be a positive integer")  # noqa: TRY003
        data = content.encode("utf-8")
        chunks: list[bytes] = []
        start = 0
        while start < len(data):
            end = min(start + chunk_bytes, len(data))
            while end > start and end < len(data) and _is_continuation(data[end]):
                end -= 1
            if end == start:
                raise ContextPackError("chunk_bytes cannot bound a single character")  # noqa: TRY003
            chunks.append(data[start:end])
            start = end
        return cls(items=tuple(chunks))

    @property
    def byte_count(self) -> int:
        """Report the serialized size of this pack, including the count itself."""

        return len(self.to_bytes())

    def content(self) -> str:
        """Reassemble the exact UTF-8 text this pack carries."""

        return b"".join(self.items).decode("utf-8")

    def to_bytes(self) -> bytes:
        """Serialize the pack into the interoperable wire encoding."""

        payload: dict[str, Any] = {
            "schema": PACK_SCHEMA,
            "items": [_encode_item(item) for item in self.items],
            "digest": self.digest,
        }
        payload["byte_count"] = _byte_count(payload)
        return _canonical(payload)

    @classmethod
    def from_bytes(cls, data: bytes, /) -> ContextPack:
        """Decode and validate one complete pack."""

        payload = _json_object(data, "pack")
        _expect_fields(payload, _PACK_FIELDS, "pack")
        if payload["schema"] != PACK_SCHEMA:
            raise ContextPackError("pack schema is not supported")  # noqa: TRY003
        pack = cls(items=_decode_items(payload["items"], "items"))
        if payload["digest"] != pack.digest:
            raise ContextPackError("pack digest does not match its items")  # noqa: TRY003
        byte_count = payload["byte_count"]
        if type(byte_count) is not int or byte_count != pack.byte_count:
            raise ContextPackError("pack byte_count does not match its encoding")  # noqa: TRY003
        return pack


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    """Name the leading base chunks to keep and the target chunks that replace the rest."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "appended", _chunks(self.appended))
        _require_digest(self.base_digest, "base_digest")
        _require_digest(self.target_digest, "target_digest")
        if type(self.prefix_items) is not int or self.prefix_items < 0:
            raise ContextPackError("prefix_items must be a non-negative integer")  # noqa: TRY003

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack, /) -> ContextPackDelta:
        """Choose the longest common prefix of two packs and carry the target suffix."""

        if not isinstance(base, ContextPack) or not isinstance(target, ContextPack):
            raise ContextPackError("a delta is built from two Context packs")  # noqa: TRY003
        prefix = 0
        for left, right in zip(base.items, target.items, strict=False):
            if left != right:
                break
            prefix += 1
        return cls(
            base_digest=base.digest,
            target_digest=target.digest,
            prefix_items=prefix,
            appended=target.items[prefix:],
        )

    @property
    def byte_count(self) -> int:
        """Report the serialized size of this delta; it is never a wire field."""

        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        """Serialize the delta into the interoperable wire encoding."""

        return _canonical({
            "schema": DELTA_SCHEMA,
            "base_digest": self.base_digest,
            "target_digest": self.target_digest,
            "prefix_items": self.prefix_items,
            "appended": [_encode_item(item) for item in self.appended],
        })

    @classmethod
    def from_bytes(cls, data: bytes, /) -> ContextPackDelta:
        """Decode and validate one delta."""

        payload = _json_object(data, "delta")
        _expect_fields(payload, _DELTA_FIELDS, "delta")
        if payload["schema"] != DELTA_SCHEMA:
            raise ContextPackError("delta schema is not supported")  # noqa: TRY003
        base_digest = _decode_digest(payload["base_digest"], "base_digest")
        target_digest = _decode_digest(payload["target_digest"], "target_digest")
        prefix_items = payload["prefix_items"]
        if type(prefix_items) is not int or prefix_items < 0:
            raise ContextPackError("prefix_items must be a non-negative integer")  # noqa: TRY003
        return cls(
            base_digest=base_digest,
            target_digest=target_digest,
            prefix_items=prefix_items,
            appended=_decode_items(payload["appended"], "appended"),
        )


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta, /) -> ContextPack:
    """Rebuild the target pack from a base pack and one received delta."""

    if not isinstance(base, ContextPack) or not isinstance(delta, ContextPackDelta):
        raise ContextPackError("a delta applies to one Context pack")  # noqa: TRY003
    if delta.base_digest != base.digest:
        raise ContextPackError("delta base digest does not match the held pack")  # noqa: TRY003
    if delta.prefix_items > len(base.items):
        raise ContextPackError("delta prefix exceeds the held pack")  # noqa: TRY003
    target = ContextPack.build((*base.items[: delta.prefix_items], *delta.appended))
    if target.digest != delta.target_digest:
        raise ContextPackError("delta target digest does not match the rebuilt pack")  # noqa: TRY003
    return target


# --- wire helpers ----------------------------------------------------------------------


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _json_object(data: object, kind: str) -> dict[str, Any]:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ContextPackError(f"{kind} must be serialized bytes")  # noqa: TRY003
    try:
        payload = json.loads(bytes(data).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContextPackError(f"{kind} is not valid UTF-8 JSON: {error}") from error  # noqa: TRY003
    if not isinstance(payload, dict):
        raise ContextPackError(f"{kind} must be a JSON object")  # noqa: TRY003
    return payload


def _expect_fields(payload: dict[str, Any], expected: frozenset[str], kind: str) -> None:
    actual = frozenset(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ContextPackError(  # noqa: TRY003
            f"{kind} fields must be exactly {sorted(expected)}, missing {missing}, extra {extra}",
        )


def _chunks(items: Iterable[bytes], /) -> tuple[bytes, ...]:
    if isinstance(items, (bytes, bytearray, memoryview, str)):
        raise ContextPackError("chunks must be a sequence of bytes objects")  # noqa: TRY003
    chunks: list[bytes] = []
    try:
        for item in items:
            if not isinstance(item, (bytes, bytearray, memoryview)):
                raise ContextPackError("each chunk must be bytes")  # noqa: TRY003
            chunks.append(bytes(item))
    except TypeError as error:
        raise ContextPackError(f"chunks must be iterable: {error}") from error  # noqa: TRY003
    for position, chunk in enumerate(chunks):
        try:
            chunk.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ContextPackError(f"chunk {position} is not valid UTF-8") from error  # noqa: TRY003
    return tuple(chunks)


def _encode_item(item: bytes) -> str:
    return base64.b64encode(item).decode("ascii")


def _decode_items(value: object, field: str) -> tuple[bytes, ...]:
    if not isinstance(value, list):
        raise ContextPackError(f"{field} must be a list of Base64 strings")  # noqa: TRY003
    items: list[bytes] = []
    for position, item in enumerate(value):
        if not isinstance(item, str):
            raise ContextPackError(f"{field}[{position}] must be a Base64 string")  # noqa: TRY003
        try:
            decoded = base64.b64decode(item.encode("ascii"), validate=True)
        except (UnicodeEncodeError, binascii.Error, ValueError) as error:
            raise ContextPackError(f"{field}[{position}] is not valid Base64") from error  # noqa: TRY003
        if _encode_item(decoded) != item:
            raise ContextPackError(f"{field}[{position}] is not canonical Base64")  # noqa: TRY003
        items.append(decoded)
    return _chunks(items)


def _pack_digest(items: tuple[bytes, ...]) -> str:
    unsigned = {"schema": PACK_SCHEMA, "items": [_encode_item(item) for item in items]}
    return DIGEST_PREFIX + hashlib.sha256(_canonical(unsigned)).hexdigest()


def _require_digest(value: object, field: str) -> None:
    _decode_digest(value, field)


def _decode_digest(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.startswith(DIGEST_PREFIX):
        raise ContextPackError(f"{field} must use sha256:<hex>")  # noqa: TRY003
    hexdigest = value[len(DIGEST_PREFIX) :]
    if len(hexdigest) != 64 or any(character not in "0123456789abcdef" for character in hexdigest):
        raise ContextPackError(f"{field} must use sha256:<hex>")  # noqa: TRY003
    return value


def _byte_count(payload: dict[str, Any]) -> int:
    """Find the byte_count whose decimal size is part of the serialized pack."""

    count = 0
    for _ in range(16):
        candidate = _canonical({**payload, "byte_count": count})
        if len(candidate) == count:
            return count
        count = len(candidate)
    raise ContextPackError("pack byte_count does not converge")  # noqa: TRY003


def _is_continuation(byte: int) -> bool:
    return byte & 0xC0 == 0x80
