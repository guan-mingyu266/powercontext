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

"""Capture one offline CSV document as one Source per row.

A row keeps the exact header order and the exact field strings, so a retrieval
citation can be traced back to the original record after reordering or a restart.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterator
from dataclasses import dataclass
from urllib.parse import quote

from pydantic import BaseModel

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

SOURCE_TYPE = "csv-row"
DEFINITION_VERSION = "1"

__all__ = [
    "CSV_ROW_SOURCE_DEFINITION",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "csv_row_source_id",
    "parse_csv_document",
]


class CSVDocumentInput(BaseModel):
    """One complete offline CSV file and the column that identifies a record."""

    document_id: str
    primary_key: str
    content: bytes


class CSVRowInput(BaseModel):
    """One record resolved against its header, values kept verbatim."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


class CSVRowSource(Source):
    """One captured row whose identity is stable under reordering and edits."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


def parse_csv_document(value: CSVDocumentInput) -> tuple[CSVRowInput, ...]:
    """Parse and validate the whole document before returning any row."""

    document_id = _validated_document_id(value.document_id)
    primary_key = value.primary_key
    if not primary_key:
        raise ValueError("primary_key must be a non-empty string")  # noqa: TRY003

    text = _decode(value.content)
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        rows = _parse_records(reader, document_id=document_id, primary_key=primary_key)
    except csv.Error as error:
        raise ValueError(f"malformed CSV content: {error}") from error  # noqa: TRY003
    return tuple(rows)


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Build the delimiter-safe identity of one row."""

    document = _validated_document_id(document_id)
    if not primary_key_value.strip():
        raise ValueError("primary_key_value must be non-empty and not whitespace")  # noqa: TRY003

    identity = f"csv:{quote(document, safe='')}:{quote(primary_key_value, safe='')}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(  # noqa: TRY003
            f"row identity must not exceed {MAX_SOURCE_ID_LENGTH} characters, got {len(identity)}",
        )
    return identity


def _parse_records(reader: Iterator[list[str]], *, document_id: str, primary_key: str) -> list[CSVRowInput]:
    header = next(reader, None)
    if header is None:
        raise ValueError("CSV document is empty")  # noqa: TRY003
    columns = tuple(header)
    _validate_header(columns)

    try:
        key_index = columns.index(primary_key)
    except ValueError:
        raise ValueError(  # noqa: TRY003
            f"primary_key {primary_key!r} does not match any header field exactly",
        ) from None

    rows: list[CSVRowInput] = []
    seen_keys: set[str] = set()
    for record in reader:
        if not record:
            # A completely empty physical record carries no field values.
            continue
        if len(record) != len(columns):
            raise ValueError(  # noqa: TRY003
                f"record has {len(record)} fields but the header declares {len(columns)}",
            )
        primary_key_value = record[key_index]
        if not primary_key_value.strip():
            raise ValueError("primary_key_value must be non-empty and not whitespace")  # noqa: TRY003
        if primary_key_value in seen_keys:
            raise ValueError(f"duplicate primary_key_value {primary_key_value!r}")  # noqa: TRY003
        seen_keys.add(primary_key_value)
        rows.append(
            CSVRowInput(
                document_id=document_id,
                primary_key=primary_key,
                primary_key_value=primary_key_value,
                columns=columns,
                values=tuple(record),
            ),
        )
    return rows


def _validate_header(columns: tuple[str, ...]) -> None:
    if not columns:
        raise ValueError("CSV header must declare at least one column")  # noqa: TRY003
    seen: set[str] = set()
    for column in columns:
        if not column or not column.strip():
            raise ValueError("CSV header fields must be non-empty and not whitespace")  # noqa: TRY003
        if column in seen:
            raise ValueError(f"duplicate CSV header field {column!r}")  # noqa: TRY003
        seen.add(column)


def _decode(content: bytes) -> str:
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"CSV content must be valid UTF-8: {error}") from error  # noqa: TRY003


def _validated_document_id(document_id: str) -> str:
    if not document_id or not document_id.strip():
        raise ValueError("document_id must be a non-empty string")  # noqa: TRY003
    if document_id != document_id.strip():
        raise ValueError("document_id must not contain surrounding whitespace")  # noqa: TRY003
    return document_id


@dataclass(frozen=True, slots=True)
class _CSVRowAdapter:
    """Resolve a parsed row into its Source and read it back exactly."""

    name: str = SOURCE_TYPE
    input_class: type[CSVRowInput] = CSVRowInput
    source_class: type[CSVRowSource] = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        source_id = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=source_id,
            definition_version=DEFINITION_VERSION,
            materialization=SourceMaterialization.CAPTURED,
            source_id=source_id,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


@dataclass(frozen=True, slots=True)
class _CSVRowTextEvidence:
    """Project a row into the standard textual Artifact evidence shape."""

    name: str = TEXT_EVIDENCE_PROJECTION_KEY.name
    version: str = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class: type[CSVRowSource] = CSVRowSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row = dict(zip(source.columns, source.values, strict=True))
        return TextEvidence(
            source_type=SOURCE_TYPE,
            source_id=source.source_id,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


CSV_ROW_SOURCE_DEFINITION: AdapterSourceDefinition[CSVRowInput, CSVRowSource, CSVRowInput] = AdapterSourceDefinition(
    adapter=_CSVRowAdapter(),
    version=DEFINITION_VERSION,
    projections=(_CSVRowTextEvidence(),),
)
