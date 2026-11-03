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

"""Capture one offline PDF as one Source per physical page and retrieve evidence pages.

A page keeps the document identity, the digest of the whole PDF and the physical page
number, so a retrieval citation can be traced back to the original page after a restart or
a document revision. Retrieval stays inside the scoped Memory: candidates come from the
public search API and every returned citation is one exact entry version, never a value
reconstructed from the question.
"""

from __future__ import annotations

import hashlib
import io
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final
from urllib.parse import quote, unquote

from pydantic import BaseModel
from pypdf import PdfReader

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

if TYPE_CHECKING:
    from powercontext.artifacts import MemoryCitation
    from powercontext.builtin.artifacts.memory.models import MemoryHit
    from powercontext.builtin.runtime import ScopedMemoryApplication

SOURCE_TYPE: Final = "pdf-page"
DEFINITION_VERSION: Final = "1"

__all__ = [
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "parse_pdf_document",
    "pdf_page_source_id",
    "search_pdf_pages",
]


class PDFDocumentInput(BaseModel):
    """One complete offline PDF document and the identity the caller assigns to it."""

    document_id: str
    content: bytes


class PDFPageInput(BaseModel):
    """One physical page as extracted from the original document."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str


class PDFPageSource(Source):
    """One captured page whose identity is stable under reordering and document revisions."""

    source_id: str
    document_id: str
    document_sha256: str
    page_number: int
    text: str


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Extract every physical page in order, or fail without returning a partial result."""

    document_id = _validated_document_id(value.document_id)
    content = value.content
    if not content:
        raise ValueError("PDF content must not be empty")  # noqa: TRY003
    document_sha256 = hashlib.sha256(content).hexdigest()
    reader = _open_document(content)
    return _extract_pages(reader, document_id=document_id, document_sha256=document_sha256)


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int) -> str:
    """Build the delimiter-safe identity of one physical page."""

    document = _validated_document_id(document_id)
    digest = _validated_digest(document_sha256)
    if page_number < 1:
        raise ValueError("page_number must be a positive physical page number")  # noqa: TRY003

    identity = f"{SOURCE_TYPE}:{quote(document, safe='')}:{digest}:{page_number}"
    if len(identity) <= MAX_SOURCE_ID_LENGTH:
        return identity
    # A document id too long for a reference keeps its identity through a digest of the id.
    # The marker ``!`` cannot start a ``quote`` payload, so a short id that happens to look
    # like a digest can never meet a digested long id on the same name.
    compact = f"{SOURCE_TYPE}:!{hashlib.sha256(document.encode('utf-8')).hexdigest()}:{digest}:{page_number}"
    if len(compact) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"page identity must not exceed {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
    return compact


async def search_pdf_pages(
    memory: ScopedMemoryApplication,
    question: str,
    limit: int = 10,
) -> tuple[MemoryCitation, ...]:
    """Return up to ``limit`` exact citations to distinct pages that may answer the question.

    The scoped Memory is searched through its public API with several focused queries: the
    raw question, its content words, the concept groups the question activates, and, when the
    question asks for a reported number, the primary financial statements. Candidate pages are
    then scored offline over the text the Memory returned; the citations handed back are the
    exact entry versions those pages were scored from.
    """

    if limit < 1 or not question.lower().strip():
        return ()
    plan = _plan_queries(question)
    hits = await _search_candidates(memory, plan, limit=limit)
    if not hits:
        return ()
    ranked = _rank(_page_candidates(hits, plan), plan, limit=limit)
    return await _citations(memory, ranked, limit=limit)


# --- document parsing ------------------------------------------------------------------


def _open_document(content: bytes) -> PdfReader:
    try:
        reader = PdfReader(io.BytesIO(content))
    except Exception as error:
        raise ValueError("content is not a usable PDF document") from error  # noqa: TRY003
    try:
        # An empty user password is still a password-protected document the caller may read;
        # ``decrypt`` only fails when a real user password is required.
        readable = not reader.is_encrypted or bool(reader.decrypt(""))
    except Exception as error:
        raise ValueError("PDF permissions could not be read") from error  # noqa: TRY003
    if not readable:
        raise ValueError("PDF document requires a user password")  # noqa: TRY003
    return reader


def _extract_pages(
    reader: PdfReader,
    *,
    document_id: str,
    document_sha256: str,
) -> tuple[PDFPageInput, ...]:
    pages: list[PDFPageInput] = []
    number = 0
    try:
        for number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            pages.append(
                PDFPageInput(
                    document_id=document_id,
                    document_sha256=document_sha256,
                    page_number=number,
                    text=text.strip(),
                ),
            )
    except Exception as error:
        raise ValueError(f"physical page {number} could not be parsed: {error}") from error  # noqa: TRY003
    if not pages:
        raise ValueError("PDF document contains no physical pages")  # noqa: TRY003
    return tuple(pages)


def _validated_document_id(document_id: str) -> str:
    if not document_id or not document_id.strip():
        raise ValueError("document_id must be a non-empty string")  # noqa: TRY003
    return document_id


def _validated_digest(document_sha256: str) -> str:
    if _SHA256.fullmatch(document_sha256) is None:
        raise ValueError("document_sha256 must be a lowercase hexadecimal SHA-256 digest")  # noqa: TRY003
    return document_sha256


# --- source definition -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _PDFPageAdapter:
    """Resolve a parsed page into its Source and read it back exactly."""

    name: str = SOURCE_TYPE
    input_class: type[PDFPageInput] = PDFPageInput
    source_class: type[PDFPageSource] = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        source_id = pdf_page_source_id(value.document_id, value.document_sha256, value.page_number)
        return PDFPageSource(
            name=source_id,
            definition_version=DEFINITION_VERSION,
            materialization=SourceMaterialization.CAPTURED,
            source_id=source_id,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


@dataclass(frozen=True, slots=True)
class _PDFPageTextEvidence:
    """Project a page into the standard textual Artifact evidence shape."""

    name: str = TEXT_EVIDENCE_PROJECTION_KEY.name
    version: str = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class: type[PDFPageSource] = PDFPageSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=SOURCE_TYPE,
            source_id=source.source_id,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


PDF_PAGE_SOURCE_DEFINITION: AdapterSourceDefinition[PDFPageInput, PDFPageSource, PDFPageInput] = (
    AdapterSourceDefinition(
        adapter=_PDFPageAdapter(),
        version=DEFINITION_VERSION,
        projections=(_PDFPageTextEvidence(),),
    )
)


# --- question analysis -----------------------------------------------------------------

_SHA256: Final = re.compile(r"[0-9a-f]{64}")
_TERM: Final = re.compile(r"[a-z]+|\d+")


def _words(value: str) -> tuple[str, ...]:
    """Read one whitespace-separated vocabulary block as ordered, unique terms."""

    return tuple(dict.fromkeys(value.split()))


# Question wording carries two kinds of low-evidence words: grammatical fillers and the
# instructions that accompany a researched question. Neither constrains the answer page, so
# both are dropped from the retrieval terms. Domain words such as "margin", "cash" or
# "legal" are deliberately kept: they are the vocabulary of the report itself.
_QUESTION_FILLERS: Final = frozenset(
    _words(
        """
    a an the and or for of to in on at by from with as is are was were be been being what which
    who how why when where do does did can could should would i we you it its our your this that
    these those please has have had not no if then than also such may will any
    """,
    ),
)
_QUESTION_INSTRUCTIONS: Final = frozenset(
    _words(
        """
    answer answers respond response give given question questions assume assuming perspective rely
    relying define defined definition calculate calculated compute using use used only shown show
    based according provide provided details detail above below mentioned described outlined
    explicit explicitly directly quantity kind kinds type types value values usd state states
    stated reporting reported report reports useful metric metrics measure measures company
    companies explain reason reasons relevant relevance
    """,
    ),
)

# Suffix stripping is deliberate rather than a full stemmer: it only has to make a question
# word meet the form the report uses ("cyclicality"/"cyclical", "geographies"/"geographic").
_SUFFIXES: Final = ("ies", "es", "s", "ing", "ed", "ity", "ly", "ation", "ment", "ance", "ence", "al")

# Concept groups map the question's phrasing onto the vocabulary a financial report uses for
# the same idea. A group only contributes when one of its triggers occurs in the question.
_CONCEPTS: Final = (
    (
        ("legal", "litigation"),
        _words(
            "legal litigation lawsuit lawsuits proceedings claims regulatory investigations inquiries"
            " contingencies disputes",
        ),
    ),
    (
        ("geography",),
        _words(
            "geographic geographies geography region regions regional countries country territories"
            " segments worldwide international divisions markets",
        ),
    ),
    (("customer",), _words("customer customers clients consumers airlines buyers distributors")),
    (
        ("segment", "category", "categories"),
        _words(
            "segment segments reportable division divisions sales revenue operating income category categories",
        ),
    ),
    (("product",), _words("product products service services offerings brands portfolio")),
    (("cyclical",), _words("cyclical cyclicality cycle cycles downturn economic industry")),
    (("production",), _words("production manufacturing build rates rate")),
    (("margin",), _words("margin margins profitability operating profit")),
    (("tax",), _words("tax taxes taxation effective provision")),
    (("liquidity",), _words("liquidity liquid current assets liabilities ratio ratios quick cash")),
    (("capital", "capex"), _words("capital expenditures capex investments property plant equipment")),
    (
        ("growth",),
        _words(
            "growth grew grow increase increased decrease decreased change changes declined drivers",
        ),
    ),
    (("debt",), _words("debt borrowings leverage maturities obligations")),
    (("inventory",), _words("inventory inventories stock")),
    (("receivable",), _words("receivable receivables accounts")),
    (("payable",), _words("payable payables accounts days")),
    (("depreciation", "amortization", "ebitda"), _words("depreciation amortization ebitda")),
    (("revenue",), _words("revenue revenues sales turnover")),
    (("risk",), _words("risk risks uncertainty uncertainties exposure exposures")),
    (("employee",), _words("employee employees headcount workforce personnel")),
    (("competition",), _words("competition competitive competitors competitor rivals")),
    (("mna",), _words("acquisitions acquisition divestitures divestiture mergers merger")),
    (("dividend",), _words("dividend dividends payout")),
    (("buyback",), _words("repurchase repurchases buyback")),
    (("seasonal",), _words("seasonal seasonality backlog orders demand")),
)

# Two independent reported-number words are enough to treat a question as a request for a
# figure a financial statement reports, which is where such evidence lives in a report.
_REPORTED_NUMBER_TERMS: Final = frozenset(
    _words(
        """
    revenue revenues sales margin ebitda capex depreciation amortization tax rate ratio quick current
    liquidity cash flow flows income profit ebit operating gross net percent capital intensive
    inventory inventories payable receivable dpo dso dividend eps earnings debt leverage roe roa
    return growth yoy change increased decreased expense expenses cost costs balance statement
    operations
    """,
    ),
)

# A primary statement is recognised from its own heading, in the order a report opens them.
# The first group requires the "consolidated" qualification the statements carrying the
# reported numbers normally use; the second group accepts the bare heading so a document that
# titles a statement without that qualifier still anchors instead of falling through.
_STATEMENT_HEADINGS: Final = tuple(
    re.compile(pattern)
    for pattern in (
        r"consolidated statements? of (?:operations|income|earnings|loss)",
        r"consolidated balance sheets?",
        r"consolidated statements? of (?:cash flows?|financial position)",
        r"consolidated statements? of comprehensive",
        r"consolidated statements? of (?:stockholders|shareholders|changes in|equity)",
        r"statements? of (?:operations|income|earnings|loss)",
        r"balance sheets?",
        r"statements? of (?:cash flows?|financial position)",
        r"statements? of comprehensive",
        r"statements? of (?:stockholders|shareholders|changes in|equity)",
    )
)

# The four headings a report uses to open its primary statements.
_STATEMENT_PROBES: Final = (
    "consolidated balance sheets",
    "consolidated statements of cash flows",
    "consolidated statements of operations",
    "consolidated statements of income",
)

_FACET_TERMS: Final = 6
_CONCEPT_FACETS: Final = 6
_ANCHORED_STATEMENTS: Final = 4
_IDF_FLOOR: Final = 1e-6


@dataclass(frozen=True, slots=True)
class _QueryPlan:
    """The searches to run for one question and the weights used to rank their results."""

    facets: tuple[str, ...]
    weights: dict[str, float]
    wants_reported_numbers: bool


@dataclass(slots=True)
class _PageCandidate:
    """One scored Memory entry together with the page text it carries."""

    hit: MemoryHit
    counts: Counter[str]
    length: int
    statement: int | None


# --- retrieval -------------------------------------------------------------------------


async def _search_candidates(
    memory: ScopedMemoryApplication,
    plan: _QueryPlan,
    *,
    limit: int,
) -> list[MemoryHit]:
    """Run every planned query and keep the union of the returned entry versions."""

    from powercontext.builtin.runtime import SearchMemoryRequest

    pool = max(limit * 3, 60)
    hits: list[MemoryHit] = []
    seen: set[tuple[str, int, str, str]] = set()
    for facet in plan.facets:
        page = await memory.search(SearchMemoryRequest(query=facet, mode="fts", limit=pool))
        for hit in page.hits:
            key = (hit.memory_ref.artifact_id, hit.memory_ref.revision, hit.entry_id, hit.entry_version_id)
            if key in seen:
                continue
            seen.add(key)
            hits.append(hit)
    return hits


def _page_candidates(hits: Iterable[MemoryHit], plan: _QueryPlan) -> list[_PageCandidate]:
    candidates = []
    for hit in hits:
        body = _page_body(hit.text)
        tokens = _tokenize(body)
        candidates.append(
            _PageCandidate(
                hit=hit,
                counts=Counter(tokens),
                length=len(tokens),
                statement=_statement_priority(body) if plan.wants_reported_numbers else None,
            ),
        )
    return candidates


def _rank(candidates: Sequence[_PageCandidate], plan: _QueryPlan, *, limit: int) -> list[_PageCandidate]:
    """Order candidate pages by term evidence, with the statements of a numeric report first."""

    if not candidates:
        return []
    document_frequency: Counter[str] = Counter()
    for candidate in candidates:
        document_frequency.update(candidate.counts.keys())
    reference = max(1.0, sum(candidate.length for candidate in candidates) / len(candidates))
    # The number of entries in the scope is not part of the Memory interface, so the inverse
    # document frequency scale is estimated from the most common candidate term.
    scale = 2 * max(document_frequency.values(), default=1)
    scores: dict[int, float] = {}
    for position, candidate in enumerate(candidates):
        score = 0.0
        for term, weight in plan.weights.items():
            term_frequency = candidate.counts.get(term, 0)
            if not term_frequency:
                continue
            frequency = document_frequency[term]
            inverse = math.log((scale - frequency + 0.5) / (frequency + 0.5))
            score += weight * max(inverse, _IDF_FLOOR) * _saturation(term_frequency, candidate.length, reference)
        scores[position] = score
    if plan.wants_reported_numbers:
        ceiling = max(scores.values(), default=0.0) or 1.0
        statements: list[tuple[int, int]] = []
        for position, candidate in enumerate(candidates):
            priority = candidate.statement
            if priority is not None:
                statements.append((priority, position))
        statements.sort()
        # A statement anchor is a strong lead, not a replacement for the caller's budget.
        anchors = min(_ANCHORED_STATEMENTS, max(2, limit // 3))
        for offset, (_priority, position) in enumerate(statements[:anchors]):
            scores[position] = ceiling * (1.2 + (anchors - offset) * 0.001)
    ordered = sorted(
        range(len(candidates)),
        key=lambda position: (-scores[position], candidates[position].hit.entry_id),
    )
    return [candidates[position] for position in ordered[: max(limit * 2, 40)]]


async def _citations(
    memory: ScopedMemoryApplication,
    ranked: Sequence[_PageCandidate],
    *,
    limit: int,
) -> tuple[MemoryCitation, ...]:
    """Resolve the ranked entries and keep one citation per distinct physical page."""

    from powercontext.artifacts import MemoryCitation
    from powercontext.builtin.runtime import GetMemoryEntryRequest

    used: set[tuple[str, int]] = set()
    citations: list[MemoryCitation] = []
    for candidate in ranked:
        if len(citations) >= limit:
            break
        citation = MemoryCitation(
            memory_ref=candidate.hit.memory_ref,
            entry_id=candidate.hit.entry_id,
            entry_version_id=candidate.hit.entry_version_id,
        )
        record = await memory.get(GetMemoryEntryRequest(citation=citation))
        keys = [
            key for key in (_page_key(reference.source_id) for reference in record.entry.sources) if key is not None
        ]
        if not keys or used.intersection(keys):
            continue
        used.update(keys)
        citations.append(record.citation)
    return tuple(citations)


# --- question and page analysis ----------------------------------------------------------


def _plan_queries(question: str) -> _QueryPlan:
    terms = _content_terms(question)
    weights: dict[str, float] = {}
    for term in terms:
        weights[term] = 1.0
    for term in terms:
        for variant in _stems(term):
            if variant != term and len(variant) >= 4:
                weights.setdefault(variant, 0.5)
    concepts = [expansion for triggers, expansion in _CONCEPTS if any(_matches(term, triggers) for term in terms)]
    for expansion in concepts[:_CONCEPT_FACETS]:
        for term in expansion:
            weights.setdefault(term, 0.5)

    facets: list[str] = [question.strip()]
    for start in range(0, len(terms), _FACET_TERMS):
        facets.append(" ".join(dict.fromkeys(terms[start : start + _FACET_TERMS])))
    for expansion in concepts[:_CONCEPT_FACETS]:
        facets.append(" ".join(expansion[: _FACET_TERMS + 2]))
    wants_reported_numbers = len(set(terms) & _REPORTED_NUMBER_TERMS) >= 2
    if wants_reported_numbers:
        facets.extend(_STATEMENT_PROBES)
    return _QueryPlan(
        facets=tuple(dict.fromkeys(facet for facet in facets if facet.strip())),
        weights=weights,
        wants_reported_numbers=wants_reported_numbers,
    )


def _content_terms(question: str) -> list[str]:
    """Return the question's distinct retrieval terms, without filler or instructions."""

    terms = [
        term for term in _tokenize(question) if term not in _QUESTION_FILLERS and term not in _QUESTION_INSTRUCTIONS
    ]
    return list(dict.fromkeys(terms))


def _tokenize(text: str) -> list[str]:
    """Split text into the lowercase word and number terms a page is compared by."""

    return _TERM.findall(text.casefold())


def _stems(term: str) -> set[str]:
    """Return the surface forms one term may take in a report."""

    variants = {term}
    for suffix in _SUFFIXES:
        if not term.endswith(suffix) or len(term) - len(suffix) < 4:
            continue
        base = term[: -len(suffix)]
        variants.add(base)
        if suffix == "ies":
            variants.add(f"{base}y")
        if suffix in {"ing", "ed", "ation", "ance", "ence"}:
            variants.add(f"{base}e")
        if suffix == "ation":
            variants.add(f"{base}ate")
    return variants


def _matches(term: str, triggers: Iterable[str]) -> bool:
    """Return whether one question term belongs to a concept group."""

    variants = _stems(term)
    for trigger in triggers:
        for left in variants:
            for right in _stems(trigger):
                if left == right or (min(len(left), len(right)) >= 5 and _is_prefix(left, right)):
                    return True
    return False


def _is_prefix(left: str, right: str) -> bool:
    return left.startswith(right) or right.startswith(left)


def _page_body(text: str) -> str:
    """Drop the document id line the standard projection puts in front of the page text."""

    return text.split("\n", 1)[1] if "\n" in text else text


def _statement_priority(body: str) -> int | None:
    """Return the position of the primary statement this page opens, if it opens one."""

    head = " ".join(body.split()).lower()[:260]
    for priority, heading in enumerate(_STATEMENT_HEADINGS):
        if heading.search(head) is not None:
            return priority
    return None


def _saturation(term_frequency: int, length: int, reference: float) -> float:
    """Dampen repeated occurrences of one term in a page of a given length."""

    return term_frequency / (term_frequency + 1.2 * (1 - 0.75 + 0.75 * length / reference))


def _page_key(source_id: str) -> tuple[str, int] | None:
    """Return the document key and physical page an identity addresses, if it is one of ours."""

    parts = source_id.split(":")
    if len(parts) != 4 or parts[0] != SOURCE_TYPE:
        return None
    try:
        page_number = int(parts[3])
    except ValueError:
        return None
    if page_number < 1:
        return None
    document = parts[1] if parts[1].startswith("!") else unquote(parts[1])
    return (document, page_number)
