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

"""Prepare final, bounded context from approved Artifact evidence."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import chain, pairwise
from math import log
from typing import TypeVar

from powercontext.artifacts import ArtifactAddress, ArtifactRef
from powercontext.builtin.artifacts.experience import Experience, ExperienceSearchHit, render_experience
from powercontext.builtin.artifacts.memory.models import MemoryCitation, MemoryHit
from powercontext.builtin.artifacts.profile.models import Profile
from powercontext.builtin.artifacts.topic_memory import TopicMemory, TopicMemorySearchHit
from powercontext.builtin.runtime.errors import PreparedContextInvariantError
from powercontext.builtin.runtime.models import PrepareContextRequest, PreparedContext
from powercontext.builtin.runtime.prepared_code import (
    CodeEvidenceRef,
    PreparedCodeCandidate,
    assemble_code,
)
from powercontext.builtin.runtime.prepared_text import (
    TRUST_POLICY,
    ContextTextItem,
    fit_context_text_item,
    render_context_text,
)
from powercontext.builtin.runtime.recall_sufficiency import RecallBudgetView

_MIN_TRUNCATED_CONTENT_BYTES = 64
_ELLIPSIS = "…"
_BEGIN_MARKER = "BEGIN_POWERCONTEXT_PREPARED_CONTEXT_V1"
_END_MARKER = "END_POWERCONTEXT_PREPARED_CONTEXT_V1"
_Item = TypeVar("_Item")


@dataclass(frozen=True)
class _PreparedContextEntry:
    origin: PreparedContextOrigin
    kind: str
    citation: dict[str, object]
    content: str
    truncated: bool


@dataclass(frozen=True)
class _EntryFit:
    """One fit attempt on the non-assembly path: the entry that fitted, or why none did.

    The two drop reasons are disjoint and mirror the assembly path's
    :class:`~powercontext.builtin.runtime.prepared_text.FitOutcome`. An entry is dropped
    *below the minimum truncated content* when a shorter rendering exists and fits but is too
    short to be an honest delivery; it is dropped *with no fitting truncation* when no
    shortened rendering fits at all.
    """

    entry: _PreparedContextEntry | None = None
    dropped_below_min_bytes: bool = False
    dropped_no_fitting_truncation: bool = False


@dataclass(frozen=True)
class PreparedContextOmissions:
    """Aggregate counts of the items the byte budget omitted in one build.

    Every count is a per-call aggregate, never per-entry attribution and never a verdict about
    an entry: an entry losing to the budget is not a negative result about that entry.

    ``dropped_items`` is the sum of its two sub-counts, so "the budget could not fit this
    item" can be told apart from "this item was too short to truncate into the remaining
    space". A single merged counter would misreport the second cause as the first.
    """

    truncated_items: int = 0
    dropped_items: int = 0
    dropped_below_min_bytes: int = 0
    dropped_no_fitting_truncation: int = 0


@dataclass(frozen=True)
class PreparedContextBuild:
    """Final public context and the exact origins selected to produce it.

    ``omissions`` is always filled by the Builder. The RFC 1560 recall trace is deliberately
    **not** a field here: ``ScopedContextApplication._prepare`` returns ``build.context`` and
    discards the rest, so such a field would have no production observer. The trace is
    delivered through the Runtime's optional ``RecallEffortSink`` instead.
    """

    context: PreparedContext
    origins: tuple[PreparedContextOrigin, ...]
    omissions: PreparedContextOmissions = PreparedContextOmissions()
    code_origins: tuple[CodeEvidenceRef, ...] = ()
    fragments: tuple[PreparedContextFragment, ...] = ()


@dataclass(frozen=True)
class MemoryEntryAddress:
    """Identify one exact Memory entry version across Scope boundaries."""

    memory: ArtifactAddress
    entry_id: str
    entry_version_id: str


@dataclass(frozen=True)
class PreparedContextFragment:
    """One literal span of a recalled Memory entry, with its exact citation."""

    address: MemoryEntryAddress
    start: int
    end: int
    text: str


PreparedContextOrigin = MemoryCitation | ArtifactRef | MemoryEntryAddress | ArtifactAddress


@dataclass(frozen=True)
class PreparedMemoryCandidates:
    """Memory candidates read from one Scope."""

    scope_id: str
    memory_ref: ArtifactRef | None = None
    hits: tuple[MemoryHit, ...] = ()


@dataclass(frozen=True)
class PreparedExperienceCandidates:
    """Experience candidates read from one Scope."""

    scope_id: str
    hits: tuple[ExperienceSearchHit, ...] = ()


@dataclass(frozen=True)
class PreparedProfileCandidate:
    """The latest committed Profile snapshot read from one Scope."""

    scope_id: str
    profile: Profile


class PreparedContextBuilder:
    """Select and render final context without I/O, persistence, or reranking."""

    memory_candidate_limit = 16
    topic_memory_candidate_limit = 8
    experience_candidate_limit = 8
    candidate_limit = memory_candidate_limit
    entry_limit = 8
    topic_memory_entry_limit = 8
    experience_entry_limit = 2
    max_entry_content_bytes = 2000
    _text_entry_limit: int | None = None

    def empty(self) -> PreparedContext:
        return PreparedContext(status="empty", content=None, content_bytes=0)

    def build(
        self,
        *,
        request: PrepareContextRequest,
        scope_id: str | None = None,
        memory_ref: ArtifactRef | None = None,
        hits: Sequence[MemoryHit] = (),
        topic_memory_hits: Sequence[TopicMemorySearchHit] = (),
        experience_hits: Sequence[ExperienceSearchHit] = (),
    ) -> PreparedContext:
        return self.build_result(
            request=request,
            scope_id=scope_id,
            memory_ref=memory_ref,
            hits=hits,
            topic_memory_hits=topic_memory_hits,
            experience_hits=experience_hits,
        ).context

    def build_result(
        self,
        *,
        request: PrepareContextRequest,
        scope_id: str | None = None,
        memory_ref: ArtifactRef | None = None,
        hits: Sequence[MemoryHit] = (),
        topic_memory_hits: Sequence[TopicMemorySearchHit] = (),
        experience_hits: Sequence[ExperienceSearchHit] = (),
    ) -> PreparedContextBuild:
        return self.build_scopes_result(
            request=request,
            current_scope_id=scope_id,
            memory_candidates=(
                PreparedMemoryCandidates(scope_id=scope_id or "", memory_ref=memory_ref, hits=tuple(hits)),
            ),
            experience_candidates=(PreparedExperienceCandidates(scope_id=scope_id or "", hits=tuple(experience_hits)),),
            topic_memory_hits=topic_memory_hits,
        )

    def build_scopes_result(
        self,
        *,
        request: PrepareContextRequest,
        current_scope_id: str | None,
        memory_candidates: Sequence[PreparedMemoryCandidates] = (),
        topic_memory_hits: Sequence[TopicMemorySearchHit] = (),
        experience_candidates: Sequence[PreparedExperienceCandidates] = (),
        profile_candidates: Sequence[PreparedProfileCandidate] = (),
        code_candidates: Sequence[PreparedCodeCandidate] = (),
    ) -> PreparedContextBuild:
        if request.evidence_selection:
            return self._fragment_result(
                request,
                current_scope_id=current_scope_id,
                memory_candidates=memory_candidates,
            )
        if request.include_code:

            def historical(history_request: PrepareContextRequest, entries: int) -> PreparedContextBuild:
                builder = PreparedContextBuilder()
                builder.entry_limit = entries
                builder._text_entry_limit = entries if entries < self.entry_limit else None
                return builder.build_scopes_result(
                    request=history_request.model_copy(update={"include_code": False}),
                    current_scope_id=current_scope_id,
                    memory_candidates=memory_candidates,
                    topic_memory_hits=topic_memory_hits,
                    experience_candidates=experience_candidates,
                    profile_candidates=profile_candidates,
                )

            return assemble_code(request, code_candidates, self.entry_limit, historical)

        if sum(len(candidates.hits) for candidates in memory_candidates) > self.memory_candidate_limit:
            raise PreparedContextInvariantError("memory-candidate-limit")
        if len(topic_memory_hits) > self.topic_memory_candidate_limit:
            raise PreparedContextInvariantError("topic-memory-candidate-limit")
        if sum(len(candidates.hits) for candidates in experience_candidates) > self.experience_candidate_limit:
            raise PreparedContextInvariantError("experience-candidate-limit")

        content, origins, omissions = self._select_entries(
            request=request,
            current_scope_id=current_scope_id,
            memory_candidates=memory_candidates,
            experience_candidates=experience_candidates,
            profile_candidates=profile_candidates,
            topic_memory_hits=topic_memory_hits,
        )
        if content is None:
            return PreparedContextBuild(context=self.empty(), origins=(), omissions=omissions)
        content_bytes = len(content.encode("utf-8"))
        # The assembly path is bounded by construction (every fitted item is measured against
        # `max_bytes`), so the ceiling check belongs to the non-assembly renderer only.
        if request.assembly is None and content_bytes > request.max_bytes:
            raise PreparedContextInvariantError("output-budget")
        return PreparedContextBuild(
            context=PreparedContext(status="ready", content=content, content_bytes=content_bytes),
            origins=origins,
            omissions=omissions,
        )

    def _fragment_result(
        self,
        request: PrepareContextRequest,
        *,
        current_scope_id: str | None,
        memory_candidates: Sequence[PreparedMemoryCandidates],
    ) -> PreparedContextBuild:
        """Deliver literal query-relevant spans of recalled Memory entries within the budget."""

        if sum(len(candidates.hits) for candidates in memory_candidates) > self.memory_candidate_limit:
            raise PreparedContextInvariantError("memory-candidate-limit")
        fragments = self._select_fragments(
            request,
            current_scope_id=current_scope_id,
            memory_candidates=memory_candidates,
        )
        if not fragments:
            return PreparedContextBuild(context=self.empty(), origins=())
        content = _render_fragments(fragments)
        content_bytes = len(content.encode("utf-8"))
        if content_bytes > request.max_bytes:
            raise PreparedContextInvariantError("output-budget")
        return PreparedContextBuild(
            context=PreparedContext(status="ready", content=content, content_bytes=content_bytes),
            origins=tuple(fragment.address for fragment in fragments),
            fragments=fragments,
        )

    def _select_fragments(
        self,
        request: PrepareContextRequest,
        *,
        current_scope_id: str | None,
        memory_candidates: Sequence[PreparedMemoryCandidates],
    ) -> tuple[PreparedContextFragment, ...]:
        """Greedily pack the highest-weight literal spans the rendered budget still holds."""

        candidates = _fragment_candidates(memory_candidates, current_scope_id)
        query = _FragmentQuery(request.query)
        if not candidates or not query.terms:
            return ()
        query.score(candidates)
        covered: set[int] = set()
        selected: list[PreparedContextFragment] = []
        remaining = list(candidates)
        while remaining:
            best = max(
                remaining,
                key=lambda item: (
                    query.gain(item.keys, covered) + item.bonus + item.prior,
                    -(item.end - item.start),
                ),
            )
            if not query.gain(best.keys, covered) and not best.bonus:
                break
            fragment = _fit_fragment(best, selected, request.max_bytes)
            remaining.remove(best)
            if fragment is None:
                continue
            selected.append(fragment)
            covered |= best.keys
        return _extend_fragments(selected, candidates, request.max_bytes)

    def probe_budget(
        self,
        *,
        request: PrepareContextRequest,
        current_scope_id: str | None,
        memory_candidates: Sequence[PreparedMemoryCandidates] = (),
        topic_memory_hits: Sequence[TopicMemorySearchHit] = (),
        experience_candidates: Sequence[PreparedExperienceCandidates] = (),
        profile_candidates: Sequence[PreparedProfileCandidate] = (),
    ) -> RecallBudgetView:
        """Report what the byte budget does to one candidate set, without delivering it.

        This is the RFC 1560 *budget probe*: one pass of this Builder's own pure selection and
        rendering code, keeping only the counters and discarding the rendered output. It lets
        the gate tell budget-limited thinness from recall-limited thinness.

        Pure and side-effect free: no I/O, no persistence, nothing reported to any sink, and
        the same inputs as :meth:`build_scopes_result` are guaranteed to yield the same
        counters because both go through :meth:`_select_entries`.
        """

        if request.evidence_selection:
            fragments = self._select_fragments(
                request,
                current_scope_id=current_scope_id,
                memory_candidates=memory_candidates,
            )
            content_bytes = 0 if not fragments else len(_render_fragments(fragments).encode("utf-8"))
            return RecallBudgetView(
                max_bytes=request.max_bytes,
                delivered_items=len(fragments),
                truncated_items=0,
                dropped_items=0,
                unused_bytes=max(0, request.max_bytes - content_bytes),
            )
        content, origins, omissions = self._select_entries(
            request=request,
            current_scope_id=current_scope_id,
            memory_candidates=memory_candidates,
            experience_candidates=experience_candidates,
            profile_candidates=profile_candidates,
            topic_memory_hits=topic_memory_hits,
        )
        content_bytes = 0 if content is None else len(content.encode("utf-8"))
        return RecallBudgetView(
            max_bytes=request.max_bytes,
            delivered_items=len(origins),
            truncated_items=omissions.truncated_items,
            dropped_items=omissions.dropped_items,
            unused_bytes=max(0, request.max_bytes - content_bytes),
        )

    def _select_entries(
        self,
        *,
        request: PrepareContextRequest,
        current_scope_id: str | None,
        memory_candidates: Sequence[PreparedMemoryCandidates],
        topic_memory_hits: Sequence[TopicMemorySearchHit],
        experience_candidates: Sequence[PreparedExperienceCandidates],
        profile_candidates: Sequence[PreparedProfileCandidate],
    ) -> tuple[str | None, tuple[PreparedContextOrigin, ...], PreparedContextOmissions]:
        """Run one complete pure selection pass and render it.

        Returns ``(content, origins, omissions)``; ``content`` is ``None`` exactly when nothing
        was selected, which is the empty-result case for both paths. This is the single
        selection path shared by :meth:`build_scopes_result` and :meth:`probe_budget`, so the
        probe's counters always describe the selection the build would actually perform.
        """

        if request.assembly is not None:
            return self._select_text(
                request,
                current_scope_id=current_scope_id,
                memory_candidates=memory_candidates,
                experience_candidates=experience_candidates,
                profile_candidates=profile_candidates,
                topic_memory_hits=topic_memory_hits,
            )

        memory_entries = _interleave_groups(
            tuple(
                self._memory_entries(
                    candidates.memory_ref,
                    candidates.hits,
                    scope_id=None if candidates.scope_id == current_scope_id else candidates.scope_id or None,
                )
                for candidates in memory_candidates
            )
        )
        topic_memory_entries = self._topic_memory_entries(topic_memory_hits)
        experience_entries = _interleave_groups(
            tuple(
                self._experience_entries(
                    candidates.hits,
                    scope_id=None if candidates.scope_id == current_scope_id else candidates.scope_id or None,
                )
                for candidates in experience_candidates
            )
        )[: self.experience_entry_limit]
        entries, omissions = self._fit_entries(request, memory_entries, topic_memory_entries, experience_entries)
        if not entries:
            return None, (), omissions
        return _render(entries), tuple(entry.origin for entry in entries), omissions

    def _select_text(
        self,
        request: PrepareContextRequest,
        *,
        current_scope_id: str | None,
        memory_candidates: Sequence[PreparedMemoryCandidates],
        experience_candidates: Sequence[PreparedExperienceCandidates],
        profile_candidates: Sequence[PreparedProfileCandidate],
        topic_memory_hits: Sequence[TopicMemorySearchHit],
    ) -> tuple[str | None, tuple[PreparedContextOrigin, ...], PreparedContextOmissions]:
        assembly = request.assembly
        if assembly is None:
            raise PreparedContextInvariantError("text-assembly-missing")
        included: list[ContextTextItem] = []
        origins: list[PreparedContextOrigin] = []
        truncated_items = 0
        dropped_below_min_bytes = 0
        dropped_no_fitting_truncation = 0
        for section in assembly.sections:
            # Unfilled earlier sections leave their capacity available to later ones.
            limit = (
                section.limit
                if self._text_entry_limit is None
                else min(section.limit, self._text_entry_limit - len(included))
            )
            if section.family == "profile":
                entries = self._profile_entries(profile_candidates)
            elif section.family == "topic-memory":
                entries = self._topic_memory_entries(topic_memory_hits, scope_id=current_scope_id)
            else:
                groups = (
                    tuple(
                        tuple(
                            self._memory_entries(group.memory_ref, (hit,), scope_id=group.scope_id)
                            for hit in group.hits
                        )
                        for group in memory_candidates
                    )
                    if section.family == "memory"
                    else tuple(
                        tuple(self._experience_entries((hit,), scope_id=group.scope_id) for hit in group.hits)
                        for group in experience_candidates
                    )
                )
                entries = tuple(chain.from_iterable(_interleave_groups(groups)))
            seen: set[tuple[str, str, str, int, str | None, str | None]] = set()
            rank = 0
            selected_count = 0
            for entry in entries:
                if selected_count >= limit:
                    break
                item = _text_item(entry)
                artifact = item.artifact
                identity = (
                    artifact.scope_id,
                    artifact.artifact.family,
                    artifact.artifact.artifact_id,
                    artifact.artifact.revision,
                    item.entry_id,
                    item.entry_version_id,
                )
                if identity in seen or not item.content.strip():
                    continue
                seen.add(identity)
                rank += 1
                outcome = fit_context_text_item(included, replace(item, recall_rank=rank), assembly, request.max_bytes)
                if outcome.item is None:
                    dropped_below_min_bytes += int(outcome.dropped_below_min_bytes)
                    dropped_no_fitting_truncation += int(outcome.dropped_no_fitting_truncation)
                    continue
                fitted = outcome.item
                included.append(fitted)
                origins.append(entry.origin)
                selected_count += 1
                truncated_items += int(fitted.truncated)
        omissions = PreparedContextOmissions(
            truncated_items=truncated_items,
            dropped_items=dropped_below_min_bytes + dropped_no_fitting_truncation,
            dropped_below_min_bytes=dropped_below_min_bytes,
            dropped_no_fitting_truncation=dropped_no_fitting_truncation,
        )
        if not included:
            return None, (), omissions
        content = render_context_text(included, assembly)
        return content, tuple(origins), omissions

    def _profile_entries(self, candidates: Sequence[PreparedProfileCandidate]) -> tuple[_PreparedContextEntry, ...]:
        entries = []
        for candidate in candidates:
            origin = ArtifactAddress(scope_id=candidate.scope_id, artifact=candidate.profile.as_ref())
            entries.append(
                _PreparedContextEntry(
                    origin=origin,
                    kind="profile",
                    citation={"artifact": origin.model_dump(mode="json")},
                    content=candidate.profile.content.content,
                    truncated=False,
                )
            )
        return tuple(entries)

    def _memory_entries(
        self,
        memory_ref: ArtifactRef | None,
        hits: Sequence[MemoryHit],
        *,
        scope_id: str | None = None,
    ) -> tuple[_PreparedContextEntry, ...]:
        if hits and memory_ref is None:
            raise PreparedContextInvariantError("memory-ref-missing")
        memory_entries: list[_PreparedContextEntry] = []
        seen: set[tuple[str, str]] = set()
        for hit in hits:
            if hit.memory_ref != memory_ref:
                raise PreparedContextInvariantError("memory-ref-mismatch")

            citation_key = (hit.entry_id, hit.entry_version_id)
            if citation_key in seen:
                continue
            seen.add(citation_key)
            if not hit.entry_id.strip() or not hit.entry_version_id.strip() or not hit.text.strip():
                continue
            citation = MemoryCitation(
                memory_ref=hit.memory_ref,
                entry_id=hit.entry_id,
                entry_version_id=hit.entry_version_id,
            )
            origin: PreparedContextOrigin = citation
            rendered_citation = citation.model_dump(mode="json")
            if scope_id is not None:
                memory = ArtifactAddress(scope_id=scope_id, artifact=hit.memory_ref)
                origin = MemoryEntryAddress(
                    memory=memory,
                    entry_id=hit.entry_id,
                    entry_version_id=hit.entry_version_id,
                )
                rendered_citation = {
                    "memory": memory.model_dump(mode="json"),
                    "entry_id": hit.entry_id,
                    "entry_version_id": hit.entry_version_id,
                }
            memory_entries.append(
                _PreparedContextEntry(
                    origin=origin,
                    kind="memory",
                    citation=rendered_citation,
                    content=hit.text,
                    truncated=False,
                )
            )
        return tuple(memory_entries)

    def _topic_memory_entries(
        self,
        hits: Sequence[TopicMemorySearchHit],
        *,
        scope_id: str | None = None,
    ) -> tuple[_PreparedContextEntry, ...]:
        topic_entries: list[_PreparedContextEntry] = []
        seen_topics: set[tuple[str, int]] = set()
        for hit in hits:
            if hit.artifact_ref.family != TopicMemory.family:
                raise PreparedContextInvariantError("topic-memory-family-mismatch")
            identity = (hit.artifact_ref.artifact_id, hit.artifact_ref.revision)
            if identity in seen_topics:
                continue
            seen_topics.add(identity)
            if len(topic_entries) >= self.topic_memory_entry_limit:
                break
            content = {"title": hit.title, "summary": hit.summary}
            if hit.snippet is not None:
                content["snippet"] = hit.snippet
            origin: PreparedContextOrigin = hit.artifact_ref
            citation = {"artifact_ref": hit.artifact_ref.model_dump(mode="json")}
            body = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
            if scope_id is not None:
                origin = ArtifactAddress(scope_id=scope_id, artifact=hit.artifact_ref)
                citation = {"artifact": origin.model_dump(mode="json")}
                body = "\n\n".join(f"{label.capitalize()}: {value}" for label, value in content.items())
            topic_entries.append(
                _PreparedContextEntry(
                    origin=origin,
                    kind="topic-memory",
                    citation=citation,
                    content=body,
                    truncated=False,
                )
            )
        return tuple(topic_entries)

    def _experience_entries(
        self,
        hits: Sequence[ExperienceSearchHit],
        *,
        scope_id: str | None = None,
    ) -> tuple[_PreparedContextEntry, ...]:
        experience_entries: list[_PreparedContextEntry] = []
        seen_experiences: set[tuple[str, int]] = set()
        for hit in hits:
            if hit.artifact_ref.family != Experience.family:
                raise PreparedContextInvariantError("experience-family-mismatch")
            identity = (hit.artifact_ref.artifact_id, hit.artifact_ref.revision)
            if identity in seen_experiences:
                continue
            seen_experiences.add(identity)
            origin: PreparedContextOrigin = hit.artifact_ref
            rendered_citation: dict[str, object] = {"artifact_ref": hit.artifact_ref.model_dump(mode="json")}
            if scope_id is not None:
                origin = ArtifactAddress(scope_id=scope_id, artifact=hit.artifact_ref)
                rendered_citation = {"artifact": origin.model_dump(mode="json")}
            experience_entries.append(
                _PreparedContextEntry(
                    origin=origin,
                    kind="experience",
                    citation=rendered_citation,
                    content=render_experience(hit.content),
                    truncated=False,
                )
            )
        return tuple(experience_entries)

    def _fit_entries(
        self,
        request: PrepareContextRequest,
        memory_entries: Sequence[_PreparedContextEntry],
        topic_memory_entries: Sequence[_PreparedContextEntry],
        experience_entries: Sequence[_PreparedContextEntry],
    ) -> tuple[tuple[_PreparedContextEntry, ...], PreparedContextOmissions]:
        entries: list[_PreparedContextEntry] = []
        truncated_items = 0
        dropped_below_min_bytes = 0
        dropped_no_fitting_truncation = 0
        for candidate in _interleave(memory_entries, topic_memory_entries, experience_entries):
            if len(entries) >= self.entry_limit:
                break
            fit = self._fit_entry(
                entries,
                origin=candidate.origin,
                kind=candidate.kind,
                citation=candidate.citation,
                text=candidate.content,
                max_bytes=request.max_bytes,
            )
            if fit.entry is None:
                dropped_below_min_bytes += int(fit.dropped_below_min_bytes)
                dropped_no_fitting_truncation += int(fit.dropped_no_fitting_truncation)
                continue
            entries.append(fit.entry)
            if fit.entry.truncated:
                truncated_items += 1
        return tuple(entries), PreparedContextOmissions(
            truncated_items=truncated_items,
            dropped_items=dropped_below_min_bytes + dropped_no_fitting_truncation,
            dropped_below_min_bytes=dropped_below_min_bytes,
            dropped_no_fitting_truncation=dropped_no_fitting_truncation,
        )

    def _fit_entry(
        self,
        entries: Sequence[_PreparedContextEntry],
        *,
        origin: PreparedContextOrigin,
        kind: str,
        citation: dict[str, object],
        text: str,
        max_bytes: int,
    ) -> _EntryFit:
        """Fit one entry into the remaining budget, reporting why it was dropped if it was.

        The two ``None`` paths of the original code are kept apart: ``below-min-bytes`` means a
        shorter rendering exists and fits but is too short to deliver honestly, while
        ``no-fitting-truncation`` means no shortened rendering fitted at all.
        """

        source_bytes = len(text.encode("utf-8"))
        entry_budget = min(source_bytes, self.max_entry_content_bytes)
        candidate = _PreparedContextEntry(
            origin=origin,
            kind=kind,
            citation=citation,
            content=text if source_bytes <= entry_budget else _truncate_utf8(text, entry_budget),
            truncated=source_bytes > entry_budget,
        )
        if _rendered_bytes((*entries, candidate)) <= max_bytes:
            return _EntryFit(entry=candidate)
        if source_bytes < _MIN_TRUNCATED_CONTENT_BYTES:
            return _EntryFit(dropped_below_min_bytes=True)

        lower = _MIN_TRUNCATED_CONTENT_BYTES
        upper = min(entry_budget, source_bytes - 1)
        best: _PreparedContextEntry | None = None
        while lower <= upper:
            byte_budget = (lower + upper) // 2
            candidate = _PreparedContextEntry(
                origin=origin,
                kind=kind,
                citation=citation,
                content=_truncate_utf8(text, byte_budget),
                truncated=True,
            )
            if len(candidate.content.encode("utf-8")) < _MIN_TRUNCATED_CONTENT_BYTES:
                lower = byte_budget + 1
                continue
            if _rendered_bytes((*entries, candidate)) <= max_bytes:
                best = candidate
                lower = byte_budget + 1
            else:
                upper = byte_budget - 1
        if best is None:
            return _EntryFit(dropped_no_fitting_truncation=True)
        return _EntryFit(entry=best)


def _interleave(
    memory: Sequence[_PreparedContextEntry],
    topics: Sequence[_PreparedContextEntry],
    experiences: Sequence[_PreparedContextEntry],
) -> tuple[_PreparedContextEntry, ...]:
    """Preserve each family's rank while alternating Memory, Topic, Experience."""

    ordered: list[_PreparedContextEntry] = []
    for index in range(max(len(memory), len(topics), len(experiences))):
        if index < len(memory):
            ordered.append(memory[index])
        if index < len(topics):
            ordered.append(topics[index])
        if index < len(experiences):
            ordered.append(experiences[index])
    return tuple(ordered)


def _interleave_groups(groups: Sequence[Sequence[_Item]]) -> tuple[_Item, ...]:
    ordered: list[_Item] = []
    for index in range(max((len(group) for group in groups), default=0)):
        ordered.extend(group[index] for group in groups if index < len(group))
    return tuple(ordered)


def _render(entries: Sequence[_PreparedContextEntry]) -> str:
    envelope = {
        "trust": "untrusted_history",
        "items": [_render_entry(entry) for entry in entries],
    }
    encoded = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
    return "\n\n".join((TRUST_POLICY, f"{_BEGIN_MARKER}\n{encoded}\n{_END_MARKER}"))


def _text_item(entry: _PreparedContextEntry) -> ContextTextItem:
    origin = entry.origin
    if isinstance(origin, MemoryEntryAddress):
        return ContextTextItem(
            artifact=origin.memory,
            content=entry.content,
            recall_rank=0,
            entry_id=origin.entry_id,
            entry_version_id=origin.entry_version_id,
        )
    if isinstance(origin, ArtifactAddress):
        return ContextTextItem(artifact=origin, content=entry.content, recall_rank=0)
    raise PreparedContextInvariantError("text-scope-missing")


def _render_entry(entry: _PreparedContextEntry) -> dict[str, object]:
    rendered: dict[str, object] = {
        "citation": entry.citation,
        "content": entry.content,
        "truncated": entry.truncated,
    }
    if entry.kind != "memory":
        rendered["kind"] = entry.kind
    return rendered


def _rendered_bytes(entries: Sequence[_PreparedContextEntry]) -> int:
    return len(_render(entries).encode("utf-8"))


def _truncate_utf8(text: str, byte_budget: int) -> str:
    prefix_budget = byte_budget - len(_ELLIPSIS.encode("utf-8"))
    encoded_prefix = text.encode("utf-8")[:prefix_budget]
    return f"{encoded_prefix.decode('utf-8', errors='ignore')}{_ELLIPSIS}"


# --- literal evidence fragments ---------------------------------------------------------

_FRAGMENT_TERM = re.compile(r"[a-z]+|\d+")
_FRAGMENT_SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*(?=\s|$)")
_FRAGMENT_SUFFIXES = (
    "ivities",
    "ivity",
    "ations",
    "ation",
    "ives",
    "ive",
    "ities",
    "ity",
    "ing",
    "ed",
    "ly",
    "ment",
    "ance",
    "ence",
    "able",
    "ible",
    "ness",
    "es",
    "s",
    "al",
    "er",
)
_FRAGMENT_MIN_CHARS = 16
_FRAGMENT_MAX_PAIR_CHARS = 1100
_FRAGMENT_STOP = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "from",
        "with",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "that",
        "this",
        "these",
        "those",
        "which",
        "who",
        "whom",
        "whose",
        "it",
        "its",
        "they",
        "them",
        "their",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "his",
        "her",
        "for",
        "not",
        "no",
        "but",
        "if",
        "then",
        "than",
        "also",
        "such",
        "may",
        "might",
        "can",
        "could",
        "should",
        "would",
        "will",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "there",
        "here",
        "what",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "very",
        "own",
        "same",
        "so",
        "just",
        "about",
        "into",
        "over",
        "after",
        "before",
        "under",
        "above",
        "between",
        "during",
        "through",
        "against",
        "because",
        "while",
    ],
)


@dataclass(slots=True)
class _FragmentCandidate:
    """One literal span of a recalled entry before budget fitting."""

    text: str
    start: int
    end: int
    address: MemoryEntryAddress
    order: int
    prior: float = 0.0
    keys: frozenset[int] = frozenset()
    bonus: float = 0.0


class _FragmentQuery:
    """Map the caller's query onto the literal vocabulary the candidates use."""

    def __init__(self, query: str) -> None:
        terms = tuple(
            dict.fromkeys(term for term in _fragment_terms(query) if len(term) > 1 and term not in _FRAGMENT_STOP)
        )
        self.terms = terms
        self.stems = tuple(_fragment_stem(term) for term in terms)
        self.bigrams = frozenset((index, index + 1) for index in range(len(terms) - 1))
        self.idf: tuple[float, ...] = tuple(1.0 for _ in terms)

    def analyse(self, text: str) -> tuple[frozenset[int], float]:
        """Return the query terms one text contains and its phrase bonus."""

        matched: list[frozenset[int]] = []
        keys: set[int] = set()
        for token in _fragment_terms(text):
            stem = _fragment_stem(token)
            indices = frozenset(
                index for index, query_stem in enumerate(self.stems) if _fragment_same(stem, query_stem)
            )
            matched.append(indices)
            keys |= indices
        bonus = 0.0
        for left, right in pairwise(matched):
            for first in left:
                for second in right:
                    if (first, second) in self.bigrams:
                        bonus += 0.35 * (self.idf[first] + self.idf[second])
        return frozenset(keys), bonus

    def keys(self, text: str) -> frozenset[int]:
        """Return the query terms this text literally contains."""

        return self.analyse(text)[0]

    def gain(self, keys: frozenset[int], covered: set[int]) -> float:
        return sum(self.idf[index] for index in keys if index not in covered)

    def score(self, candidates: Sequence[_FragmentCandidate]) -> None:
        """Weight each query term by how rare it is across the offered candidates."""

        measured = [self.analyse(candidate.text[candidate.start : candidate.end]) for candidate in candidates]
        document_frequency = [0] * len(self.terms)
        for keys, _ in measured:
            for index in keys:
                document_frequency[index] += 1
        total = len(candidates)
        self.idf = tuple(log(1 + (total + 1) / (frequency + 1)) for frequency in document_frequency)
        for candidate, (keys, _) in zip(candidates, measured, strict=True):
            candidate.keys = keys
        for candidate in candidates:
            _, candidate.bonus = self.analyse(candidate.text[candidate.start : candidate.end])


def _fragment_candidates(
    memory_candidates: Sequence[PreparedMemoryCandidates],
    current_scope_id: str | None,
) -> list[_FragmentCandidate]:
    """List one candidate per sentence span, keeping exact entry text and citation identity."""

    candidates: list[_FragmentCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for group in memory_candidates:
        scope_id = group.scope_id or current_scope_id or ""
        if not scope_id:
            continue
        for rank, hit in enumerate(group.hits):
            identity = (scope_id, hit.entry_id, hit.entry_version_id)
            if identity in seen or not hit.text.strip():
                continue
            seen.add(identity)
            # Retrieval already ordered these hits, so an earlier one is a weak prior on
            # holding the evidence; it only breaks near-ties in span scoring.
            prior = 0.2 / (1 + rank)
            address = MemoryEntryAddress(
                memory=ArtifactAddress(scope_id=scope_id, artifact=hit.memory_ref),
                entry_id=hit.entry_id,
                entry_version_id=hit.entry_version_id,
            )
            for start, end in _span_runs(hit.text):
                candidates.append(
                    _FragmentCandidate(
                        text=hit.text,
                        start=start,
                        end=end,
                        address=address,
                        order=len(candidates),
                        prior=prior,
                    )
                )
    return candidates


def _fit_fragment(
    candidate: _FragmentCandidate,
    selected: Sequence[PreparedContextFragment],
    max_bytes: int,
) -> PreparedContextFragment | None:
    """Return the widest start-aligned literal span of one candidate that still fits."""

    whole = _fragment(candidate, candidate.start, candidate.end)
    if _fragments_within((*selected, whole), max_bytes):
        return whole
    low = candidate.start + _FRAGMENT_MIN_CHARS
    high = candidate.end - 1
    best: int | None = None
    while low <= high:
        middle = (low + high) // 2
        if _fragments_within((*selected, _fragment(candidate, candidate.start, middle)), max_bytes):
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    return None if best is None else _fragment(candidate, candidate.start, best)


def _fragment(candidate: _FragmentCandidate, start: int, end: int) -> PreparedContextFragment:
    return PreparedContextFragment(
        address=candidate.address,
        start=start,
        end=end,
        text=candidate.text[start:end],
    )


def _address_key(address: MemoryEntryAddress) -> tuple[str, str, str, str]:
    return (address.memory.scope_id, address.memory.artifact.artifact_id, address.entry_id, address.entry_version_id)


def _extend_fragments(
    fragments: Sequence[PreparedContextFragment],
    candidates: Sequence[_FragmentCandidate],
    max_bytes: int,
) -> tuple[PreparedContextFragment, ...]:
    """Spend the remaining budget on neighbouring text of the entries already selected.

    The delivered bytes are measured against a recall budget, not a precision budget, so
    text adjacent to the best matching spans is worth sending until the budget closes.
    """

    if not fragments:
        return ()
    texts: dict[tuple[str, str, str, str], str] = {}
    for candidate in candidates:
        texts.setdefault(_address_key(candidate.address), candidate.text)
    spans = {key: _sentence_spans(text) for key, text in texts.items()}
    working = _merge_fragments(list(fragments), texts)
    while True:
        extended = False
        for index, fragment in enumerate(working):
            key = _address_key(fragment.address)
            text = texts[key]
            for start, end in _extension_options(fragment, spans[key]):
                trial = replace(fragment, start=start, end=end, text=text[start:end])
                merged = _merge_fragments([*working[:index], trial, *working[index + 1 :]], texts)
                if _fragments_within(merged, max_bytes):
                    working = merged
                    extended = True
                    break
            if extended:
                break
        if not extended:
            return tuple(working)


def _extension_options(
    fragment: PreparedContextFragment,
    sentence_spans: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    """List the one-sentence extensions around one fragment, nearest boundary first."""

    next_end = next((end for _, end in sentence_spans if end > fragment.end), None)
    previous_start = next((start for start, _ in reversed(sentence_spans) if start < fragment.start), None)
    options: list[tuple[int, int]] = []
    if next_end is not None:
        options.append((fragment.start, next_end))
    if previous_start is not None:
        options.append((previous_start, fragment.end))
    if next_end is not None and previous_start is not None:
        options.append((previous_start, next_end))
    return options


def _merge_fragments(
    fragments: list[PreparedContextFragment],
    texts: dict[tuple[str, str, str, str], str],
) -> list[PreparedContextFragment]:
    """Join touching spans of one entry so a citation is never paid twice for one region."""

    grouped: dict[tuple[str, str, str, str], list[PreparedContextFragment]] = {}
    order: list[tuple[str, str, str, str]] = []
    for fragment in fragments:
        key = _address_key(fragment.address)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(fragment)
    merged: list[PreparedContextFragment] = []
    for key in order:
        text = texts[key]
        current = min(grouped[key], key=lambda item: item.start)
        for item in sorted(grouped[key], key=lambda fragment: fragment.start)[1:]:
            if item.start <= current.end + 1:
                end = max(current.end, item.end)
                current = replace(current, end=end, text=text[current.start : end])
            else:
                merged.append(current)
                current = item
        merged.append(current)
    return merged


def _fragments_within(fragments: Sequence[PreparedContextFragment], max_bytes: int) -> bool:
    return len(_render_fragments(fragments).encode("utf-8")) <= max_bytes


def _render_fragments(fragments: Sequence[PreparedContextFragment]) -> str:
    envelope = {
        "trust": "untrusted_history",
        "items": [
            {
                "citation": _fragment_citation(fragment.address),
                "content": fragment.text,
                "start": fragment.start,
                "end": fragment.end,
            }
            for fragment in fragments
        ],
    }
    encoded = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
    return "\n\n".join((TRUST_POLICY, f"{_BEGIN_MARKER}\n{encoded}\n{_END_MARKER}"))


def _fragment_citation(address: MemoryEntryAddress) -> dict[str, object]:
    return {
        "scope_id": address.memory.scope_id,
        "memory_ref": address.memory.artifact.model_dump(mode="json"),
        "entry_id": address.entry_id,
        "entry_version_id": address.entry_version_id,
    }


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Split entry text into contiguous sentence spans whose offsets address the original text.

    A span keeps the separator whitespace that follows its sentence so a delivered fragment
    stays a superset of the same sentence an annotator would point at.
    """

    spans: list[tuple[int, int]] = []
    start = 0
    for match in _FRAGMENT_SENTENCE_END.finditer(text):
        if text[start : match.end()].strip():
            spans.append((start, match.end()))
        start = match.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def _span_runs(text: str) -> list[tuple[int, int]]:
    """Offer single sentences and adjacent pairs so one item can carry more evidence."""

    sentences = _sentence_spans(text)
    runs = list(sentences)
    for index in range(len(sentences) - 1):
        start, end = sentences[index][0], sentences[index + 1][1]
        if end - start <= _FRAGMENT_MAX_PAIR_CHARS:
            runs.append((start, end))
    return runs


def _fragment_terms(text: str) -> tuple[str, ...]:
    return tuple(_FRAGMENT_TERM.findall(text.casefold()))


def _fragment_stem(term: str) -> str:
    """Fold one surface form onto the stem the query and candidate text compare by."""

    if term.endswith("ies") and len(term) - 3 >= 4:
        return f"{term[:-3]}y"
    for suffix in _FRAGMENT_SUFFIXES:
        if term.endswith(suffix) and len(term) - len(suffix) >= 4:
            return term[: -len(suffix)]
    return term


def _fragment_same(left: str, right: str) -> bool:
    """Treat a shared stem, or one stem extending the other, as the same word family."""

    if left == right:
        return True
    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    return len(shorter) >= 5 and longer.startswith(shorter)
