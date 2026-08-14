from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from fieldmate.brain.qdrant.repository import (
    QdrantMemoryRepository,
)
from fieldmate.brain.state.models import DiagnosticState

from .context import (
    DiagnosticContext,
    context_from_evidence,
)
from .context_intelligence import (
    ContextIntelligence,
)
from .evidence import (
    normalize_evidence,
)
from .planner import (
    RetrievalMode,
    RetrievalPlan,
    plan_retrieval,
)


# ============================================================
# RESULT
# ============================================================


@dataclass(frozen=True)
class RetrievalResult:
    """Complete result of one retrieval operation."""

    context: DiagnosticContext
    plan: RetrievalPlan
    latency_ms: float

    timed_out: bool = False
    prefetched: bool = False

    # Retrieval, not the voice layer, owns this decision.
    # True means the returned evidence is strong enough to
    # require grounded reasoning.
    relevant: bool = False

    cached_response: str | None = None
    cached: bool = False


# ============================================================
# PREFETCH ENTRY
# ============================================================


@dataclass
class _PrefetchEntry:
    query: str
    task: asyncio.Task[Any]
    created_at: float
    scope_key: str
    plan_mode: str
    limit: int


# ============================================================
# ORCHESTRATOR
# ============================================================


class RetrievalOrchestrator:
    """
    Adaptive, bounded retrieval coordinator.

    Architecture:

        query
          |
          v
        Planner
          |
          +---- dense
          +---- sparse
          +---- hybrid
          |
          v
        Qdrant
          |
          v
        raw points
          |
          v
        Evidence normalization
          |
          v
        Context Intelligence
          |
          v
        DiagnosticContext
          |
          v
        Brain


    Guarantees:

    - retrieval never decides diagnosis
    - retrieval never modifies diagnostic state
    - hot-path retrieval is time bounded
    - speculative prefetch never blocks the hot path
    - contradictory evidence is preserved
    - raw Qdrant representations do not leak into Brain
    - Qdrant failures degrade into an empty context
    - stale prefetches are discarded
    """

    def __init__(
        self,
        repository: QdrantMemoryRepository,
        *,
        timeout_ms: int = 900,
        prefetch_timeout_ms: int = 3000,
        prefetch_ttl_ms: int = 5000,
        relevance_threshold: float = 0.35,
        context_intelligence: ContextIntelligence | None = None,
        semantic_cache: Any | None = None,
    ) -> None:

        if timeout_ms <= 0:
            raise ValueError(
                "timeout_ms must be > 0"
            )

        if prefetch_timeout_ms <= 0:
            raise ValueError(
                "prefetch_timeout_ms must be > 0"
            )

        if prefetch_ttl_ms <= 0:
            raise ValueError(
                "prefetch_ttl_ms must be > 0"
            )

        if not 0.0 <= relevance_threshold <= 1.0:
            raise ValueError(
                "relevance_threshold must be between 0 and 1"
            )

        self.repository = repository

        self.timeout_seconds = (
            timeout_ms / 1000.0
        )

        self.prefetch_timeout_seconds = (
            prefetch_timeout_ms / 1000.0
        )

        self.prefetch_ttl_seconds = (
            prefetch_ttl_ms / 1000.0
        )

        self.relevance_threshold = relevance_threshold

        self.context_intelligence = (
            context_intelligence
            or ContextIntelligence()
        )

        self.semantic_cache = semantic_cache

        self._prefetch: dict[
            str,
            _PrefetchEntry,
        ] = {}

        self._lock = asyncio.Lock()

        self._closed = False

        # Only a tiny number of speculative entries are useful.
        # Keeping the latest entry per retrieval scope avoids
        # wasting Qdrant work on obsolete STT partials.
        self._max_prefetch_entries = 8

    # ========================================================
    # MAIN RETRIEVAL
    # ========================================================

    async def retrieve(
        self,
        query: str,
        *,
        state: DiagnosticState | None = None,
        equipment_model: str | None = None,
        equipment_family: str | None = None,
        equipment_serial: str | None = None,
        system: str | None = None,
        subsystem: str | None = None,
        component: str | None = None,
        fault_code: str | None = None,
        memory_types: list[str] | None = None,
        statuses: list[str] | None = None,
        scope: str | None = None,
        owner_id: str | None = None,
        verified_only: bool = False,
        limit: int = 8,
    ) -> RetrievalResult:
        """
        Perform one bounded retrieval operation.

        `state` is optional for compatibility with callers that
        only have structured equipment/filter information.

        When state is supplied, ContextIntelligence performs
        state-aware evidence ranking and contradiction handling.
        """

        started = time.perf_counter()

        normalized_query = (
            " ".join(
                query.strip().split()
            )
        )

        if not normalized_query:
            return RetrievalResult(
                context=DiagnosticContext(),
                plan=plan_retrieval(""),
                latency_ms=0.0,
            )

        if limit < 1:
            limit = 1

        # ----------------------------------------------------
        # PLAN
        # ----------------------------------------------------

        plan = plan_retrieval(
            normalized_query,
            has_equipment_context=(
                equipment_model is not None
                or equipment_family is not None
                or equipment_serial is not None
            ),
            has_fault_context=(
                fault_code is not None
            ),
        )

        # ----------------------------------------------------
        # BUILD FILTER
        # ----------------------------------------------------

        query_filter = (
            self.repository.build_filter(
                equipment_model=equipment_model,
                equipment_family=equipment_family,
                equipment_serial=equipment_serial,
                system=system,
                subsystem=subsystem,
                component=component,
                fault_code=fault_code,
                memory_types=memory_types,
                statuses=statuses,
                scope=scope,
                owner_id=owner_id,
                verified_only=verified_only,
            )
        )

        # ----------------------------------------------------
        # SEMANTIC CACHE HIT
        # ----------------------------------------------------

        if self.semantic_cache and getattr(self.semantic_cache, "enabled", True):
            cached_result = await self.semantic_cache.lookup(
                normalized_query,
                owner_id=owner_id,
                equipment_model=equipment_model,
                equipment_family=equipment_family,
                equipment_serial=equipment_serial,
                system=system,
                subsystem=subsystem,
                component=component,
                fault_code=fault_code,
                verified_only=verified_only,
            )
            if cached_result is not None:
                cached_context = cached_result.context
                return RetrievalResult(
                    context=cached_context,
                    plan=plan,
                    latency_ms=(
                        time.perf_counter()
                        - started
                    ) * 1000.0,
                    prefetched=False,
                    relevant=self._context_is_relevant(cached_context) or bool(getattr(cached_result, "response_text", None)),
                    cached_response=getattr(cached_result, "response_text", None),
                    cached=True,
                )

        # ----------------------------------------------------
        # PREFETCH HIT
        # ----------------------------------------------------

        cache_key = self._cache_key(
            query=normalized_query,
            plan=plan,
            equipment_model=equipment_model,
            equipment_family=equipment_family,
            equipment_serial=equipment_serial,
            system=system,
            subsystem=subsystem,
            component=component,
            fault_code=fault_code,
            memory_types=memory_types,
            statuses=statuses,
            scope=scope,
            owner_id=owner_id,
            verified_only=verified_only,
        )

        prefetched = (
            await self._consume_completed_prefetch(
                cache_key
            )
        )

        # Primd-style convergence: a completed partial-query
        # retrieval can satisfy the final query without another
        # network round trip when the final transcript has only
        # grown in a semantically compatible way.
        if prefetched is None:
            prefetched = (
                await self._consume_compatible_prefetch(
                    query=normalized_query,
                    plan=plan,
                    scope_key=self._scope_key(
                        plan=plan,
                        equipment_model=equipment_model,
                        equipment_family=equipment_family,
                        equipment_serial=equipment_serial,
                        system=system,
                        subsystem=subsystem,
                        component=component,
                        fault_code=fault_code,
                        memory_types=memory_types,
                        statuses=statuses,
                        scope=scope,
                        owner_id=owner_id,
                        verified_only=verified_only,
                    ),
                )
            )

        if prefetched:
            context = self._build_context(
                prefetched,
                state=state,
                query=normalized_query,
                retrieval_mode=plan.mode.value,
                limit=limit,
                prefetched=True,
            )

            if self._context_is_relevant(context):
                return RetrievalResult(
                    context=context,
                    plan=plan,
                    latency_ms=(
                        time.perf_counter()
                        - started
                    ) * 1000.0,
                    prefetched=True,
                    relevant=True,
                )

        # ----------------------------------------------------
        # NORMAL BOUNDED RETRIEVAL
        # ----------------------------------------------------

        try:
            points = await asyncio.wait_for(
                self._execute(
                    plan=plan,
                    query=normalized_query,
                    query_filter=query_filter,
                    limit=limit,
                ),
                timeout=self.timeout_seconds,
            )


        except asyncio.TimeoutError:

            elapsed = (
                time.perf_counter()
                - started
            ) * 1000.0

            return RetrievalResult(
                context=DiagnosticContext(),
                plan=plan,
                latency_ms=elapsed,
                timed_out=True,
                relevant=False,
            )

        except asyncio.CancelledError:
            raise

        except Exception:
            # Retrieval failure must not crash the voice
            # conversation.
            #
            # The Brain can continue without historical memory.
            elapsed = (
                time.perf_counter()
                - started
            ) * 1000.0

            return RetrievalResult(
                context=DiagnosticContext(),
                plan=plan,
                latency_ms=elapsed,
                relevant=False,
            )

        # ----------------------------------------------------
        # BUILD DOMAIN CONTEXT
        # ----------------------------------------------------

        context = self._build_context(
            points,
            state=state,
            query=normalized_query,
            retrieval_mode=plan.mode.value,
            limit=limit,
        )

        relevant = self._context_is_relevant(
            context
        )

        if relevant and self.semantic_cache and getattr(self.semantic_cache, "enabled", True):
            if hasattr(self.semantic_cache, "store_background"):
                self.semantic_cache.store_background(
                    normalized_query,
                    context,
                    owner_id=owner_id,
                    equipment_model=equipment_model,
                    equipment_family=equipment_family,
                    equipment_serial=equipment_serial,
                    system=system,
                    subsystem=subsystem,
                    component=component,
                    fault_code=fault_code,
                    verified_only=verified_only,
                )
            else:
                asyncio.create_task(
                    self.semantic_cache.store(
                        normalized_query,
                        context,
                        owner_id=owner_id,
                        equipment_model=equipment_model,
                        equipment_family=equipment_family,
                        equipment_serial=equipment_serial,
                        system=system,
                        subsystem=subsystem,
                        component=component,
                        fault_code=fault_code,
                        verified_only=verified_only,
                    )
                )

        elapsed = (
            time.perf_counter()
            - started
        ) * 1000.0

        return RetrievalResult(
            context=context,
            plan=plan,
            latency_ms=elapsed,
            relevant=relevant,
        )

    # ========================================================
    # RELEVANCE
    # ========================================================

    def _context_is_relevant(
        self,
        context: DiagnosticContext,
    ) -> bool:
        """Decide relevance from normalized evidence scores."""

        evidence = tuple(
            getattr(context, "evidence", ()) or ()
        )

        if not evidence:
            return False

        threshold = getattr(
            self,
            "relevance_threshold",
            0.35,
        )

        strongest = 0.0

        for item in evidence:
            try:
                score = float(
                    getattr(
                        item,
                        "relevance_score",
                        0.0,
                    )
                )
                mode = getattr(item, "retrieval_mode", "")
                if mode == "hybrid" or score < 0.1:
                    # Qdrant Reciprocal Rank Fusion (RRF) scores are mathematically bounded
                    # in [0.016, 0.033]. Scale RRF scores to 0-1 range for relevance thresholding.
                    score = min(1.0, score * 30.0)
            except (TypeError, ValueError):
                score = 0.0

            strongest = max(strongest, score)

        return strongest >= threshold

    # ========================================================
    # CONTEXT CONSTRUCTION
    # ========================================================

    def _build_context(
        self,
        points: Any,
        *,
        state: DiagnosticState | None,
        query: str,
        retrieval_mode: str,
        limit: int,
        prefetched: bool = False,
        timed_out: bool = False,
    ) -> DiagnosticContext:
        """
        Convert Qdrant output into the application's domain
        context.

        Pipeline:

            Qdrant points
                ↓
            Evidence
                ↓
            ContextIntelligence
                ↓
            DiagnosticContext
        """

        evidence = normalize_evidence(
            list(points or []),
            state=state,
            retrieval_mode=retrieval_mode,
        )


        if not evidence:
            return DiagnosticContext()

        # ----------------------------------------------------
        # State-aware context intelligence.
        # ----------------------------------------------------

        if state is not None:

            return (
                self.context_intelligence
                .build_diagnostic_context(
                    evidence,
                    state,
                    query=query,
                    retrieval_mode=retrieval_mode,
                    prefetched=prefetched,
                    timed_out=timed_out,
                )
            )

        # ----------------------------------------------------
        # Compatibility path.
        #
        # Some callers may not have DiagnosticState yet.
        # Do not fabricate diagnostic state merely for
        # retrieval.
        # ----------------------------------------------------

        limited = list(
            evidence[:limit]
        )

        return context_from_evidence(
            limited,
            query=query,
            retrieval_mode=retrieval_mode,
            max_memories=limit,
            prefetched=prefetched,
            timed_out=timed_out,
        )

    # ========================================================
    # SPECULATIVE PREFETCH
    # ========================================================

    async def prefetch(
        self,
        query: str,
        *,
        equipment_model: str | None = None,
        equipment_family: str | None = None,
        equipment_serial: str | None = None,
        system: str | None = None,
        subsystem: str | None = None,
        component: str | None = None,
        fault_code: str | None = None,
        memory_types: list[str] | None = None,
        statuses: list[str] | None = None,
        scope: str | None = None,
        owner_id: str | None = None,
        verified_only: bool = False,
        limit: int = 8,
    ) -> None:
        """
        Start speculative retrieval.

        This method intentionally does not await the actual
        Qdrant request.

        The next retrieve() call may consume the completed
        result if its cache key matches.
        """

        if self._closed:
            return

        normalized_query = (
            " ".join(
                query.strip().split()
            )
        )

        if not normalized_query:
            return

        if limit < 1:
            limit = 1

        plan = plan_retrieval(
            normalized_query,
            has_equipment_context=(
                equipment_model is not None
                or equipment_family is not None
                or equipment_serial is not None
            ),
            has_fault_context=(
                fault_code is not None
            ),
        )

        query_filter = (
            self.repository.build_filter(
                equipment_model=equipment_model,
                equipment_family=equipment_family,
                equipment_serial=equipment_serial,
                system=system,
                subsystem=subsystem,
                component=component,
                fault_code=fault_code,
                memory_types=memory_types,
                statuses=statuses,
                scope=scope,
                owner_id=owner_id,
                verified_only=verified_only,
            )
        )

        cache_key = self._cache_key(
            query=normalized_query,
            plan=plan,
            equipment_model=equipment_model,
            equipment_family=equipment_family,
            equipment_serial=equipment_serial,
            system=system,
            subsystem=subsystem,
            component=component,
            fault_code=fault_code,
            memory_types=memory_types,
            statuses=statuses,
            scope=scope,
            owner_id=owner_id,
            verified_only=verified_only,
        )

        scope_key = self._scope_key(
            plan=plan,
            equipment_model=equipment_model,
            equipment_family=equipment_family,
            equipment_serial=equipment_serial,
            system=system,
            subsystem=subsystem,
            component=component,
            fault_code=fault_code,
            memory_types=memory_types,
            statuses=statuses,
            scope=scope,
            owner_id=owner_id,
            verified_only=verified_only,
        )

        async with self._lock:

            existing = self._prefetch.get(
                cache_key
            )

            if existing is not None:

                if not existing.task.done():
                    return

                self._prefetch.pop(
                    cache_key,
                    None,
                )

            # Supersede older partials for the same retrieval
            # scope. The latest transcript is the only one that
            # can meaningfully converge on the final utterance.
            for old_key, old_entry in tuple(
                self._prefetch.items()
            ):
                if old_entry.scope_key != scope_key:
                    continue
                if old_key == cache_key:
                    continue
                if not old_entry.task.done():
                    old_entry.task.cancel()
                self._prefetch.pop(
                    old_key,
                    None,
                )

            task = asyncio.create_task(
                self._prefetch_execute(
                    plan=plan,
                    query=normalized_query,
                    query_filter=query_filter,
                    limit=limit,
                )
            )

            self._prefetch[
                cache_key
            ] = _PrefetchEntry(
                query=normalized_query,
                task=task,
                created_at=time.perf_counter(),
                scope_key=scope_key,
                plan_mode=plan.mode.value,
                limit=limit,
            )

            # Hard bound speculative cache size.
            while len(self._prefetch) > self._max_prefetch_entries:
                oldest_key = min(
                    self._prefetch,
                    key=lambda key: self._prefetch[key].created_at,
                )
                oldest = self._prefetch.pop(
                    oldest_key
                )
                if not oldest.task.done():
                    oldest.task.cancel()

    # ========================================================
    # PREFETCH EXECUTION
    # ========================================================

    async def _prefetch_execute(
        self,
        *,
        plan: RetrievalPlan,
        query: str,
        query_filter: Any,
        limit: int,
    ) -> list[Any] | None:
        """
        Execute speculative retrieval under its own timeout.

        Errors are swallowed because prefetch is opportunistic.
        """

        try:

            result = await asyncio.wait_for(
                self._execute(
                    plan=plan,
                    query=query,
                    query_filter=query_filter,
                    limit=limit,
                ),
                timeout=(
                    self.prefetch_timeout_seconds
                ),
            )

            return result

        except asyncio.CancelledError:
            raise

        except Exception:
            return None

    # ========================================================
    # PREFETCH CONSUMPTION
    # ========================================================

    async def _consume_completed_prefetch(
        self,
        cache_key: str,
    ) -> list[Any] | None:
        """
        Consume a completed speculative retrieval.

        Critical invariant:

            NEVER wait for a still-running prefetch.

        If it isn't already complete, normal retrieval proceeds.
        """

        async with self._lock:

            entry = self._prefetch.get(
                cache_key
            )

            if entry is None:
                return None

            age = (
                time.perf_counter()
                - entry.created_at
            )

            # ------------------------------------------------
            # STALE
            # ------------------------------------------------

            if (
                age
                > self.prefetch_ttl_seconds
            ):

                if not entry.task.done():
                    entry.task.cancel()

                self._prefetch.pop(
                    cache_key,
                    None,
                )

                return None

            # ------------------------------------------------
            # STILL RUNNING
            # ------------------------------------------------

            if not entry.task.done():
                return None

            self._prefetch.pop(
                cache_key,
                None,
            )

        # ----------------------------------------------------
        # Retrieve result outside lock.
        # ----------------------------------------------------

        try:
            result = entry.task.result()

        except (
            asyncio.CancelledError,
            Exception,
        ):
            return None

        if not result:
            return None

        return result

    # ========================================================
    # COMPATIBLE PREFETCH CONSUMPTION
    # ========================================================

    async def _consume_compatible_prefetch(
        self,
        *,
        query: str,
        plan: RetrievalPlan,
        scope_key: str,
    ) -> list[Any] | None:
        """
        Consume a completed partial retrieval when the final
        transcript has converged on that partial.

        This is deliberately a cheap lexical convergence gate.
        It avoids downloading/embedding another Qdrant query on
        the final-turn critical path while refusing obviously
        unrelated partials.
        """

        async with self._lock:
            candidates = [
                entry
                for entry in self._prefetch.values()
                if entry.scope_key == scope_key
                and entry.plan_mode == plan.mode.value
                and entry.task.done()
            ]

            if not candidates:
                return None

            entry = max(
                candidates,
                key=lambda item: item.created_at,
            )

            age = (
                time.perf_counter()
                - entry.created_at
            )

            if age > self.prefetch_ttl_seconds:
                for key, value in tuple(self._prefetch.items()):
                    if value is entry:
                        self._prefetch.pop(key, None)
                return None

            similarity = self._query_convergence(
                entry.query,
                query,
            )

            if similarity < 0.55:
                return None

            for key, value in tuple(self._prefetch.items()):
                if value is entry:
                    self._prefetch.pop(key, None)
                    break

        try:
            return entry.task.result() or None
        except (asyncio.CancelledError, Exception):
            return None

    @staticmethod
    def _query_convergence(
        partial: str,
        final: str,
    ) -> float:
        """Cheap monotonic STT convergence score in [0, 1]."""

        left = tuple(dict.fromkeys(partial.lower().split()))
        right = tuple(dict.fromkeys(final.lower().split()))

        if not left or not right:
            return 0.0

        left_set = set(left)
        right_set = set(right)

        overlap = len(left_set & right_set) / len(left_set)

        # STT partials normally grow monotonically. Give that
        # pattern a strong bonus, but never allow a short prefix
        # to bypass the overlap requirement entirely.
        prefix_bonus = (
            0.15
            if len(right) >= len(left)
            and right[: len(left)] == left
            else 0.0
        )

        return min(1.0, overlap + prefix_bonus)

    # ========================================================
    # EXECUTION
    # ========================================================

    async def _execute(
        self,
        *,
        plan: RetrievalPlan,
        query: str,
        query_filter: Any,
        limit: int,
    ) -> list[Any]:
        """
        Execute exactly one retrieval strategy.

        Strategy selection belongs exclusively to Planner.
        """

        if plan.mode == RetrievalMode.SPARSE:

            result = (
                await self.repository.search_sparse(
                    query,
                    limit=limit,
                    query_filter=query_filter,
                )
            )

        elif plan.mode == RetrievalMode.HYBRID:

            result = (
                await self.repository.search_hybrid(
                    query,
                    limit=limit,
                    query_filter=query_filter,
                )
            )

        else:

            result = (
                await self.repository.search_dense(
                    query,
                    limit=limit,
                    query_filter=query_filter,
                )
            )

        return list(result or [])

    # ========================================================
    # CACHE SCOPE
    # ========================================================

    @staticmethod
    def _scope_key(
        *,
        plan: RetrievalPlan,
        equipment_model: str | None,
        equipment_family: str | None,
        equipment_serial: str | None,
        system: str | None,
        subsystem: str | None,
        component: str | None,
        fault_code: str | None,
        memory_types: list[str] | None,
        statuses: list[str] | None,
        scope: str | None,
        owner_id: str | None,
        verified_only: bool,
    ) -> str:
        return "|".join((
            plan.mode.value,
            (equipment_model or "").strip().lower(),
            (equipment_family or "").strip().lower(),
            (equipment_serial or "").strip().lower(),
            (system or "").strip().lower(),
            (subsystem or "").strip().lower(),
            (component or "").strip().lower(),
            (fault_code or "").strip().lower(),
            ",".join(sorted(str(v).strip().lower() for v in (memory_types or []) if str(v).strip())),
            ",".join(sorted(str(v).strip().lower() for v in (statuses or []) if str(v).strip())),
            (scope or "").strip().lower(),
            (owner_id or "").strip().lower(),
            "verified" if verified_only else "any",
        ))

    # ========================================================
    # CACHE KEY
    # ========================================================

    @staticmethod
    def _cache_key(
        *,
        query: str,
        plan: RetrievalPlan,
        equipment_model: str | None,
        equipment_family: str | None,
        equipment_serial: str | None,
        system: str | None,
        subsystem: str | None,
        component: str | None,
        fault_code: str | None,
        memory_types: list[str] | None,
        statuses: list[str] | None,
        scope: str | None,
        owner_id: str | None,
        verified_only: bool,
    ) -> str:
        """
        Build a deterministic cache key.

        Every retrieval-affecting filter is included.

        This prevents a dangerous situation where:

            same query + different equipment

        accidentally consumes the wrong prefetched result.
        """

        normalized_query = (
            " ".join(
                query.lower().split()
            )
        )

        def normalize_list(
            values: list[str] | None,
        ) -> str:
            if not values:
                return ""

            return ",".join(
                sorted(
                    str(value).strip().lower()
                    for value in values
                    if str(value).strip()
                )
            )

        parts = (
            plan.mode.value,
            normalized_query,
            (
                equipment_model
                or ""
            ).strip().lower(),
            (
                equipment_family
                or ""
            ).strip().lower(),
            (
                equipment_serial
                or ""
            ).strip().lower(),
            (
                system
                or ""
            ).strip().lower(),
            (
                subsystem
                or ""
            ).strip().lower(),
            (
                component
                or ""
            ).strip().lower(),
            (
                fault_code
                or ""
            ).strip().lower(),
            normalize_list(
                memory_types
            ),
            normalize_list(
                statuses
            ),
            (
                scope
                or ""
            ).strip().lower(),
            (
                owner_id
                or ""
            ).strip().lower(),
            "verified" if verified_only else "any",
        )

        return "|".join(parts)

    # ========================================================
    # SHUTDOWN
    # ========================================================

    async def close(self) -> None:
        """
        Cancel outstanding speculative retrievals.

        This method is safe to call more than once.
        """

        if self._closed:
            return

        self._closed = True

        async with self._lock:

            tasks = [
                entry.task
                for entry in self._prefetch.values()
                if not entry.task.done()
            ]

            self._prefetch.clear()

        if tasks:

            for task in tasks:
                task.cancel()

            await asyncio.gather(
                *tasks,
                return_exceptions=True,
            )