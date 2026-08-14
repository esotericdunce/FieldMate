from __future__ import annotations

import logging
import os
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, NamedTuple

from qdrant_client import models

from fieldmate.brain.qdrant.repository import QdrantMemoryRepository
from fieldmate.brain.retrieval.context import (
    DiagnosticContext,
    RetrievedMemory,
)
from fieldmate.brain.retrieval.evidence import Evidence

from dataclasses import dataclass
from typing import Any, NamedTuple


@dataclass
class SemanticCacheHit:
    """
    Result of a successful semantic cache lookup.

    Supports 2-tuple unpacking:
        context, score = hit
    As well as named attribute access:
        hit.context, hit.score, hit.response_text, hit.query
    """
    context: DiagnosticContext
    score: float
    response_text: str | None = None
    query: str = ""

    def __iter__(self):
        yield self.context
        yield self.score

    def __getitem__(self, item: int | str):
        if item == 0 or item == "context":
            return self.context
        elif item == 1 or item == "score":
            return self.score
        elif item == "response_text":
            return self.response_text
        elif item == "query":
            return self.query
        raise IndexError("SemanticCacheHit index out of range")


logger = logging.getLogger("fieldmate.brain.retrieval.semantic_cache")


def serialize_context(context: DiagnosticContext) -> dict[str, Any]:
    """Serialize a DiagnosticContext into a JSON-friendly dict payload."""
    def _ser_ev(e: Evidence) -> dict[str, Any]:
        return {
            "evidence_id": getattr(e, "evidence_id", ""),
            "memory_id": getattr(e, "memory_id", ""),
            "memory_type": getattr(e, "memory_type", ""),
            "content": getattr(e, "content", ""),
            "source": getattr(e, "source", "qdrant"),
            "equipment_model": getattr(e, "equipment_model", None),
            "fault_codes": list(getattr(e, "fault_codes", ())),
            "relevance_score": getattr(e, "relevance_score", 1.0),
            "confidence": getattr(e, "confidence", 1.0),
            "verification_status": getattr(e, "verification_status", "unverified"),
            "provenance": getattr(e, "provenance", "semantic_cache"),
            "relation": getattr(e, "relation", "supporting"),
            "case_reference": getattr(e, "case_reference", None),
            "retrieval_mode": getattr(e, "retrieval_mode", "semantic_cache"),
        }

    return {
        "evidence": [_ser_ev(e) for e in getattr(context, "evidence", ())],
        "supporting": [_ser_ev(e) for e in getattr(context, "supporting", ())],
        "contradicting": [_ser_ev(e) for e in getattr(context, "contradicting", ())],
        "neutral": [_ser_ev(e) for e in getattr(context, "neutral", ())],
        "procedures": [_ser_ev(e) for e in getattr(context, "procedures", ())],
        "past_cases": [_ser_ev(e) for e in getattr(context, "past_cases", ())],
        "resolutions": [_ser_ev(e) for e in getattr(context, "resolutions", ())],
        "token_budget": getattr(context, "token_budget", 4000),
    }


def deserialize_context(data: dict[str, Any]) -> DiagnosticContext:
    """Reconstruct a DiagnosticContext from a payload dict."""
    def _deser_ev(e: dict[str, Any]) -> Evidence:
        return Evidence(
            evidence_id=e.get("evidence_id", f"ev_{uuid.uuid4().hex[:8]}"),
            memory_id=e.get("memory_id", f"mem_{uuid.uuid4().hex[:8]}"),
            memory_type=e.get("memory_type", "resolution"),
            content=e.get("content", ""),
            source=e.get("source", "qdrant"),
            equipment_model=e.get("equipment_model"),
            fault_codes=tuple(e.get("fault_codes", ())),
            relevance_score=float(e.get("relevance_score", 1.0)),
            confidence=float(e.get("confidence", 1.0)),
            verification_status=e.get("verification_status", "unverified"),
            provenance=e.get("provenance", "semantic_cache"),
            relation=e.get("relation", "supporting"),
            case_reference=e.get("case_reference"),
            retrieval_mode=e.get("retrieval_mode", "semantic_cache"),
        )

    ev_list = list(data.get("evidence", []))
    if not ev_list and "memories" in data:
        for m in data.get("memories", []):
            ev_list.append({
                "evidence_id": f"ev_{m.get('memory_id', uuid.uuid4().hex[:8])}",
                "memory_id": m.get("memory_id", f"mem_{uuid.uuid4().hex[:8]}"),
                "memory_type": m.get("memory_type", "resolution"),
                "content": m.get("content", ""),
                "source": m.get("source", "qdrant"),
                "equipment_model": m.get("equipment_model"),
                "fault_codes": m.get("fault_codes", []),
                "relevance_score": m.get("score", 1.0),
                "confidence": m.get("confidence", 1.0),
                "verification_status": m.get("verification_status", "unverified"),
                "provenance": m.get("provenance", "semantic_cache"),
                "relation": m.get("relation", "supporting"),
            })

    evidence = tuple(_deser_ev(e) for e in ev_list)

    return DiagnosticContext(
        evidence=evidence,
    )


class QdrantSemanticCache:
    """
    Qdrant-backed, per-user isolated, context-guarded semantic cache layer.

    Architecture:

        Query + User Identity (owner_id) + Hardware Context
                               │
                               v
                    SemanticCache.lookup()
                               │
                     +---------+---------+
                     |                   |
                  CACHE HIT          CACHE MISS
                  (similarity         (fallthrough to
                   >= threshold)       prefetch & knowledge base)
                     |                   |
                     v                   v
                  Context           Qdrant Search
                  returned               │
                  <10ms                  v
                                    Store in cache

    Guarantees:
    - True Per-User Isolation: filtered by owner_id.
    - Full Context Guarding: filters by equipment_model, equipment_family,
      equipment_serial, system, subsystem, component, fault_code, verified_only.
    - TTL / Staleness Expiration: entries older than ttl_seconds are invalidated.
    - Defensive Error Handling: Qdrant/network errors are swallowed cleanly.
    """

    def __init__(
        self,
        repository: QdrantMemoryRepository,
        *,
        collection_name: str = "fieldmate_semantic_cache",
        threshold: float = 0.90,
        ttl_seconds: float = 86400.0,
        enabled: bool = True,
    ) -> None:
        self.repository = repository
        self.collection_name = collection_name
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        self.enabled = enabled
        self._initialized = False
        self._background_tasks: set[Any] = set()
        self._memory_cache: OrderedDict[str, tuple[float, SemanticCacheHit]] = OrderedDict()
        self._memory_cache_max_size: int = 256

    def _in_memory_key(
        self,
        query: str,
        owner_id: str | None = None,
        equipment_model: str | None = None,
        equipment_family: str | None = None,
        equipment_serial: str | None = None,
        system: str | None = None,
        subsystem: str | None = None,
        component: str | None = None,
        fault_code: str | None = None,
        verified_only: bool = False,
    ) -> str:
        norm_q = " ".join(query.strip().lower().split())
        return (
            f"{owner_id or 'none'}|"
            f"{equipment_model or 'none'}|"
            f"{equipment_family or 'none'}|"
            f"{equipment_serial or 'none'}|"
            f"{system or 'none'}|"
            f"{subsystem or 'none'}|"
            f"{component or 'none'}|"
            f"{fault_code or 'none'}|"
            f"{verified_only}|"
            f"{norm_q}"
        )

    async def ensure_collection(self) -> None:
        """Create the semantic cache collection and payload schema indexes in Qdrant."""
        if not self.enabled:
            return

        if self._initialized:
            return

        try:
            collections = await self.repository.client.get_collections()
            collection_names = {c.name for c in collections.collections}

            if self.collection_name not in collection_names:
                logger.info(
                    ">>> CREATING SEMANTIC CACHE COLLECTION: %s",
                    self.collection_name,
                )
                await self.repository.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        self.repository.config.dense_vector_name: models.VectorParams(
                            size=self.repository.config.dense_vector_size,
                            distance=models.Distance.COSINE,
                        )
                    },
                )

            # Ensure payload schema indexes exist for all filter attributes
            index_specs = [
                ("owner_id", models.PayloadSchemaType.KEYWORD),
                ("equipment_model", models.PayloadSchemaType.KEYWORD),
                ("equipment_family", models.PayloadSchemaType.KEYWORD),
                ("equipment_serial", models.PayloadSchemaType.KEYWORD),
                ("system", models.PayloadSchemaType.KEYWORD),
                ("subsystem", models.PayloadSchemaType.KEYWORD),
                ("component", models.PayloadSchemaType.KEYWORD),
                ("fault_code", models.PayloadSchemaType.KEYWORD),
                ("verified_only", models.PayloadSchemaType.BOOL),
                ("created_at", models.PayloadSchemaType.INTEGER),
            ]

            for field_name, schema_type in index_specs:
                try:
                    await self.repository.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field_name,
                        field_schema=schema_type,
                    )
                except Exception:
                    # Index might already exist; safely ignore
                    pass

            self._initialized = True
            logger.info(">>> SEMANTIC CACHE READY: %s", self.collection_name)
        except Exception as exc:
            logger.warning("Failed to initialize semantic cache collection: %s", exc)

    def _build_filter(
        self,
        *,
        owner_id: str | None = None,
        equipment_model: str | None = None,
        equipment_family: str | None = None,
        equipment_serial: str | None = None,
        system: str | None = None,
        subsystem: str | None = None,
        component: str | None = None,
        fault_code: str | None = None,
        verified_only: bool = False,
    ) -> models.Filter | None:
        conditions: list[models.Condition] = []

        # TTL Range Filter
        min_created_at = int(time.time() - self.ttl_seconds)
        conditions.append(
            models.FieldCondition(
                key="created_at",
                range=models.Range(gte=min_created_at),
            )
        )

        if owner_id:
            conditions.append(
                models.FieldCondition(
                    key="owner_id",
                    match=models.MatchValue(value=owner_id),
                )
            )

        # Context-Guarded Soft Filters: Match exact value OR general (null/empty)
        soft_fields = [
            ("equipment_model", equipment_model),
            ("equipment_family", equipment_family),
            ("equipment_serial", equipment_serial),
            ("system", system),
            ("subsystem", subsystem),
            ("component", component),
            ("fault_code", fault_code),
        ]

        for field_name, value in soft_fields:
            if value:
                conditions.append(
                    models.Filter(
                        should=[
                            models.FieldCondition(
                                key=field_name,
                                match=models.MatchValue(value=value),
                            ),
                            models.IsEmptyCondition(
                                is_empty=models.PayloadField(key=field_name)
                            ),
                        ]
                    )
                )

        if verified_only:
            conditions.append(
                models.FieldCondition(
                    key="verified_only",
                    match=models.MatchValue(value=True),
                )
            )

        return models.Filter(must=conditions)

    async def lookup(
        self,
        query: str,
        *,
        owner_id: str | None = None,
        equipment_model: str | None = None,
        equipment_family: str | None = None,
        equipment_serial: str | None = None,
        system: str | None = None,
        subsystem: str | None = None,
        component: str | None = None,
        fault_code: str | None = None,
        verified_only: bool = False,
    ) -> SemanticCacheHit | None:
        """
        Check semantic cache for a matching query within full user & context constraints.

        Returns SemanticCacheHit if similarity >= threshold and valid, or None on cache miss/error.
        """
        if not self.enabled or not query.strip():
            return None

        # 1. In-Memory LRU Fast Path (<0.1ms)
        mem_key = self._in_memory_key(
            query,
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
        if mem_key in self._memory_cache:
            ts, cached_hit = self._memory_cache[mem_key]
            if time.time() - ts <= self.ttl_seconds:
                self._memory_cache.move_to_end(mem_key)
                logger.info(
                    ">>> IN-MEMORY SEMANTIC CACHE HIT (0.1ms) owner=%s query=%r",
                    owner_id,
                    query,
                )
                return cached_hit
            else:
                del self._memory_cache[mem_key]

        if not self._initialized:
            await self.ensure_collection()

        try:
            query_filter = self._build_filter(
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

            result = await self.repository.client.query_points(
                collection_name=self.collection_name,
                query=models.Document(
                    text=query,
                    model=self.repository.config.dense_model,
                ),
                using=self.repository.config.dense_vector_name,
                query_filter=query_filter,
                limit=1,
                with_payload=True,
                with_vectors=False,
            )

            if not result.points:
                return None

            best_point = result.points[0]
            score = float(best_point.score)

            if score < self.threshold:
                logger.debug(
                    ">>> SEMANTIC CACHE MISS score=%.4f threshold=%.4f query=%r",
                    score,
                    self.threshold,
                    query,
                )
                return None

            payload = getattr(best_point, "payload", {}) or {}
            created_at = payload.get("created_at", 0)

            # Defensive TTL expiration check
            if time.time() - created_at > self.ttl_seconds:
                logger.info(">>> SEMANTIC CACHE EXPIRED id=%s", best_point.id)
                try:
                    await self.repository.client.delete(
                        collection_name=self.collection_name,
                        points_selector=models.PointIdsList(points=[best_point.id]),
                        wait=False,
                    )
                except Exception:
                    pass
                return None

            context_data = payload.get("context")
            if not context_data:
                return None

            context = deserialize_context(context_data)
            response_text = payload.get("response_text")

            # Defensive verified_only check
            if verified_only:
                unverified = any(
                    e.verification_status != "verified" for e in context.evidence
                )
                if unverified:
                    logger.debug(
                        ">>> SEMANTIC CACHE MISS (unverified context on verified_only request)"
                    )
                    return None

            hit_count = payload.get("hit_count", 1) + 1

            # Update hit_count and last_used_at asynchronously in Qdrant
            try:
                await self.repository.client.set_payload(
                    collection_name=self.collection_name,
                    payload={"hit_count": hit_count, "last_used_at": int(time.time())},
                    points=[best_point.id],
                    wait=False,
                )
            except Exception:
                pass

            logger.info(
                ">>> SEMANTIC CACHE HIT score=%.4f owner=%s hit_count=%d query=%r",
                score,
                owner_id,
                hit_count,
                query,
            )
            hit_obj = SemanticCacheHit(
                context=context,
                score=score,
                response_text=response_text,
                query=query,
            )

            # Store in in-memory LRU cache
            self._memory_cache[mem_key] = (time.time(), hit_obj)
            if len(self._memory_cache) > self._memory_cache_max_size:
                self._memory_cache.popitem(last=False)

            return hit_obj

        except Exception as exc:
            logger.warning("Semantic cache lookup failed (falling through): %s", exc)
            return None

        except Exception as exc:
            logger.warning("Semantic cache lookup failed (falling through): %s", exc)
            return None

    async def store(
        self,
        query: str,
        context: DiagnosticContext | str | None = None,
        *,
        response_text: str | None = None,
        owner_id: str | None = None,
        equipment_model: str | None = None,
        equipment_family: str | None = None,
        equipment_serial: str | None = None,
        system: str | None = None,
        subsystem: str | None = None,
        component: str | None = None,
        fault_code: str | None = None,
        verified_only: bool = False,
    ) -> str | None:
        """
        Store a successful retrieval context or generated response in the semantic cache.
        """
        if not self.enabled or not query.strip():
            return None

        if not self._initialized:
            await self.ensure_collection()

        if isinstance(context, str):
            if not response_text:
                response_text = context
            context = DiagnosticContext()
        elif context is None:
            context = DiagnosticContext()

        # Build fallback context from response_text if retrieval context is empty
        if not getattr(context, "evidence", ()) and response_text:
            ev_id = f"ev_cached_{uuid.uuid4().hex[:8]}"
            mem_id = f"cached_{uuid.uuid4().hex[:8]}"
            evidence_obj = Evidence(
                evidence_id=ev_id,
                memory_id=mem_id,
                memory_type="resolution",
                content=response_text,
                source="semantic_cache",
                equipment_model=equipment_model,
                fault_codes=(fault_code,) if fault_code else (),
                relevance_score=1.0,
                confidence=0.95,
                verification_status="verified",
                provenance="semantic_cache",
                relation="supporting",
                retrieval_mode="semantic_cache",
            )
            context = DiagnosticContext(
                evidence=(evidence_obj,),
            )

        if not context.evidence:
            return None

        try:
            point_id = str(uuid.uuid4())
            payload = {
                "query": query,
                "owner_id": owner_id,
                "equipment_model": equipment_model,
                "equipment_family": equipment_family,
                "equipment_serial": equipment_serial,
                "system": system,
                "subsystem": subsystem,
                "component": component,
                "fault_code": fault_code,
                "verified_only": verified_only,
                "response_text": response_text,
                "context": serialize_context(context),
                "hit_count": 1,
                "created_at": int(time.time()),
                "last_used_at": int(time.time()),
            }

            point = models.PointStruct(
                id=point_id,
                vector={
                    self.repository.config.dense_vector_name: models.Document(
                        text=query,
                        model=self.repository.config.dense_model,
                    )
                },
                payload=payload,
            )

            await self.repository.client.upsert(
                collection_name=self.collection_name,
                points=[point],
                wait=True,
            )

            logger.info(
                ">>> SEMANTIC CACHE STORED query=%r owner=%s id=%s",
                query,
                owner_id,
                point_id,
            )

            # Update in-memory LRU cache immediately
            mem_key = self._in_memory_key(
                query,
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
            self._memory_cache[mem_key] = (
                time.time(),
                SemanticCacheHit(
                    context=context,
                    score=1.0,
                    response_text=response_text,
                    query=query,
                ),
            )
            if len(self._memory_cache) > self._memory_cache_max_size:
                self._memory_cache.popitem(last=False)

            return point_id

        except Exception as exc:
            logger.warning("Failed to store query in semantic cache: %s", exc)
            return None

    def store_background(
        self,
        query: str,
        context: DiagnosticContext | str | None = None,
        *,
        response_text: str | None = None,
        owner_id: str | None = None,
        equipment_model: str | None = None,
        equipment_family: str | None = None,
        equipment_serial: str | None = None,
        system: str | None = None,
        subsystem: str | None = None,
        component: str | None = None,
        fault_code: str | None = None,
        verified_only: bool = False,
    ) -> None:
        """Launch store() as a strongly-referenced background task to avoid GC cancellation."""
        if not self.enabled or not query.strip():
            return
        import asyncio
        task = asyncio.create_task(
            self.store(
                query,
                context,
                response_text=response_text,
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
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def invalidate_owner(self, owner_id: str | None = None) -> bool:
        """Clear cached queries for a user (e.g. after a new confirmed resolution is saved)."""
        if not self.enabled:
            return True
        try:
            # Clear in-memory LRU cache entries for this owner
            prefix = f"{owner_id or 'none'}|"
            keys_to_del = [k for k in self._memory_cache if k.startswith(prefix) or not owner_id]
            for k in keys_to_del:
                self._memory_cache.pop(k, None)

            filter_cond = None
            if owner_id:
                filter_cond = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="owner_id",
                            match=models.MatchValue(value=owner_id),
                        )
                    ]
                )
            await self.repository.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=filter_cond) if filter_cond else models.FilterSelector(filter=models.Filter()),
                wait=True,
            )
            logger.info(">>> SEMANTIC CACHE INVALIDATED for owner=%s", owner_id)
            return True
        except Exception as exc:
            logger.warning("Failed to invalidate semantic cache for owner=%s: %s", owner_id, exc)
            return False
