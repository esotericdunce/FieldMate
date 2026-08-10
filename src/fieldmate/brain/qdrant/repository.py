from __future__ import annotations

from typing import Any

from qdrant_client import (
    AsyncQdrantClient,
    models,
)

from fieldmate.brain.memory.identity import (
    memory_identity,
)

from fieldmate.brain.memory.models import (
    MemoryRecord,
)

from .config import QdrantConfig


class QdrantMemoryRepository:
    """
    Persistent Qdrant-backed repository for FieldMate memory.

    Architecture:

        MemoryManager
              |
              v
        QdrantMemoryRepository
              |
        +-----+------+----------------+
        |            |                |
        v            v                v
      Dense        Sparse          Hybrid
        |            |                |
        +------------+----------------+
                     |
                     v
                Qdrant Cloud

    Responsibilities:

        - Qdrant connection
        - collection initialization
        - collection schema validation
        - payload indexes
        - MemoryRecord serialization
        - memory upserts
        - dense retrieval
        - sparse/BM25 retrieval
        - hybrid retrieval
        - filtered retrieval

    This class does NOT decide:

        - whether a memory is true
        - whether a memory is verified
        - whether a memory should be promoted
        - whether a case is resolved
        - whether two memories contradict one another

    Those decisions belong to the memory/domain layer.

    Qdrant is persistence + retrieval infrastructure.
    """

    def __init__(
        self,
        config: QdrantConfig,
    ) -> None:

        self.config = config

        self.client = AsyncQdrantClient(
            url=config.url,
            api_key=config.api_key,
            cloud_inference=True,
            timeout=config.timeout_seconds,
        )

        self._initialized = False
        self._closed = False

    # =========================================================
    # COLLECTION INITIALIZATION
    # =========================================================

    async def ensure_collection(
        self,
    ) -> None:
        """
        Ensure the FieldMate memory collection exists and has
        the expected schema.

        This should happen during application startup, never
        inside the conversational hot path.
        """

        if self._closed:
            raise RuntimeError(
                "Qdrant repository is closed."
            )

        if self._initialized:
            return

        exists = await self.client.collection_exists(
            self.config.collection_name
        )

        if not exists:

            await self.client.create_collection(
                collection_name=(
                    self.config.collection_name
                ),

                vectors_config={
                    self.config.dense_vector_name:
                        models.VectorParams(
                            size=(
                                self.config
                                .dense_vector_size
                            ),
                            distance=(
                                models.Distance.COSINE
                            ),
                        )
                },

                sparse_vectors_config={
                    self.config.sparse_vector_name:
                        models.SparseVectorParams(
                            modifier=models.Modifier.IDF
                        )
                },
            )

        else:
            await self._validate_collection()

        await self._ensure_indexes()

        self._initialized = True

    async def _validate_collection(
        self,
    ) -> None:
        """
        Validate an existing collection.

        We deliberately fail instead of silently changing an
        incompatible collection.

        This protects existing FieldMate knowledge.
        """

        info = await self.client.get_collection(
            self.config.collection_name
        )

        vectors = info.config.params.vectors

        if not isinstance(vectors, dict):
            raise RuntimeError(
                "FieldMate memory collection does not "
                "use named vectors."
            )

        dense_config = vectors.get(
            self.config.dense_vector_name
        )

        if dense_config is None:
            raise RuntimeError(
                "FieldMate memory collection is missing "
                f"dense vector "
                f"'{self.config.dense_vector_name}'."
            )

        if (
            dense_config.size
            != self.config.dense_vector_size
        ):
            raise RuntimeError(
                "FieldMate dense vector size mismatch: "
                f"expected "
                f"{self.config.dense_vector_size}, "
                f"got {dense_config.size}."
            )

        sparse_vectors = (
            info.config.params.sparse_vectors
        )

        if sparse_vectors is None:
            raise RuntimeError(
                "FieldMate memory collection is missing "
                "sparse vectors."
            )

        if (
            self.config.sparse_vector_name
            not in sparse_vectors
        ):
            raise RuntimeError(
                "FieldMate memory collection is missing "
                f"sparse vector "
                f"'{self.config.sparse_vector_name}'."
            )

    # =========================================================
    # PAYLOAD INDEXES
    # =========================================================

    async def _ensure_indexes(
        self,
    ) -> None:
        """
        Ensure payload indexes needed by retrieval exist.
        """

        info = await self.client.get_collection(
            self.config.collection_name
        )

        existing_schema = (
            info.payload_schema or {}
        )

        keyword_fields = (
            "memory_type",
            "status",
            "equipment_family",
            "equipment_model",
            "equipment_serial",
            "system",
            "subsystem",
            "component",
            "scope",
            "owner_id",
            "fault_codes",
        )

        for field_name in keyword_fields:

            if field_name in existing_schema:
                continue

            await self.client.create_payload_index(
                collection_name=(
                    self.config.collection_name
                ),
                field_name=field_name,
                field_schema=(
                    models.PayloadSchemaType.KEYWORD
                ),
                wait=True,
            )

    # =========================================================
    # FILTER BUILDER
    # =========================================================

    @staticmethod
    def build_filter(
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
        include_deprecated: bool = False,
    ) -> models.Filter | None:
        """
        Build a Qdrant filter from domain constraints.

        All explicit constraints are ANDed together.

        Deprecated memories are excluded by default.
        """

        conditions: list[
            models.Condition
        ] = []

        # -----------------------------------------------------
        # EQUIPMENT
        # -----------------------------------------------------

        if equipment_model:

            conditions.append(
                models.FieldCondition(
                    key="equipment_model",
                    match=models.MatchValue(
                        value=equipment_model
                    ),
                )
            )

        if equipment_family:

            conditions.append(
                models.FieldCondition(
                    key="equipment_family",
                    match=models.MatchValue(
                        value=equipment_family
                    ),
                )
            )

        if equipment_serial:

            conditions.append(
                models.FieldCondition(
                    key="equipment_serial",
                    match=models.MatchValue(
                        value=equipment_serial
                    ),
                )
            )

        # -----------------------------------------------------
        # DIAGNOSTIC DOMAIN
        # -----------------------------------------------------

        if system:

            conditions.append(
                models.FieldCondition(
                    key="system",
                    match=models.MatchValue(
                        value=system
                    ),
                )
            )

        if subsystem:

            conditions.append(
                models.FieldCondition(
                    key="subsystem",
                    match=models.MatchValue(
                        value=subsystem
                    ),
                )
            )

        if component:

            conditions.append(
                models.FieldCondition(
                    key="component",
                    match=models.MatchValue(
                        value=component
                    ),
                )
            )

        if fault_code:

            conditions.append(
                models.FieldCondition(
                    key="fault_codes",
                    match=models.MatchValue(
                        value=fault_code
                    ),
                )
            )

        # -----------------------------------------------------
        # MEMORY TYPE
        # -----------------------------------------------------

        if memory_types:

            conditions.append(
                models.FieldCondition(
                    key="memory_type",
                    match=models.MatchAny(
                        any=memory_types
                    ),
                )
            )

        # -----------------------------------------------------
        # STATUS
        # -----------------------------------------------------

        if statuses:

            conditions.append(
                models.FieldCondition(
                    key="status",
                    match=models.MatchAny(
                        any=statuses
                    ),
                )
            )

        if verified_only:

            conditions.append(
                models.FieldCondition(
                    key="status",
                    match=models.MatchValue(
                        value="verified"
                    ),
                )
            )

        # -----------------------------------------------------
        # SCOPE
        # -----------------------------------------------------

        if scope:

            conditions.append(
                models.FieldCondition(
                    key="scope",
                    match=models.MatchValue(
                        value=scope
                    ),
                )
            )

        if owner_id:

            conditions.append(
                models.FieldCondition(
                    key="owner_id",
                    match=models.MatchValue(
                        value=owner_id
                    ),
                )
            )

        # -----------------------------------------------------
        # DEPRECATED MEMORY PROTECTION
        # -----------------------------------------------------

        must_not: list[
            models.Condition
        ] = []

        if not include_deprecated:

            must_not.append(
                models.FieldCondition(
                    key="status",
                    match=models.MatchValue(
                        value="deprecated"
                    ),
                )
            )

        # -----------------------------------------------------
        # EMPTY FILTER
        # -----------------------------------------------------

        if not conditions and not must_not:
            return None

        return models.Filter(
            must=conditions or None,
            must_not=must_not or None,
        )

    # =========================================================
    # MEMORY -> PAYLOAD
    # =========================================================

    @staticmethod
    def _payload(
        memory: MemoryRecord,
        *,
        canonical_id: str,
    ) -> dict[str, Any]:
        """
        Serialize a MemoryRecord into Qdrant payload.

        The payload contains structured retrieval metadata plus
        provenance required by the reasoning layer.
        """

        return {
            # -------------------------------------------------
            # IDENTITY
            # -------------------------------------------------

            "memory_id": canonical_id,

            # -------------------------------------------------
            # CORE MEMORY
            # -------------------------------------------------

            "memory_type": (
                memory.memory_type.value
            ),

            "status": (
                memory.status.value
            ),

            "content": memory.content,

            "confidence": memory.confidence,

            # -------------------------------------------------
            # DOMAIN
            # -------------------------------------------------

            "equipment_model": (
                memory.equipment_model
            ),

            "equipment_serial": (
                memory.equipment_serial
            ),

            "equipment_family": (
                memory.equipment_family
            ),

            "system": memory.system,

            "subsystem": memory.subsystem,

            "component": memory.component,

            "fault_codes": memory.fault_codes,

            # -------------------------------------------------
            # PROVENANCE
            # -------------------------------------------------

            "source_ids": memory.source_ids,

            "evidence": [
                {
                    "evidence_type": (
                        item.evidence_type.value
                    ),
                    "reference_id": (
                        item.reference_id
                    ),
                    "description": (
                        item.description
                    ),
                    "confidence": (
                        item.confidence
                    ),
                    "created_at": (
                        item.created_at.isoformat()
                    ),
                }
                for item in memory.evidence
            ],

            # -------------------------------------------------
            # EVOLUTION
            # -------------------------------------------------

            "observation_count": (
                memory.observation_count
            ),

            "successful_resolution_count": (
                memory.successful_resolution_count
            ),

            "contradiction_count": (
                memory.contradiction_count
            ),

            "last_confirmed_at": (
                memory.last_confirmed_at.isoformat()
                if memory.last_confirmed_at
                else None
            ),

            # -------------------------------------------------
            # TIMESTAMPS
            # -------------------------------------------------

            "created_at": (
                memory.created_at.isoformat()
            ),

            "updated_at": (
                memory.updated_at.isoformat()
            ),

            # -------------------------------------------------
            # LIFECYCLE
            # -------------------------------------------------

            "supersedes_memory_id": (
                memory.supersedes_memory_id
            ),

            # -------------------------------------------------
            # METADATA
            # -------------------------------------------------

            "tags": memory.tags,

            "metadata": memory.metadata,

            # -------------------------------------------------
            # COMMON FILTER SHORTCUTS
            # -------------------------------------------------

            "scope": memory.metadata.get(
                "scope",
                "global",
            ),

            "owner_id": memory.metadata.get(
                "owner_id"
            ),
        }

    # =========================================================
    # POINT CONSTRUCTION
    # =========================================================

    def _build_point(
        self,
        memory: MemoryRecord,
    ) -> models.PointStruct:
        """
        Convert a MemoryRecord into a Qdrant point.

        Both dense and sparse embeddings are generated by
        Qdrant Cloud Inference.
        """

        point_id = memory_identity(
            memory
        )

        content = memory.content

        return models.PointStruct(
            id=point_id,

            vector={
                self.config.dense_vector_name:
                    models.Document(
                        text=content,
                        model=(
                            self.config.dense_model
                        ),
                    ),

                self.config.sparse_vector_name:
                    models.Document(
                        text=content,
                        model=(
                            self.config.sparse_model
                        ),
                    ),
            },

            payload=self._payload(
                memory,
                canonical_id=point_id,
            ),
        )

    # =========================================================
    # UPSERT
    # =========================================================

    async def upsert_memory(
        self,
        memory: MemoryRecord,
        *,
        wait: bool = False,
    ) -> str:
        """
        Persist one memory.

        Default wait=False keeps background memory persistence
        cheap for conversational usage.

        Critical persistence callers can explicitly use wait=True.
        """

        if self._closed:
            raise RuntimeError(
                "Qdrant repository is closed."
            )

        point = self._build_point(
            memory
        )

        await self.client.upsert(
            collection_name=(
                self.config.collection_name
            ),
            points=[point],
            wait=wait,
        )

        return str(point.id)

    # =========================================================
    # BATCH UPSERT
    # =========================================================

    async def upsert_memories(
        self,
        memories: list[MemoryRecord],
        *,
        wait: bool = False,
    ) -> list[str]:
        """
        Persist multiple memories in one Qdrant operation.
        """

        if self._closed:
            raise RuntimeError(
                "Qdrant repository is closed."
            )

        if not memories:
            return []

        points = [
            self._build_point(memory)
            for memory in memories
        ]

        await self.client.upsert(
            collection_name=(
                self.config.collection_name
            ),
            points=points,
            wait=wait,
        )

        return [
            str(point.id)
            for point in points
        ]

    # =========================================================
    # DENSE SEARCH
    # =========================================================

    async def search_dense(
        self,
        query: str,
        *,
        limit: int = 8,
        query_filter: models.Filter | None = None,
    ):
        """
        Semantic retrieval using Qdrant Cloud Inference.

        Best for natural-language descriptions of symptoms,
        behavior, and troubleshooting situations.
        """

        self._validate_query(
            query,
            limit,
        )

        result = await self.client.query_points(
            collection_name=(
                self.config.collection_name
            ),

            query=models.Document(
                text=query,
                model=(
                    self.config.dense_model
                ),
            ),

            using=(
                self.config.dense_vector_name
            ),

            query_filter=query_filter,

            limit=limit,

            with_payload=models.PayloadSelectorInclude(
                include=[
                    "memory_id",
                    "memory_type",
                    "status",
                    "content",
                    "confidence",
                    "equipment_model",
                    "equipment_family",
                    "equipment_serial",
                    "system",
                    "subsystem",
                    "component",
                    "fault_codes",
                    "evidence",
                ]
            ),

            with_vectors=False,
        )

        return result.points

    # =========================================================
    # SPARSE SEARCH
    # =========================================================

    async def search_sparse(
        self,
        query: str,
        *,
        limit: int = 8,
        query_filter: models.Filter | None = None,
    ):
        """
        Exact/token-oriented retrieval using Qdrant BM25.

        Particularly useful for:

            - Windows error codes
            - Event IDs
            - model numbers
            - driver names
            - fault codes
            - component names
            - exact technical terminology
        """

        self._validate_query(
            query,
            limit,
        )

        result = await self.client.query_points(
            collection_name=(
                self.config.collection_name
            ),

            query=models.Document(
                text=query,
                model=(
                    self.config.sparse_model
                ),
            ),

            using=(
                self.config.sparse_vector_name
            ),

            query_filter=query_filter,

            limit=limit,

            with_payload=models.PayloadSelectorInclude(
                include=[
                    "memory_id",
                    "memory_type",
                    "status",
                    "content",
                    "confidence",
                    "equipment_model",
                    "equipment_family",
                    "equipment_serial",
                    "system",
                    "subsystem",
                    "component",
                    "fault_codes",
                    "evidence",
                ]
            ),

            with_vectors=False,
        )

        return result.points

    # =========================================================
    # HYBRID SEARCH
    # =========================================================

    async def search_hybrid(
        self,
        query: str,
        *,
        limit: int = 8,
        query_filter: models.Filter | None = None,
        prefetch_limit: int | None = None,
    ):
        """
        Dense + BM25 hybrid retrieval.

        Qdrant performs the two retrievals and combines them
        using native Reciprocal Rank Fusion.

        The RetrievalOrchestrator decides whether this more
        expensive path is justified.
        """

        self._validate_query(
            query,
            limit,
        )

        if prefetch_limit is None:
            prefetch_limit = max(
                limit * 2,
                10,
            )

        result = await self.client.query_points(
            collection_name=(
                self.config.collection_name
            ),

            prefetch=[
                models.Prefetch(
                    query=models.Document(
                        text=query,
                        model=(
                            self.config
                            .dense_model
                        ),
                    ),
                    using=(
                        self.config
                        .dense_vector_name
                    ),
                    limit=prefetch_limit,
                    filter=query_filter,
                ),

                models.Prefetch(
                    query=models.Document(
                        text=query,
                        model=(
                            self.config
                            .sparse_model
                        ),
                    ),
                    using=(
                        self.config
                        .sparse_vector_name
                    ),
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
            ],

            query=models.RrfQuery(
                rrf=models.Rrf(
                    k=60,
                )
            ),

            query_filter=query_filter,

            limit=limit,

            with_payload=models.PayloadSelectorInclude(
                include=[
                    "memory_id",
                    "memory_type",
                    "status",
                    "content",
                    "confidence",
                    "equipment_model",
                    "equipment_family",
                    "equipment_serial",
                    "system",
                    "subsystem",
                    "component",
                    "fault_codes",
                    "evidence",
                ]
            ),

            with_vectors=False,
        )

        return result.points

    # =========================================================
    # COMPATIBILITY SEARCH
    # =========================================================

    async def search(
        self,
        query: str,
        *,
        limit: int = 8,
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
        include_deprecated: bool = False,
    ):
        """
        Compatibility search wrapper.

        Production retrieval should normally go through
        RetrievalOrchestrator so that it can choose dense,
        sparse, or hybrid retrieval adaptively.
        """

        query_filter = self.build_filter(
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
            include_deprecated=include_deprecated,
        )

        return await self.search_hybrid(
            query,
            limit=limit,
            query_filter=query_filter,
        )

    # =========================================================
    # QUERY VALIDATION
    # =========================================================

    @staticmethod
    def _validate_query(
        query: str,
        limit: int,
    ) -> None:

        if not query or not query.strip():
            raise ValueError(
                "Qdrant search query cannot be empty."
            )

        if limit <= 0:
            raise ValueError(
                "Qdrant search limit must be greater than zero."
            )

    # =========================================================
    # CLOSE
    # =========================================================

    async def close(
        self,
    ) -> None:
        """
        Idempotently close the Qdrant connection.
        """

        if self._closed:
            return

        self._closed = True

        await self.client.close()