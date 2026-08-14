from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from fieldmate.brain.retrieval.context import (
    DiagnosticContext,
    RetrievedMemory,
)
from fieldmate.brain.retrieval.evidence import Evidence
from fieldmate.brain.retrieval.orchestrator import RetrievalOrchestrator
from fieldmate.brain.retrieval.semantic_cache import (
    QdrantSemanticCache,
    deserialize_context,
    serialize_context,
)


def test_context_serialization_roundtrip():
    evidence = Evidence(
        evidence_id="ev_456",
        memory_id="mem_123",
        memory_type="procedure",
        content="Check power adapter voltage output.",
        source="qdrant",
        equipment_model="Precision 5680",
        fault_codes=("ERR_PWR_01",),
        relevance_score=0.95,
        confidence=0.9,
        verification_status="verified",
        provenance="qdrant_dense",
        relation="supporting",
    )
    original_context = DiagnosticContext(
        evidence=(evidence,),
    )

    serialized = serialize_context(original_context)
    reconstructed = deserialize_context(serialized)

    assert len(reconstructed.evidence) == 1
    assert reconstructed.evidence[0].evidence_id == "ev_456"
    assert reconstructed.evidence[0].content == "Check power adapter voltage output."


@pytest.mark.asyncio
async def test_per_user_isolation():
    mock_repo = MagicMock()
    mock_client = AsyncMock()
    mock_repo.client = mock_client
    mock_repo.config.dense_model = "sentence-transformers/all-minilm-l6-v2"
    mock_repo.config.dense_vector_name = "dense"

    cache = QdrantSemanticCache(
        mock_repo,
        collection_name="test_cache",
        threshold=0.90,
    )

    # Mock query_points to check filter conditions
    def mock_query_points(collection_name, query, using, query_filter, **kwargs):
        # Extract owner_id match condition from query_filter
        owner_condition = None
        if query_filter and query_filter.must:
            for cond in query_filter.must:
                if getattr(cond, "key", None) == "owner_id":
                    owner_condition = cond.match.value

        if owner_condition == "tech_user_A":
            ev = Evidence(
                evidence_id="ev_A",
                memory_id="mem_A",
                memory_type="procedure",
                content="Tech A specific diagnostic result",
                source="qdrant",
                equipment_model=None,
                fault_codes=(),
                relevance_score=0.96,
                confidence=0.96,
                verification_status="verified",
                provenance="qdrant_dense",
                relation="supporting",
            )
            ctx = DiagnosticContext(evidence=(ev,))
            point = MagicMock()
            point.id = "p_A"
            point.score = 0.95
            point.payload = {
                "context": serialize_context(ctx),
                "hit_count": 1,
                "created_at": int(time.time()),
            }
            return MagicMock(points=[point])

        # User B or missing owner_id gets no points
        return MagicMock(points=[])

    mock_client.query_points.side_effect = mock_query_points

    # 1. Tech A lookup -> HIT
    res_A = await cache.lookup("hydraulic pressure dropping", owner_id="tech_user_A")
    assert res_A is not None
    ctx_A, score_A = res_A
    assert ctx_A.evidence[0].content == "Tech A specific diagnostic result"

    # 2. Tech B lookup -> MISS (User isolation enforced!)
    res_B = await cache.lookup("hydraulic pressure dropping", owner_id="tech_user_B")
    assert res_B is None

    # 3. Unauthenticated/No owner_id lookup -> MISS
    res_none = await cache.lookup("hydraulic pressure dropping", owner_id=None)
    assert res_none is None


@pytest.mark.asyncio
async def test_equipment_serial_and_subsystem_isolation():
    mock_repo = MagicMock()
    mock_client = AsyncMock()
    mock_repo.client = mock_client
    mock_repo.config.dense_model = "sentence-transformers/all-minilm-l6-v2"
    mock_repo.config.dense_vector_name = "dense"

    cache = QdrantSemanticCache(
        mock_repo,
        collection_name="test_cache",
        threshold=0.90,
    )

    def mock_query_points(query_filter, **kwargs):
        conditions = {}
        if query_filter and query_filter.must:
            for item in query_filter.must:
                if hasattr(item, "key") and hasattr(item, "match") and item.match:
                    conditions[item.key] = item.match.value
                elif hasattr(item, "should") and item.should:
                    for sub in item.should:
                        if hasattr(sub, "key") and hasattr(sub, "match") and sub.match:
                            conditions[sub.key] = sub.match.value
        
        # Only return point if serial is SERIAL_123 and subsystem is hydraulic
        if conditions.get("equipment_serial") == "SERIAL_123" and conditions.get("subsystem") == "hydraulic":
            ev = Evidence(
                evidence_id="e1",
                memory_id="m1",
                memory_type="procedure",
                content="Serial 123 hydraulic fix",
                source="qdrant",
                equipment_model=None,
                fault_codes=(),
                relevance_score=0.95,
                confidence=0.9,
                verification_status="verified",
                provenance="qdrant",
                relation="supporting",
            )
            ctx = DiagnosticContext(evidence=(ev,))
            pt = MagicMock(id="p1", score=0.95, payload={"context": serialize_context(ctx), "created_at": int(time.time())})
            return MagicMock(points=[pt])
        return MagicMock(points=[])

    mock_client.query_points.side_effect = mock_query_points

    # Matching serial & subsystem -> HIT
    res_match = await cache.lookup(
        "pressure loss",
        owner_id="tech_1",
        equipment_serial="SERIAL_123",
        subsystem="hydraulic",
    )
    assert res_match is not None
    assert res_match[0].evidence[0].content == "Serial 123 hydraulic fix"

    # Different serial -> MISS
    res_diff_serial = await cache.lookup(
        "pressure loss",
        owner_id="tech_1",
        equipment_serial="SERIAL_456",
        subsystem="hydraulic",
    )
    assert res_diff_serial is None

    # Different subsystem -> MISS
    res_diff_sub = await cache.lookup(
        "pressure loss",
        owner_id="tech_1",
        equipment_serial="SERIAL_123",
        subsystem="electrical",
    )
    assert res_diff_sub is None


@pytest.mark.asyncio
async def test_verified_only_isolation():
    mock_repo = MagicMock()
    mock_client = AsyncMock()
    mock_repo.client = mock_client
    mock_repo.config.dense_model = "sentence-transformers/all-minilm-l6-v2"
    mock_repo.config.dense_vector_name = "dense"

    cache = QdrantSemanticCache(mock_repo, collection_name="test_cache")

    # Unverified evidence in cached context
    unverified_evidence = Evidence(
        evidence_id="ev_unv",
        memory_id="m_unv",
        memory_type="procedure",
        content="Unverified procedure",
        source="qdrant",
        equipment_model=None,
        fault_codes=(),
        relevance_score=0.9,
        confidence=0.5,
        verification_status="unverified",
        provenance="qdrant_dense",
        relation="supporting",
    )
    ctx_unverified = DiagnosticContext(evidence=(unverified_evidence,))

    pt = MagicMock()
    pt.id = "pt_unv"
    pt.score = 0.95
    pt.payload = {"context": serialize_context(ctx_unverified), "created_at": int(time.time())}

    mock_client.query_points.return_value = MagicMock(points=[pt])

    # lookup with verified_only=True must reject unverified context -> MISS
    res_verified_req = await cache.lookup("check adapter", verified_only=True)
    assert res_verified_req is None


@pytest.mark.asyncio
async def test_ttl_expiration():
    mock_repo = MagicMock()
    mock_client = AsyncMock()
    mock_repo.client = mock_client
    mock_repo.config.dense_model = "sentence-transformers/all-minilm-l6-v2"
    mock_repo.config.dense_vector_name = "dense"

    cache = QdrantSemanticCache(
        mock_repo,
        collection_name="test_cache",
        ttl_seconds=3600.0,  # 1 hour TTL
    )

    ev = Evidence(
        evidence_id="e1",
        memory_id="m1",
        memory_type="procedure",
        content="Old context",
        source="qdrant",
        equipment_model=None,
        fault_codes=(),
        relevance_score=0.9,
        confidence=0.9,
        verification_status="verified",
        provenance="qdrant",
        relation="supporting",
    )
    ctx = DiagnosticContext(evidence=(ev,))
    expired_point = MagicMock()
    expired_point.id = "exp_1"
    expired_point.score = 0.95
    expired_point.payload = {
        "context": serialize_context(ctx),
        "created_at": int(time.time() - 7200),  # 2 hours old
    }

    mock_client.query_points.return_value = MagicMock(points=[expired_point])

    res_expired = await cache.lookup("overheating laptop")
    assert res_expired is None
    # Verify expired point deletion was triggered
    mock_client.delete.assert_called_once()


@pytest.mark.asyncio
async def test_store_response_text_fallback():
    mock_repo = MagicMock()
    mock_client = AsyncMock()
    mock_repo.client = mock_client
    mock_repo.config.dense_model = "sentence-transformers/all-minilm-l6-v2"
    mock_repo.config.dense_vector_name = "dense"

    cache = QdrantSemanticCache(mock_repo, collection_name="test_cache")

    empty_context = DiagnosticContext()
    point_id = await cache.store(
        "my dell laptop was overheating",
        empty_context,
        response_text="For overheating Dell laptops, switch to power-saving mode.",
        owner_id="tech_john_doe",
    )

    assert point_id is not None
    mock_client.upsert.assert_called_once()
    stored_point = mock_client.upsert.call_args.kwargs["points"][0]
    assert stored_point.payload["response_text"] == "For overheating Dell laptops, switch to power-saving mode."
    assert stored_point.payload["owner_id"] == "tech_john_doe"


@pytest.mark.asyncio
async def test_orchestrator_propagates_owner_id():
    mock_repo = MagicMock()
    mock_repo.build_filter.return_value = None

    mock_cache = AsyncMock()
    mock_cache.enabled = True
    mock_cache.lookup.return_value = None

    orchestrator = RetrievalOrchestrator(
        mock_repo,
        semantic_cache=mock_cache,
    )

    await orchestrator.retrieve("low battery warning", owner_id="tech_user_99")

    mock_cache.lookup.assert_called_once()
    assert mock_cache.lookup.call_args.kwargs["owner_id"] == "tech_user_99"
