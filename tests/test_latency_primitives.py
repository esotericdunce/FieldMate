from fieldmate.brain.retrieval.orchestrator import RetrievalOrchestrator
from fieldmate.brain.retrieval.planner import plan_retrieval


def test_partial_converges_on_final():
    assert RetrievalOrchestrator._query_convergence(
        "my lenovo laptop wifi",
        "my lenovo laptop wifi keeps disconnecting",
    ) >= 0.55


def test_unrelated_queries_do_not_converge():
    assert RetrievalOrchestrator._query_convergence(
        "my lenovo laptop wifi",
        "the printer is making a strange noise",
    ) < 0.55


def test_planner_is_available():
    plan = plan_retrieval("my laptop wifi keeps disconnecting")
    assert plan.mode is not None
