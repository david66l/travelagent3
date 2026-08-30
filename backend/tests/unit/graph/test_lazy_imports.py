import sys


def test_graph_package_does_not_eagerly_import_orchestrator():
    for name in ("graph", "graph.graph"):
        sys.modules.pop(name, None)

    import graph

    assert "graph.graph" not in sys.modules
    assert graph.__all__ == [
        "AgentState",
        "SessionManager",
        "build_graph",
        "get_graph",
        "set_graph",
    ]


def test_graph_public_model_remains_available_through_lazy_api():
    import graph
    from graph.models import AgentState

    assert graph.AgentState is AgentState
