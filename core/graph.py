"""Görev 1 partial LangGraph pipeline (OCR ve Drafter hariç)."""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, StateGraph

from agents.classifier_agent import ClassifierAgent
from agents.mevzuat_agent import MevzuatAgent
from agents.validator_agent import ValidatorAgent
from core.state import DocumentState, create_initial_state

# Agent instance'ları (testlerde build_graph ile override edilebilir)
classifier = ClassifierAgent()
mevzuat = MevzuatAgent()
validator = ValidatorAgent()

RouteTarget = Literal["validator", "end"]


def route_after_mevzuat(state: DocumentState) -> RouteTarget:
    """Mevzuat sonrası validator veya bitiş rotası."""
    if state.get("validation_status") == "error":
        return "end"
    if state.get("missing_fields"):
        return "validator"
    return "end"


def build_graph(
    classifier_agent: ClassifierAgent | None = None,
    mevzuat_agent: MevzuatAgent | None = None,
    validator_agent: ValidatorAgent | None = None,
) -> Any:
    """Görev 1 StateGraph'ini derler."""
    clf = classifier_agent or classifier
    mev = mevzuat_agent or mevzuat
    val = validator_agent or validator

    graph: StateGraph = StateGraph(DocumentState)

    graph.add_node("classifier", clf.invoke)
    graph.add_node("mevzuat", mev.invoke)
    graph.add_node("validator", val.invoke)

    graph.set_entry_point("classifier")
    graph.add_edge("classifier", "mevzuat")
    graph.add_conditional_edges(
        "mevzuat",
        route_after_mevzuat,
        {
            "validator": "validator",
            "end": END,
        },
    )
    graph.add_edge("validator", END)

    return graph.compile()


_graph: Any | None = None


def get_graph() -> Any:
    """Derlenmiş graph singleton'ı döner."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def reset_graph() -> None:
    """Testler için graph singleton'ını sıfırlar."""
    global _graph
    _graph = None


async def run_pipeline(raw_input: str, input_type: str = "text") -> DocumentState:
    """Görev 1 pipeline'ını çalıştırır."""
    state = create_initial_state(raw_input, input_type)
    graph = get_graph()
    result = await graph.ainvoke(state)
    return result  # type: ignore[return-value]
