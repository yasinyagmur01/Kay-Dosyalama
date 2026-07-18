"""Tam LangGraph pipeline: OCR → Classifier → Mevzuat → [Validator] → Drafter → Router."""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, StateGraph

from agents.classifier_agent import ClassifierAgent
from agents.drafter_agent import DrafterAgent
from agents.mevzuat_agent import MevzuatAgent
from agents.ocr_agent import OCRAgent
from agents.routing_agent import RoutingAgent
from agents.validator_agent import ValidatorAgent, apply_user_responses
from core.state import DocumentState, create_initial_state

# Agent instance'ları (testlerde build_graph / modül override ile değiştirilebilir)
ocr = OCRAgent()
classifier = ClassifierAgent()
mevzuat = MevzuatAgent()
validator = ValidatorAgent()
drafter = DrafterAgent()
router = RoutingAgent()

OcrRoute = Literal["classifier", "end"]
MevzuatRoute = Literal["validator", "drafter", "end"]


def route_after_ocr(state: DocumentState) -> OcrRoute:
    """OCR sonrası classifier veya bitiş rotası."""
    if state.get("validation_status") == "error":
        return "end"
    if not (state.get("raw_text") or "").strip():
        return "end"
    return "classifier"


def route_after_mevzuat(state: DocumentState) -> MevzuatRoute:
    """Mevzuat sonrası validator, drafter veya bitiş rotası."""
    if state.get("validation_status") == "error":
        return "end"
    if state.get("missing_fields"):
        return "validator"
    return "drafter"


def build_graph(
    ocr_agent: OCRAgent | None = None,
    classifier_agent: ClassifierAgent | None = None,
    mevzuat_agent: MevzuatAgent | None = None,
    validator_agent: ValidatorAgent | None = None,
    drafter_agent: DrafterAgent | None = None,
    routing_agent: RoutingAgent | None = None,
) -> Any:
    """Tam pipeline StateGraph'ini derler."""
    ocr_node = ocr_agent or ocr
    clf = classifier_agent or classifier
    mev = mevzuat_agent or mevzuat
    val = validator_agent or validator
    drf = drafter_agent or drafter
    rte = routing_agent or router

    graph: StateGraph = StateGraph(DocumentState)

    graph.add_node("ocr", ocr_node.invoke)
    graph.add_node("classifier", clf.invoke)
    graph.add_node("mevzuat", mev.invoke)
    graph.add_node("validator", val.invoke)
    graph.add_node("drafter", drf.invoke)
    graph.add_node("router", rte.invoke)

    graph.set_entry_point("ocr")

    graph.add_conditional_edges(
        "ocr",
        route_after_ocr,
        {
            "classifier": "classifier",
            "end": END,
        },
    )
    graph.add_edge("classifier", "mevzuat")
    graph.add_conditional_edges(
        "mevzuat",
        route_after_mevzuat,
        {
            "validator": "validator",
            "drafter": "drafter",
            "end": END,
        },
    )
    graph.add_edge("validator", "drafter")
    graph.add_edge("drafter", "router")
    graph.add_edge("router", END)

    return graph.compile()


_graph: Any | None = None


def get_graph() -> Any:
    """Derlenmiş graph singleton'ı döner."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def reset_graph() -> None:
    """Test injection için graph singleton'ını sıfırlar."""
    global _graph
    _graph = None


async def run_pipeline(
    raw_input: str,
    input_type: str = "text",
    user_responses: dict | None = None,
) -> DocumentState:
    """Tam pipeline'ı çalıştırır; user_responses varsa entity'lere uygular."""
    state = create_initial_state(raw_input, input_type)
    if user_responses:
        cleaned = {
            str(key): str(value).strip()
            for key, value in user_responses.items()
            if value is not None and str(value).strip()
        }
        if cleaned:
            state = {**state, **apply_user_responses(state, cleaned)}
    graph = get_graph()
    result = await graph.ainvoke(state)
    return result  # type: ignore[return-value]
