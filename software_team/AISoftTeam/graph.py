from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated

from .agents import Analyst, coordinator, Coder, Tester, Searcher, Coder_2
from .agents.coordinator import Steps


class State(TypedDict):
    messages: Annotated[list, add_messages]
    websearch: list
    steps: Steps
    current_code: str
    current_code_with_test: str
    current_step: int


def create_team():
    graph = StateGraph(State)
    graph.add_node("researcher", Searcher().invoke)
    graph.add_node("analyst", Analyst().invoke)
    graph.add_node("coordinator", coordinator)
    graph.add_node("coder", Coder().invoke)
    graph.add_node("tester", Tester().invoke)
    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst", "coordinator")
    graph.add_edge("coordinator", "coder")
    graph.add_edge("coder", "tester")
    graph.add_edge("tester", END)
    return graph.compile()


def create_small_team():
    graph = StateGraph(State)
    graph.add_node("researcher", Searcher().invoke)
    graph.add_node("coder", Coder_2().invoke)
    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "coder")
    graph.add_edge("coder", END)
    return graph.compile()
