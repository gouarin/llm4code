from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langchain_core.messages import HumanMessage
from langgraph.types import Send

from ..agents import Supervisor, Junior, Reviewer
from ..agents.state import State


class FeatureGraph:
    def __init__(self, nb_juniors=2):
        self.nb_juniors = nb_juniors
        self.supervisor = Supervisor()
        self.junior = Junior()
        self.reviewer = Reviewer()

        self.memory = MemorySaver()

        self.graph = StateGraph(state_schema=State)
        self.graph.add_edge(START, "analyse")
        self.graph.add_edge("analyse", "junior")
        self.graph.add_conditional_edges("analyse", self.send_to_junior, ["junior"])
        # self.graph.add_edge("junior", "reviewing")

        self.graph.add_node("analyse", self.supervisor.invoke)
        self.graph.add_node("junior", self.junior.invoke)
        self.graph.add_node("reviewing", self.reviewer.invoke)
        self.app = self.graph.compile(checkpointer=self.memory)
        self.current_code = ""

    def send_to_junior(self, state: State):
        return [Send("junior", state) for i in range(self.nb_juniors)]

    def invoke(self, query):
        config = {"configurable": {"thread_id": "abc123"}}

        state = {
            "messages": [HumanMessage(content=query)],
            "current_code": self.current_code,
        }
        response = self.app.invoke(state, config)
        self.current_code = state["current_code"]
        return response
