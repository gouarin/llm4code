from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from typing import Sequence
import operator


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_code: BaseMessage
    solutions: Annotated[list, operator.add]
