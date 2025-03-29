from pydantic import BaseModel, Field
from typing import List
import xml.etree.ElementTree as ET

from .utils import extract_code_blocks


class Step(BaseModel):
    """Describe one step found in the user message."""

    description: str = Field(description="The description of the step")
    web_search: str = Field(
        description="A web search that can help to implement the step"
    )


class Steps(BaseModel):
    """The steps defined in the user message split in several parts"""

    # context: str = Field(description="The introduction of the message (all text before the steps description)")

    request: str = Field(description="The request of the user")
    web_search: List[str] = Field(
        description="A List of web search that can help to implement the step"
    )
    steps: List[str] = Field(description="The list of steps defined in the message.")


def coordinator(state):
    _, code_blocks = extract_code_blocks(state["messages"][-1].content)
    root = ET.fromstring(code_blocks)

    request = root[0].text
    web_search = []
    steps = []
    for elem in root.iter():
        if elem.tag == "step":
            steps.append(elem.text)
        if elem.tag == "web_search":
            for child in elem:
                web_search.append(child.text)

    return {
        "steps": Steps(request=request, web_search=web_search, steps=steps),
        "current_code": "",
        "current_step": -1,
    }
