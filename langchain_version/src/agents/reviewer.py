import os
import logging
from langchain_core.messages import HumanMessage

from .state import State
from .default import DefaultAgent
from .utils import extract_and_write_code

logger = logging.getLogger(__name__)


class Reviewer(DefaultAgent):
    def __init__(
        self,
        model="qwen2.5-coder:7b",
        temperature=0.0,
        num_ctx=32768,
        prompt=os.path.join(os.path.dirname(__file__), "reviewer.md"),
    ):
        super().__init__(
            model=model, temperature=temperature, num_ctx=num_ctx, prompt=prompt
        )
        self.ite = 0

    def invoke(self, state: State):
        logger.info("start reviewer")

        solutions = "\n\n".join(
            [f"solution for Agent {i}:\n {s}" for i, s in enumerate(state["solutions"])]
        )

        message = f"""
The work of your team is resumed in the following
{solutions}

The current code is:
{state['current_code']}

        """

        prompt = self.prompt_template.invoke(
            {"messages": state["messages"] + [HumanMessage(message)]}
        )
        response = self.agent.invoke(prompt)

        logger.debug(response.pretty_print())
        logger.info("end reviewer")

        current_code = extract_and_write_code(
            response.content, f"current_code_{self.ite}"
        )
        self.ite += 1
        return {"messages": response, "current_code": current_code}
