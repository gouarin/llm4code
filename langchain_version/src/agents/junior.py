import os
import logging
from langchain_core.messages import HumanMessage

from .state import State
from .default import DefaultAgent

logger = logging.getLogger(__name__)


class Junior(DefaultAgent):
    def __init__(
        self,
        model="qwen2.5-coder:7b",
        temperature=0.0,
        num_ctx=32768,
        prompt=os.path.join(os.path.dirname(__file__), "junior.md"),
    ):
        super().__init__(
            model=model, temperature=temperature, num_ctx=num_ctx, prompt=prompt
        )

    def invoke(self, state: State):
        logger.info("start junior")
        prompt = self.prompt_template.invoke(
            {
                "messages": state["messages"][:-1]
                + [
                    HumanMessage(state["messages"][-1].content),
                    HumanMessage(
                        f"You have to add your code to the following existing code:\n{state['current_code']}"
                    ),
                ]
            }
        )
        response = self.agent.invoke(prompt)

        logger.debug(response.pretty_print())
        logger.info("end junior")

        return {"solutions": [response.content]}
