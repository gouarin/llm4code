import os
from textwrap import dedent
from typing import List
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel, OpenAIModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

from langchain_community.tools import DuckDuckGoSearchResults


class CodeState(BaseModel):
    """Python code proposed"""

    python_code: str = Field(description="The Python script that implement the request")


JUNIOR_PROMPT = """
You are a junior software developer with 2 years of experience. You have worked on various scientific projects in Python. Your expertise lies in writing clean and efficient code, debugging issues, and learning new technologies.

You are always eager to learn new technologies and apply them in your work. Your code is always clean and concise and well documented.

Your answer must verify the following bullets:
- Your answer is a Python script and nothing else.
- You implement only what is asking: not tests, no main function, no imports that are not used.
- You don't add extra explanations.
- You don't add extra comments.
- Your code is well formatted and follows PEP 8 guidelines.
- You provide docstrings using the numpydoc format.
- You'll try to use as few dependencies as possible and rely on those that are commonly used.
- The code should run as it is without any user input.

"""


class Coder_2:
    def __init__(self):
        OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
        CODER_MODEL = os.getenv("CODER_MODEL")

        ollama_model = OpenAIModel(
            model_name=CODER_MODEL,
            provider=OpenAIProvider(base_url=OLLAMA_BASE_URL + "/v1/"),
        )

        self.junior_agent = Agent(
            model=ollama_model,
            system_prompt=JUNIOR_PROMPT,
            model_settings=OpenAIModelSettings(
                temperature=0.0,
            ),
            instrument=True,
            retries=4,
            # result_type=CodeState,
        )
        self.junior_agent.instrument_all()

    def invoke(self, state, **kwargs):
        print("**** Coder ****")
        request = state["messages"][-1].content
        search_results = state["websearch"]

        response = self.junior_agent.run_sync(
            f"""

            The request is: {request}

            The web search are:
            {search_results}

            Give me a working python implementation of this request.
            """,
            **kwargs,
        )
        print("response: ", response)
        with open(os.path.join(os.getcwd(), "output/coder_response.md"), "w") as f:
            f.write(response.data)
        state["current_code"] = response.data
        return state
