import os
from textwrap import dedent
from typing import List
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel, OpenAIModelSettings
from pydantic_ai.providers.openai import OpenAIProvider


class CodeState(BaseModel):
    """Code proposed by the junior split in two sections: imports and code"""

    imports: str = Field(description="import statement to use")
    code: str = Field(description="The Python script without the import lines")


TESTER_PROMPT = """
You are the best to test new codes and find the bugs in them.

You write tests using pytest to ensure that if there are any errors or bugs, your tests will explain what went wrong and how it can be fixed.

"""


class Tester:
    def __init__(self):
        OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
        TESTER_MODEL = os.getenv("TESTER_MODEL")

        ollama_model = OpenAIModel(
            model_name=TESTER_MODEL,
            provider=OpenAIProvider(base_url=OLLAMA_BASE_URL + "/v1/"),
        )

        self.junior_agent = Agent(
            model=ollama_model,
            system_prompt=TESTER_PROMPT,
            model_settings=OpenAIModelSettings(
                temperature=0.0,
            ),
            # retries=4,
            # result_type=CodeState,
        )

    def invoke(self, state, **kwargs):
        print("**** Tester ****")
        response = self.junior_agent.run_sync(
            f"""

            The code you have to test is: {state["current_code"]}

            Add in the same file the tests for the code.

            """,
            **kwargs,
        )
        print("response: ", response)
        with open(os.path.join(os.getcwd(), "output/tester_response.md"), "w") as f:
            f.write(response.data)
        state["current_code_with_test"] = response.data
        return state
