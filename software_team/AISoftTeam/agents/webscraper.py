import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.chat_models import ChatOllama
from langchain_community.tools import DuckDuckGoSearchResults
from duckduckgo_search import DDGS

import xml.etree.ElementTree as ET

from .utils import extract_code_blocks
from .extractor import extract_content

load_dotenv()


PROMPT = """You are a smart assistant bot that helps answer questions from the user.

- You identify a list of web search to be done according to the several topics of the question to find information to help to answer the question and nothing else.
- You don't try to answer the question.
- You just identify the web search to be done.
- Add the research of the mathematical formula of the studies.
- For each <web_search>, you will write a list of keywords to search on the web to find information to help to answer the question.
- You pay attention that each <web_search> is about a different topic.
- You use the following structure to answer the question:

```xml
<root>
    <web_search id="1">
        List of keywords to search on the web to find information to help to answer the question
    </web_search>
    <web_search id="2">
        List of keywords to search on the web to find information to help to answer the question
    </web_search>
    <web_search id="3">
        List of keywords to search on the web to find information to help to answer the question
    </web_search>
    <web_search id="4">
        List of keywords to search on the web to find information to help to answer the question
    </web_search>
    <web_search id="5">
        List of keywords to search on the web to find information to help to answer the question
    </web_search>
</root>
```

- You have to use this structure to your response and nothing else.
- There is just keywords in each <websearch> tag, no codes, no explanations.

"""

PROMPT_2 = """You are a smart assistant bot that helps answer questions from the user.

- You identify a list of web search to be done according to the several topics of the question to find information to help to answer the question and nothing else.
- You don't try to answer the question.
- You just identify the web search to be done.
- For each <web_search>, you will write what you want to search on the web to find information to help to answer the question. It must be a sub question.
- You pay attention that each <web_search> is about a different topic.
- You use the following structure to answer the question:

    <root>
        <web_search id="1">
            sub question to search on the web to find information to help to answer the question
        </web_search>
        <web_search id="2">
            sub question to search on the web to find information to help to answer the question
        </web_search>
        ...
        <web_search id="n">
            sub question to search on the web to find information to help to answer the question
        </web_search>
    </root>

- You have to use this structure to your response and nothing else.
- There is just one question in each <websearch> tag, no codes, no explanations.

"""

structure = """
    <root>
        <web_search id="1">
            List of keywords to search on the web to find information to help to answer the question
        </web_search>
        <web_search id="2">
            List of keywords to search on the web to find information to help to answer the question
        </web_search>
        ...
        <web_search id="n">
            List of keywords to search on the web to find information to help to answer the question
        </web_search>
    </root>
"""


class Searcher:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
        SEARCHER_MODEL = os.getenv("SEARCHER_MODEL")

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PROMPT),
                (
                    "human",
                    """
                    The user request is : {task}. Tell me more about it.

                    {extra}
                    """,
                ),
            ]
        )
        llm = ChatOllama(
            model=SEARCHER_MODEL,
            seed=42,
            temperature=0.2,
            num_ctx=32768,
            base_url=OLLAMA_BASE_URL,
        )
        self.researcher_llm = prompt | llm

    def invoke(self, state, **kwargs):
        """Invoke the agent with the given question and context."""
        question = state["messages"][-1]
        print("**** Researcher ****")
        # print("question: ", question.content)
        print("question: ", question)

        response = self.researcher_llm.invoke({"task": question, "extra": ""}, **kwargs)
        iteration = 1
        length, _ = extract_code_blocks(response.content)
        while length == 0 and iteration < 3:
            print("iteration: ", iteration)
            response = self.researcher_llm.invoke(
                {
                    "task": question,
                    "extra": f"you're previous answer doesn't follow the structure: {structure} \n please changed it accordingly",
                },
                **kwargs,
            )
            length, _ = extract_code_blocks(response.content)
            with open(
                os.path.join(os.getcwd(), f"output/searcher_response_{iteration}.md"),
                "w",
            ) as f:
                f.write(response.content)
            iteration += 1

        if iteration == 3:
            raise Exception("No code blocks found in response")

        with open(os.path.join(os.getcwd(), "output/searcher_response.md"), "w") as f:
            f.write(response.content)

        _, code_blocks = extract_code_blocks(response.content)
        root = ET.fromstring(code_blocks)

        web_search = []
        for elem in root.iter():
            if elem.tag == "web_search":
                web_search.append(elem.text)

        search_results = []
        for s in web_search:
            print("searching: ", s)
            results = DDGS().text(s)
            search_results.append(results)

        return {"websearch": search_results}
