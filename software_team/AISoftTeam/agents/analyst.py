import os
from textwrap import dedent
from typing import List
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel, OpenAIModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.embeddings import OllamaEmbeddings

from langchain_ollama.chat_models import ChatOllama
from langchain_chroma import Chroma

load_dotenv()


PROMPT = """You are the best to split a coding request into several small steps easily implementable by a junior developer.

- Think twice before answering.
- Use heavily the previous researches that you have done for this request.
- You read the question and split it into several steps to be able to analyse it in depth and provide the more precise answer you can.
- Your answer will not contain any code.
- For each step, add a title and the algorithm steps that will be translate in any programming language.
- The XML tag is important and must be used as described below.
- The XML tag name must be unchanged.
- You use only these XML tags: <root>, <request>, <material>, <algorithm>, <title>, <inputs>, <outputs>, <steps>, <tests> and nothing else.
- Each step will be writing in markdown format.
- The <material> section must contain at least 1000 words, give ideas of the overall and the mathematical formula of the studies.
- The algorithm steps must be detailed and must contain at least 10 steps.
- The algorithm steps must be as clear as possible and don't need any reasoning, just apply the algorithm.
- The development environment is already set up. You don't have to add that in the steps.
- You will provide a test section with the exact inputs values and the exact expected outputs values for each test (see example below).
- YOUR RESPONSE MUST STRICTLY HAVE THE FOLLOWING STRUCTURE AND NO OTHER SECTIONS:

```xml
<root>
    <request>
    reformulate the user request to a clear and concise paragraph which will be used by the team as a starting point.
    </request>
    <material>
        Long resume of the previous researches that can help to implement the request and give ideas of the overall. (at least 1000 words)
    </material>
    <algorithm id="1">
        <title>
            Title of the algorithm (to be replaced by the actual title)
        </title>
        <inputs>
            List of inputs of the algorithm (to be replaced by the actual inputs)
        </inputs>
        <outputs>
            List of outputs of the algorithm (to be replaced by the actual outputs)
        </outputs>
        <steps>
            The full description of the algorithm step by step that will be translate in any programming language. At least 10 steps. (to be replaced by the actual description)
        </steps>
    </algorithm>
    <algorithm id="2">
        <title>
            Title of the algorithm (to be replaced by the actual title)
        </title>
        <inputs>
            List of inputs of the algorithm (to be replaced by the actual inputs)
        </inputs>
        <outputs>
            List of outputs of the algorithm (to be replaced by the actual outputs)
        </outputs>
        <steps>
            The full description of the algorithm step by step that will be translate in any programming language. At least 10 steps. (to be replaced by the actual description)
        </steps>
    </algorithm>
    ...
    <algorithm id="n">
        <title>
            Title of the algorithm (to be replaced by the actual title)
        </title>
        <inputs>
            List of inputs of the algorithm (to be replaced by the actual inputs)
        </inputs>
        <outputs>
            List of outputs of the algorithm (to be replaced by the actual outputs)
        </outputs>
        <steps>
            The full description of the algorithm step by step that will be translate in any programming language. At least 10 steps. (to be replaced by the actual description)
        </steps>
    </algorithm>
    <tests>
        List of tests to be performed to ensure the code is working correctly.
    </tests>
</root>
```
Example of response:
```xml
<root>
    <request>
    Write a Python function that generates the Fibonacci numbers.
    </request>
    <material>
    In mathematics, the Fibonacci sequence is a sequence in which each element is the sum of the two elements that precede it. Numbers that are part of the Fibonacci sequence are known as Fibonacci numbers, commonly denoted Fn. Many writers begin the sequence with 0 and 1, although some authors start it from 1 and 1 and some (as did Fibonacci) from 1 and 2. Starting from 0 and 1, the sequence begins:

    0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, ... (sequence A000045 in the OEIS)

    A tiling with squares whose side lengths are successive Fibonacci numbers: 1, 1, 2, 3, 5, 8, 13 and 21

    The Fibonacci numbers were first described in Indian mathematics as early as 200 BC in work by Pingala on enumerating possible patterns of Sanskrit poetry formed from syllables of two lengths. They are named after the Italian mathematician Leonardo of Pisa, also known as Fibonacci, who introduced the sequence to Western European mathematics in his 1202 book Liber Abaci.

    Fibonacci numbers appear unexpectedly often in mathematics, so much so that there is an entire journal dedicated to their study, the Fibonacci Quarterly. Applications of Fibonacci numbers include computer algorithms such as the Fibonacci search technique and the Fibonacci heap data structure, and graphs called Fibonacci cubes used for interconnecting parallel and distributed systems. They also appear in biological settings, such as branching in trees, the arrangement of leaves on a stem, the fruit sprouts of a pineapple, the flowering of an artichoke, and the arrangement of a pine cone's bracts, though they do not occur in all species.

    Fibonacci numbers are also strongly related to the golden ratio: Binet's formula expresses the n-th Fibonacci number in terms of n and the golden ratio, and implies that the ratio of two consecutive Fibonacci numbers tends to the golden ratio as n increases. Fibonacci numbers are also closely related to Lucas numbers, which obey the same recurrence relation and with the Fibonacci numbers form a complementary pair of Lucas sequences.

    The Fibonacci numbers may be defined by the recurrence relation F[0]=0, F[1]=1, and F[n]=F[n-1]+F[n-2] for n > 1.

    The first 20 Fibonacci numbers Fn are: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181.

    </material>
    <algorithm id="1">
        <title>
            Fibonacci function
        </title>
        <inputs>
            n: the number of Fibonacci numbers to generate
        </inputs>
        <outputs>
            A list of the first `n` Fibonacci numbers
        </outputs>
        <steps>
            1. Define a function named `fibonacci` that takes an integer `n` as input.
            2. Initialize two variables `a` and `b` to 0 and 1 respectively.
            3. Use a loop to iterate from 0 to `n-1`.
            4. In each iteration, calculate the next Fibonacci number by adding `a` and `b`. We recall that the Fibonacci sequence is defined as F(n) = F(n-1) + F(n-2) with F(0) = 0 and F(1) = 1.
            5. Update `a` and `b` to the next two Fibonacci numbers.
            6. Return the list of Fibonacci numbers.
        </steps>
    </algorithm>
    <tests>
        1. Test the function with different values of `n` to ensure it work correctly. For example, test with `n=5` and `n=10` where the expected output is `[0, 1, 1, 2, 3, 5]` and `[0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]` respectively.
        2. Test the function with negative values of `n` to ensure it handles invalid input correctly. For example, test with `n=-1` where the expected output is an error message.
    </tests>
</root>
```


"""


class Analyst(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
        ANALYST_MODEL = os.getenv("ANALYST_MODEL")
        ANALYST_EMBED_MODEL = os.getenv("EMBED_MODEL")
        CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR")

        embeddings = OllamaEmbeddings(
            model=ANALYST_EMBED_MODEL, base_url=OLLAMA_BASE_URL
        )

        self.vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
            collection_name="langgraph-rag",
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PROMPT),
                ("human", "The user request is : {task}. Tell me more about it."),
                ("ai", "Of course ! Here is what I found : {context}"),
                (
                    "human",
                    "Now it's time to implement the request: {task}. Could you give me a detailed description of the algorithms to implement ?",
                ),
            ]
        )
        llm = ChatOllama(
            model=ANALYST_MODEL,
            temperature=0.2,
            num_ctx=32768,
            seed=42,
            base_url=OLLAMA_BASE_URL,
        )
        self.analyst_llm = self.prompt | llm

        # ollama_model = OpenAIModel(
        #     model_name=ANALYST_MODEL,
        #     provider=OpenAIProvider(base_url=OLLAMA_BASE_URL + "/v1/"),
        # )
        # self.analyst_llm = Agent(
        #     model=ollama_model,
        #     system_prompt=PROMPT,
        #     instrument=True,
        #     tools=[duckduckgo_search_tool()],
        #     model_settings=OpenAIModelSettings(
        #         temperature=0.0,
        #     ),
        #     retries=4,
        # )

        # self.analyst_llm.instrument_all()

    def invoke(self, state, **kwargs):
        """Invoke the agent with the given question and context."""
        # retriever = self.vectorstore.as_retriever()
        # context = retriever.invoke(question)
        # context = "\n".join([c.page_content for c in context])
        # # fmt: off
        # question_with_context = dedent(f"""
        # You can use the following context to help you answer the question:

        # <context>
        # {context}
        # </context>

        # The question is:

        # <question>
        # {question}
        # </question>

        # """
        # )
        # # fmt: on
        # print(question_with_context)
        # # return self.analyst_llm.invoke({"question": question_with_context})
        # # return self.agent.run_sync(question_with_context, result_type=Steps, **kwargs)
        # # return self.agent.run_sync(question_with_context, **kwargs)
        question = state["messages"][-1]
        print("**** Analyst ****")
        print("question: ", question)

        print(self.prompt.invoke({"task": question, "context": state["websearch"]}))
        response = self.analyst_llm.invoke(
            {"task": question, "context": state["websearch"]}, **kwargs
        )
        # length, _ = extract_code_blocks(response.content)
        # while length == 0 and iteration < 3:
        #     print("iteration: ", iteration)
        #     response = self.analyst_llm.invoke(
        #         {
        #             "task": question,
        #             "websearch": state["websearch"],
        #             "extra": f"you're previous answer doesn't follow the structure: {structure} \n please changed it accordingly",
        #         },
        #         **kwargs,
        #     )
        #     length, _ = extract_code_blocks(response.content)
        #     with open(
        #         os.path.join(os.getcwd(), f"output/analyst_response_{iteration}.md"),
        #         "w",
        #     ) as f:
        #         f.write(response.content)
        #     iteration += 1

        # if iteration == 3:
        #     raise Exception("No code blocks found in response")

        # message = f"""
        # The request is: {question.content}

        # We already made some research on the web. The results are:
        # {state["websearch"]}

        # """

        # response = self.analyst_llm.run_sync(message, **kwargs)

        with open(os.path.join(os.getcwd(), "output/analyst_response.md"), "w") as f:
            f.write(response.content)
        # print("response: ", response)
        return {"messages": response}
