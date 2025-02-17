from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .state import State
from .utils import read_prompt_from_file
from abc import ABC, abstractmethod

llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0.0, num_ctx=2 * 4096)


class DefaultAgent(ABC):
    def __init__(
        self,
        model="qwen2.5-coder:7b",
        temperature=0.0,
        num_ctx=32768,
        prompt=None,
    ):
        # self.agent = ChatOllama(model=model, temperature=temperature, num_ctx=num_ctx)
        self.agent = llm

        messages = [
            MessagesPlaceholder(variable_name="messages"),
        ]
        if prompt:
            messages.insert(0, SystemMessage(read_prompt_from_file(prompt)))
        self.prompt_template = ChatPromptTemplate.from_messages(messages)

    @abstractmethod
    def invoke(self, state: State):
        pass
