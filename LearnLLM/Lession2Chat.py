from langchain.llms.base import LLM
from langchain_g4f import G4FLLM
from g4f import Provider, models

from langchain.schema import (
    AIMessage,
    SystemMessage,
    HumanMessage
)

llm: LLM = G4FLLM(
        model=models.gpt_4o,
        provider=Provider.Chatgpt4o, 
)

messages = [
    SystemMessage(content="You are a very useful bot to convert given key into music!"),
    HumanMessage(content="cat"),
]
print(llm.invoke(messages))
