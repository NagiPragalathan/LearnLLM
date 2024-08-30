from langchain_core.messages import HumanMessage, SystemMessage
from g4f import Provider, models
from langchain.llms.base import LLM
from langchain_g4f import G4FLLM

from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

model: LLM = G4FLLM(
        model=models.gpt_35_turbo,
        provider=Provider.MetaAI, 
    )

messages = [
    SystemMessage(content="Translate the following from English into Italian"),
    HumanMessage(content="hi!"),
]

result = model.invoke(messages)
print(parser.invoke(result))

