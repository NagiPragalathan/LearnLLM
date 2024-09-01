from g4f import models, Provider
from langchain.llms.base import LLM
from langchain_g4f import G4FLLM


llm: LLM = G4FLLM(
    model=models.gpt_4o,
    provider=Provider.MetaAI
)

name = llm("I have a dog pet and i want a cool name for it. Suggest me coll names for my pet.")
print(name)
