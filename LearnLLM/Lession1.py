from langchain.llms.base import LLM
from langchain_g4f import G4FLLM
from g4f import Provider, models


a=G4FLLM(
        model=models.gpt_4o,
        provider=Provider.Chatgpt4o, 
)

print(a.invoke("write the code for snake game in python"))

