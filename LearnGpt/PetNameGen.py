from langchain.llms.base import LLM
from g4f import models, Provider
from langchain_g4f import G4FLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm: LLM = G4FLLM(
    model=models.gpt_4o,
    provider=Provider.MetaAI
) 

def genName(llm, petType):
    template = "I have a {petType}. I want to choose the name for it. so suggest me some names"
    prompt = PromptTemplate(template=template, input_variables=["petType"])
    Connect = LLMChain(llm=llm, prompt=prompt)
    names = Connect.run({"petType":petType})
    return names

print(llm.invoke("I have a cat. I want to choose the name for it. so suggest me some names"))

# print(genName(llm, "cat"))
