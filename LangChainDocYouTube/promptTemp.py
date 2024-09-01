from g4f import models, Provider
from langchain.llms.base import LLM
from langchain_g4f import G4FLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm: LLM = G4FLLM(
    model=models.gpt_4o,
    provider=Provider.MetaAI
)

def generate_name(llm, animal_type):
    temp_str = "I have a {animal_type} pet and i want a cool name for it. Suggest me coll names for my pet."
    prompt_temp = PromptTemplate(
        template=temp_str,
        input_variables=["animal_type"]
    )
    name_chain = LLMChain(llm=llm, prompt=prompt_temp)
    response = name_chain({"animal_type":animal_type})
    return response

print(generate_name(llm, "cat"))