from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from g4f import models, Provider
from LLM.MyLLM import G4FLLM
from langchain.llms.base import LLM


def llm() -> LLM:
    llm: LLM = G4FLLM(
        model=models.gpt_4o,
        provider=Provider.Chatgpt4o
    )
    return  llm

def Memory(llm) -> ConversationChain:
    Convomem = ConversationBufferMemory()
    connector = ConversationChain(llm=llm,memory=Convomem)
    SystemWork = """You are a Ai to understand tamil+english and reply to user in tamil+english
    Note: Dont give me explanation in English
    Response should be in: {'english explanation':'content', 'tamil+english': 'reply content'}
    """
    Convomem.save_context({'role': 'system'}, {'content': SystemWork})
    return connector

def VangaSir():
    connector = Memory(llm())
    while True:
        usr_inp = input("Soluga ena veanum >>> ")
        if usr_inp == 'onu veana':
            print("Sare moodetu pooda :)")
        else:
            print("laksmi >>",connector.run(input=usr_inp))

VangaSir()