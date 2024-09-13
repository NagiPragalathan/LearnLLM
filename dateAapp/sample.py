from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from g4f import models, Provider
from langchain_g4f import G4FLLM
from langchain.llms.base import LLM


def llm() -> LLM:
    llm = G4FLLM(
        model=models.gpt_4o,
        provider=Provider.Bing
    )
    return llm

def Memory(llm: LLM) -> ConversationChain:
    Convomem = ConversationBufferMemory()
    
    # Forceful and explicit instructions in Note
    SystemWork = """You are an AI that understands and replies in Tainglish.
    Note:
    - imagine your name is banu and you are a romantic girl 
    - Do not use Tamil script at all. Avoid any Tamil letters in your response.
    - Do not provide any explanations in English. Just respond in Tainglish.
    - Only respond in Tainglish, which is Tamil written in English letters. 
    """

    Convomem.save_context({'role': 'system'}, {'content': SystemWork})

    connector = ConversationChain(llm=llm, memory=Convomem)
    return connector

def VangaSir():
    connector = Memory(llm())
    while True:
        usr_inp = input("Soluga ena veanum >>> ")
        if usr_inp.lower() == 'onu veana':
            print("Sare moodetu pooda :)")
            break  # Exiting the loop to stop the conversation
        else:
            response = connector.run(input=usr_inp)
            print("laksmi >>", response)

VangaSir()
