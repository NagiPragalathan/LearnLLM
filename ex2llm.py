from g4f import Provider, models
from langchain.llms.base import LLM
from langchain_g4f import G4FLLM

def main():
    # Replace Provider.Bing with a provider that supports gpt-3.5-turbo
    llm: LLM = G4FLLM(
        model=models.gpt_35_turbo,
        provider=Provider.MetaAI,  # Assuming OpenAI is a valid provider for gpt-3.5-turbo
    )

    res = llm.invoke("hello")  # Use .invoke instead of __call__()
    print(res)  # Hello! How can I assist you today?

if __name__ == "__main__":
    main()
