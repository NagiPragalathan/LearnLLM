from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from g4f import Provider, models
from langchain.llms.base import LLM
from langchain_g4f import G4FLLM

# Initialize the LLM with your API key (assuming you're using OpenAI)
def initialize_llm():
    llm: LLM = G4FLLM(
        model=models.gpt_35_turbo,
        provider=Provider.MetaAI, 
    )
    return llm

# Define the prompt template
def create_prompt_template():
    template = "You are a helpful assistant. The user says: {user_input}\n\nYour response:"
    prompt = PromptTemplate(input_variables=["user_input"], template=template)
    return prompt

# Define the chat chain
def create_chat_chain(llm):
    prompt = create_prompt_template()
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain

# Function to start the chat
def start_chat(chain):
    print("Chatbot is ready! Type 'exit' to stop the conversation.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        response = chain.run(user_input=user_input)
        print(f"Bot: {response}\n")

# Main function to run the chat application
def main():
    # Replace with your actual OpenAI API key
    
    llm = initialize_llm()
    chat_chain = create_chat_chain(llm)
    start_chat(chat_chain)

if __name__ == "__main__":
    main()
