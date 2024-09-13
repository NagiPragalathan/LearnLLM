from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms.base import LLM

from g4f import models, Provider
# from LLM.MyLLM import G4FLLM
from langchain_g4f import G4FLLM

FormQuestions = """Survey on Women Safety JavaScript isn't enabled in your browser, so this file can't be opened. Enable and reload. Survey on Women Safety We are developing a non-profitable product to ensure every one of our sisters are safe against any form of injustice. Please help us by providing insights to cater our product more efficiently bringing a social reform! (We ensure to maintain an order of non-disclosure for all the information entered) Sign in to Google to save your progress. Learn more * Indicates required question Name: * Your answer Occupational status: * Working Student Age: * Your answer Where do you reside currently? * Your answer What mode of travel do you use to reach your college/workplace? * Private Bus Train On a scale of 10 how much do you think a metrocity is safe for women to travel alone at nights? * 1 2 3 4 5 6 7 8 9 10 Have you previously used or aware of any women safety application? * Yes No Other: How feasible do you think using an SOS application with haptic responses would be? * Your answer In case of an emergency, would it be feasible for you to raise a complaint through an application ? * Yes No Have you ever felt that people around you would judge if you share them with any form of harrasment you have faced in your life? * Yes No Maybe Do you think sharing them with an AI powered therapist would help sort things out? * Your answer Incase of any injustice would you feel comfortable raising a complaint regarding the issue directly to the authorities through an anonymous application? * Yes No Do you have any personal suggestions or implementations that you might need in an SOS application? * Your answer Submit Clear form Never submit passwords through Google Forms. This content is neither created nor endorsed by Google. Report Abuse - Terms of Service - Privacy Policy Forms"""
MyData = """
Enthusiastic Full Stack Developer with a solid foundation in programming, development, large language models (LLM). 
Proven success in hackathons and hands-on experience in both frontend and backend development. Proficient in a variety 
of languages, frameworks, and tools, and passionate about Web3 and blockchain technologies. ...
"""
def llm() -> LLM:
    llm_: LLM = G4FLLM(
        model=models.gpt_35_turbo,
        provider=Provider.MetaAI,
    )
    return llm_

# Define the prompt template to ask the LLM to answer the survey questions based on the given data
prompt_template = """
You are an AI that is supposed to answer the following survey questions. Answer them based on the given user profile and 
data where possible. If the question is not relevant to the data or common, respond as per general logic or by guessing 
based on the provided context.

User Data:
{data}

Survey Questions:
{questions}

Answer the questions as per the user profile.
"""

# Create the LangChain template and chain
prompt = PromptTemplate(
    input_variables=["data", "questions"],
    template=prompt_template
)

# Set up the LLM chain with the custom LLM
chain = LLMChain(llm=llm(), prompt=prompt)

# Prepare the inputs
inputs = {
    "data": MyData,
    "questions": FormQuestions
}

# Run the chain to get the responses
response = chain.run(inputs)

# Print the output
print(response)
