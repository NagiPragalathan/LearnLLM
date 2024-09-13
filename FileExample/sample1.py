from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from g4f import models, Provider
from langchain_g4f import G4FLLM
from langchain.llms.base import LLM

llm: LLM = G4FLLM(
    model=models.gpt_4o,
    provider=Provider.Chatgpt4o
)

# Load a document
loader = UnstructuredFileLoader("./your_document.txt")
documents = loader.load()

# Split the document into manageable chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

# Set up the chain with a summarization template
summary_template = """Summarize the following text:
{text}
Summary:"""
prompt = PromptTemplate(template=summary_template, input_variables=["text"])

# Create the LangChain summarization chain
summarization_chain = LLMChain(prompt=prompt, llm=llm)


overall = ""

# Summarize each chunk
for chunk in chunks:
    summary = summarization_chain.run({"text": chunk.page_content})
    overall = overall + summary
    
print(overall)

# Define a prompt template for asking questions about the summary
qa_template = """Based on the following summary:
{summary}
Question: {question}
Answer:"""
qa_prompt = PromptTemplate(template=qa_template, input_variables=["summary", "question"])

# Create the LangChain QA chain
qa_chain = LLMChain(prompt=qa_prompt, llm=llm)

# Ask a question
question = "What is the main theme of the document?"
answer = qa_chain.run({"summary": overall, "question": question})

print(f"Question: {question}")
print(f"Answer: {answer}")
