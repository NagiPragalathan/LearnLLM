import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from g4f import models, Provider
from langchain_g4f import G4FLLM
from langchain.llms.base import LLM
import pickle

# Step 1: Extract Text from the PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Step 2: Chunk the Text
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    return chunks

# Step 3: Initialize the GPT Model
def initialize_gpt_model():
    llm = G4FLLM(
        model=models.gpt_4o,
        provider=Provider.Chatgpt4o
    )
    return llm

# Step 4: Create or Load Embeddings for Each Chunk
def create_or_load_embeddings(chunks, embedding_file="embeddings.pkl", index_file="faiss_index.index"):
    if os.path.exists(embedding_file) and os.path.exists(index_file):
        # Load embeddings and FAISS index from disk
        with open(embedding_file, "rb") as f:
            embeddings = pickle.load(f)
        index = faiss.read_index(index_file)
        model = SentenceTransformer('all-MiniLM-L6-v2')  # Initialize the model for querying
    else:
        # Create embeddings
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(chunks, convert_to_numpy=True)

        # Create a FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        # Save embeddings and index to disk
        with open(embedding_file, "wb") as f:
            pickle.dump(embeddings, f)
        faiss.write_index(index, index_file)

    return index, embeddings, model

# Step 5: Query the Document
def query_document(index, embeddings, model, chunks, question, llm):
    # Convert the question into an embedding
    question_embedding = model.encode([question], convert_to_numpy=True)

    # Search the FAISS index for the most similar chunks
    _, indices = index.search(question_embedding, k=5)  # Get top 5 relevant chunks

    # Retrieve the most relevant chunks
    relevant_chunks = [chunks[i] for i in indices[0]]

    # Combine retrieved chunks
    combined_docs = " ".join(relevant_chunks)

    # Create a QA chain
    qa_template = """You are a knowledgeable assistant. Based on the following content:
    {content}
    Answer the question:
    Question: {question}
    Answer:"""
    qa_prompt = PromptTemplate(template=qa_template, input_variables=["content", "question"])
    qa_chain = LLMChain(prompt=qa_prompt, llm=llm)
    
    # Get the answer based on the combined relevant chunks
    answer = qa_chain.run({"content": combined_docs, "question": question})
    return answer

# Main Execution
if __name__ == "__main__":
    # Specify your PDF file path
    pdf_path = "C:/Users/nagip/Downloads/Programming Persistent Memory, Steve Scargall.pdf"

    # Extract text from the PDF
    book_text = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(book_text)} characters from the book.")

    # Chunk the text
    chunks = chunk_text(book_text)
    print(f"Split the book into {len(chunks)} chunks.")

    # Initialize the GPT model
    llm = initialize_gpt_model()

    # Create or load embeddings and vector store
    index, embeddings, model = create_or_load_embeddings(chunks)
    print("Created or loaded embeddings for the document.")

    # Example usage: Ask a question
    question = "what is mmap() and where it's located?"
    answer = query_document(index, embeddings, model, chunks, question, llm)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
