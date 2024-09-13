from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """This is a very long text that we want to process. It might contain multiple paragraphs, sentences, and characters. 
The goal is to split this text into smaller chunks so that we can handle it better with a language model."""

# Initialize the text splitter with a desired chunk size and overlap
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=70)

# Split the text into chunks
chunks = splitter.split_text(text)

# Output the chunks
for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1}: {chunk}\n")
