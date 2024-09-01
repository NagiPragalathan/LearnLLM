from langchain.embeddings import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings()
embedded_query = embeddings_model.embed_query("Who created the world wide web?")
embedded_query[:5]