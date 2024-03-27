import streamlit as st
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama


reader = SimpleDirectoryReader(input_files=['data\\Omeka-Ebook.pdf'])
documents = reader.load_data()

# bge embedding model
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

# ollama
Settings.llm = Ollama(model="mistral", request_timeout=30.0)

index = VectorStoreIndex(documents)

query_engine = index.as_query_engine()

question = st.text_input('O que você gostaria de saber com base no livro?')

if question:
    answer = query_engine.query(question)
    st.write(answer)