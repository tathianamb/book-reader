from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama


class LLM:
    def __init__(self, index_path):
        self.llm_predictor = Ollama(model="mistral", request_timeout=30.0)
        self.index = VectorStoreIndex.from_documents(index_path)

    def query(self, question):
        return self.index.query(question)
    
