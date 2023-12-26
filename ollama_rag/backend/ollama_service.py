"""ollama Backend Service."""
import os
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.text_splitter import NLTKTextSplitter
from langchain.vectorstores import Qdrant
from loguru import logger
from omegaconf import DictConfig
from qdrant_client import QdrantClient, models

from ollama_rag.utils.configuration import load_config
from ollama_rag.utils.utility import generate_prompt



load_dotenv()

class OllamaService:
    @load_config(location="config/main.yml")
    def __init__(self, cfg: DictConfig, collection_name: str):
        self.cfg = cfg
        self.collection_name = collection_name
        self.embedding = OllamaEmbeddings(
            base_url=cfg.ollama_embeddings.url, model=cfg.ollama_embeddings.model
        )
        qdrant_client = QdrantClient(
            cfg.qdrant.url,
            port=cfg.qdrant.port,
            api_key=os.getenv("QDRANT_API_KEY"),
            prefer_grpc=cfg.qdrant.prefer_grpc,
        )
        self.vector_db = Qdrant(
            client=qdrant_client,
            collection_name=collection_name,
            embeddings=self.embedding,
        )
        self.model = Ollama(base_url=cfg.ollama.url, model=cfg.ollama.model)

    def embedd_documents(self, dir: str) -> None:
        loader = DirectoryLoader(dir, glob="*.pdf", loader_cls=PyPDFLoader)
        splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = loader.load_and_split(splitter)
        logger.info(f"Loaded {len(docs)} documents.")
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        self.vector_db.add_texts(texts=texts, metadatas=metadatas)
        logger.info("SUCCESS: Texts embedded.")

    def embedd_text(self, text: str, file_name: str, seperator: str) -> None:
        text_list: List = text.split(seperator)
        if not text_list[0]:
            text_list.pop(0)
        if not text_list[-1]:
            text_list.pop(-1)
        metadata = file_name
        metadata_list: List = [
            {"source": f"{metadata}_{str(i)}", "page": 0} for i in range(len(text_list))
        ]
        self.vector_db.add_texts(texts=text_list, metadatas=metadata_list)
        logger.info("SUCCESS: Text embedded.")

    def summarize_text(self, text: str) -> str:
        prompt = generate_prompt(
            prompt_name="openai-summarization.j2", text=text, language="de"
        )
        return self.model(prompt)

    def completion_text(self, prompt: str) -> str:
        return self.model(prompt)

    def search_documents(
        self, query: str, amount: int, threshold: float = 0.0
    ) -> List[Tuple[Document, float]]:
        docs = self.vector_db.similarity_search_with_score(
            query=query, k=amount, score_threshold=threshold
        )
        logger.info("SUCCESS: Documents found.")
        return docs

    def qa(
        self,
        documents: list[tuple[Document, float]],
        query: str,
        summarization: bool = False,
        language: str = "de",
    ):
        if len(documents) == 1:
            text = documents[0][0].page_content
            meta_data = documents[0][0].metadata
        else:
            texts = [doc[0].page_content for doc in documents]
            if summarization:
                text = "".join(self.summarize_text(t) for t in texts)
            else:
                text = " ".join(texts)
            meta_data = [doc[0].metadata for doc in documents]
            meta_data = {k: v for d in meta_data for k, v in d.items()}
        prompt = generate_prompt("ollama-qa.j2", text=text, query=query, language="en")
        try:
            logger.info("starting completion")
            answer = self.completion_text(prompt)
            logger.info(f"completion done with answer {answer}")
        except ValueError as e:
            if e.args[0] == "PROMPT_TOO_LONG":
                logger.info("Prompt too long. Summarizing.")
                short_text = self.summarize_text(text)
                prompt = generate_prompt(
                    "ollama-qa.j2", text=short_text, query=query, language=language
                )
                answer = self.completion_text(prompt)
            logger.error(f"Error: {e}")
        logger.info(f"Answer: {answer}, Meta Data: {meta_data}, Prompt: {prompt}")
        return answer, prompt, meta_data



if __name__ == "__main__":
    initialize_ollama_vector_db()

    ollama = OllamaService(collection_name="ollama")
    ollama.embedd_documents(dir="tests/resources/")

    # print(f'Summary: {summarize_text_ollama(text="Das ist ein Test.")}')

    # print(f'Completion: {completion_text_ollama(text="Das ist ein Test.", query="Was ist das?")}')

    # answer, prompt, meta_data = qa_ollama(
    #     documents=search_documents_ollama(query="Das ist ein Test.", amount=3),
    #     query="Was ist das?",
    # )

    # logger.info(f"Answer: {answer}")
    # logger.info(f"Prompt: {prompt}")
    # logger.info(f"Metadata: {meta_data}")
