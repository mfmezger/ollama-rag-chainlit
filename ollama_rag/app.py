"""FastAPI Backend & Chainlit for the Ollama RAG."""
import os
from typing import List, Optional  
from langchain import hub
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import chainlit as cl
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from typing import List, Optional, Tuple
from ollama_rag.utils.vdb import get_db_connection
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.text_splitter import NLTKTextSplitter
from langchain.vectorstores import Qdrant
from loguru import logger
from omegaconf import DictConfig
from qdrant_client import QdrantClient
from chainlit.server import app
from fastapi import Request
from fastapi.responses import (
    HTMLResponse,
)
from ollama_rag.utils.configuration import load_config

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.openapi.utils import get_openapi
from langchain.docstore.document import Document as LangchainDocument
from loguru import logger
from omegaconf import DictConfig
from qdrant_client import QdrantClient, models
from qdrant_client.http.models.models import UpdateResult
from starlette.responses import JSONResponse

from ollama_rag.backend.ollama_service import OllamaService
from ollama_rag.data_model.request_data_model import (
    CustomPromptCompletion,
    EmbeddTextFilesRequest,
    ExplainQARequest,
    QARequest,
    SearchRequest,
)
from ollama_rag.data_model.response_data_model import (
    EmbeddingResponse,
    ExplainQAResponse,
    QAResponse,
    SearchResponse,
)
from ollama_rag.utils.utility import (
    combine_text_from_list,
    create_tmp_folder,
)

load_dotenv()

ollama_service = OllamaService(collection_name="ollama")

@app.get("/hello")
def hello(request: Request):
    print(request.headers)
    return HTMLResponse("Hello Woasfasdfsadfrld")


@app.post("/embeddings/documents")
async def post_embedd_documents(
    files: List[UploadFile] = File(...),
    token: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> EmbeddingResponse:
    """Uploads multiple documents to the backend.

    Args:
        files (List[UploadFile], optional): Uploaded files. Defaults to File(...).

    Returns:
        JSONResponse: The response as JSON.
    """
    logger.info("Embedding Multiple Documents")
    tmp_dir = create_tmp_folder()

    file_names = []

    for file in files:
        file_name = file.filename
        file_names.append(file_name)

        # Save the file to the temporary folder
        if tmp_dir is None or not os.path.exists(tmp_dir):
            raise ValueError("Please provide a temporary folder to save the files.")

        if file_name is None:
            raise ValueError("Please provide a file to save.")

        with open(os.path.join(tmp_dir, file_name), "wb") as f:
            f.write(await file.read())

    ollama_service.embedd_documents(dir=tmp_dir)

    return EmbeddingResponse(status="success", files=file_names)


# Set up RetrievelQA model
QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-mistral")


# load the LLM
def initialize_llm():
    llm = Ollama(
        model="zephyr",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    return llm


def retrieval_qa_chain(llm, vectorstore):
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True,
    )
    return qa_chain


def qa_bot():
    llm = initialize_llm()
    vectorstore = get_db_connection(collection_name="ollama")

    qa = retrieval_qa_chain(llm, vectorstore)
    return qa


@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Firing up the research info bot...")
    await msg.send()
    msg.content = "Hi, welcome to research info bot. What is your query?"
    await msg.update()
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    print(f"response: {res}")
    answer = res["result"]
    answer = answer.replace(".", ".\n")
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources: " + str(str(sources))
    else:
        answer += f"\nNo Sources found"

    await cl.Message(content=answer).send()
