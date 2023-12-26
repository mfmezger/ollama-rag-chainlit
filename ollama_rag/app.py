"""FastAPI Backend & Chainlit for the Ollama RAG."""
import os
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

import chainlit as cl
from chainlit.server import app
from dotenv import load_dotenv
from fastapi import File, UploadFile
from langchain import hub
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from loguru import logger

from ollama_rag.backend.ollama_service import OllamaService
from ollama_rag.data_model.response_data_model import EmbeddingResponse
from ollama_rag.utils.utility import create_tmp_folder
from ollama_rag.utils.vdb import get_db_connection, initialize_ollama_vector_db

load_dotenv()

# make sure the qdrant collection exists
initialize_ollama_vector_db()

ollama_service = OllamaService(collection_name="ollama")


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
    """Initialize the LLM."""
    llm = Ollama(
        model="zephyr",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    return llm


def retrieval_qa_chain(llm, vectorstore):
    """Setup the retrieval QA chain."""
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True,
    )
    return qa_chain


def qa_agent():
    """Initialize the QA Agent."""
    llm = initialize_llm()
    vectorstore = get_db_connection(collection_name="ollama")

    qa = retrieval_qa_chain(llm, vectorstore)
    return qa


@cl.on_chat_start
async def start():
    """Start the chat."""
    chain = qa_agent()
    msg = cl.Message(content="Starting the RAG.")
    await msg.send()
    msg.content = "Hi you can upload pdfs and start asking questions."
    await msg.update()
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    """Main message handler."""
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"])
    cb.answer_reached = True

    # use pathlib to check if the folder input exists if not create it

    Path("input").mkdir(parents=True, exist_ok=True)

    # save all of the message elements as pdf files with uuids
    for element in message.elements:
        with open(f"input/{uuid4()}.pdf", "wb") as f:
            f.write(element.content)

    from ollama_rag.backend.ollama_service import OllamaService

    ollama_service = OllamaService(collection_name="ollama")

    ollama_service.embedd_documents(dir="input/")

    # delete all of the files in the input folder
    import os

    folder = "input/"

    for filename in os.listdir(folder):
        os.remove(os.path.join(folder, filename))

    res = await chain.acall(message.content, callbacks=[cb])
    print(f"response: {res}")
    answer = res["result"]
    answer = answer.replace(".", ".\n")
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources: {str(str(sources))}"
    else:
        answer += "\nNo Sources found"

    await cl.Message(content=answer).send()
