"""Script that contains the Pydantic Models for the Rest Request."""
from enum import Enum
from typing import List, Optional

from fastapi import UploadFile
from pydantic import BaseModel, Field


class Language(str, Enum):
    """The Language Enum."""

    DETECT = "detect"
    GERMAN = "de"
    ENGLISH = "en"


class Filtering(BaseModel):
    """The Filtering Model."""

    threshold: float = Field(0.0, title="Threshold", description="The threshold to use for the search.")
    filter: Optional[dict] = Field(None, title="Filter", description="Filter for the database search with metadata.")


class EmbeddTextFilesRequest(BaseModel):
    """The request for the Embedd Text Files endpoint."""

    files: List[UploadFile] = Field(..., description="The list of text files to embed.")
    seperator: str = Field("###", description="The seperator to use between embedded texts.")


class SearchRequest(BaseModel):
    """The request parameters for searching the database."""

    query: str = Field(..., title="Query", description="The search query.")
    filtering: Filtering
    collection_name: Optional[str] = Field(None, title="Name of the Collection", description="Name of the Qdrant Collection.")
    amount: int = Field(1, title="Amount", description="The number of search results to return.")
    token: Optional[str] = Field(None, title="Token", description="The API token for the LLM provider.")


class EmbeddTextRequest(BaseModel):
    """The request parameters for embedding text."""

    text: str = Field(..., title="Text", description="The text to embed.")
    file_name: str = Field(..., title="File Name", description="The name of the file to save the embedded text to.")
    seperator: str = Field("###", title="seperator", description="The seperator to use between embedded texts.")


class CustomPromptCompletion(BaseModel):
    """The Custom Prompt Completion Model."""

    token: str = Field(..., title="Token", description="The API token for the LLM provider.")
    prompt: str = Field(..., title="Prompt", description="The prompt to use for the completion.")
    model: str = Field(..., title="Model", description="The model to use for the completion.")
    max_tokens: int = Field(256, title="Max Tokens", description="The maximum number of tokens to generate.")
    temperature: float = Field(..., title="Temperature", description="The temperature to use for the completion.")
    stop_sequences: List[str] = Field([], title="Stop Sequences", description="The stop sequences to use for the completion.")


class QARequest(BaseModel):
    """Request for the QA endpoint."""

    language: Language = Field(Language.DETECT, title="Language", description="The language to use for the answer.")
    history: Optional[int] = Field(0, title="History", description="The number of previous questions to include in the context.")
    history_list: List[str] = Field([], title="History List", description="A list of previous questions to include in the context.")
    search: SearchRequest


class ExplainQARequest(BaseModel):
    """The request parameters for explaining the output."""

    qa: QARequest = Field(..., title="QA Request", description="The QA Request to explain.")
    threshold_explain: float = Field(0.0, title="Threshold Explain", description="The threshold to use for the explanation.")
