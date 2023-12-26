"""This is the utility module."""
import os
import uuid

from langchain.prompts import PromptTemplate
from langchain.text_splitter import NLTKTextSplitter
from loguru import logger


def combine_text_from_list(input_list: list) -> str:
    """Combines all strings in a list to one string.

    Args:
        input_list (list): List of strings

    Raises:
        TypeError: Input list must contain only strings

    Returns:
        str: Combined string
    """
    # iterate through list and combine all strings to one
    combined_text = ""

    logger.info(f"List: {input_list}")

    for text in input_list:
        # verify that text is a string
        if isinstance(text, str):
            # combine the text in a new line
            combined_text += "\n".join(text)

        else:
            raise TypeError("Input list must contain only strings")

    return combined_text


def generate_prompt(prompt_name: str, text: str, query: str = "", language: str = "de") -> str:
    """Generates a prompt for the Luminous API using a Jinja template.

    Args:
        prompt_name (str): The name of the file containing the Jinja template.
        text (str): The text to be inserted into the template.
        query (str): The query to be inserted into the template.
        language (str): The language the query should output.

    Returns:
        str: The generated prompt.

    Raises:
        FileNotFoundError: If the specified prompt file cannot be found.
    """
    try:
        match language:
            case "en":
                lang = "en"
            case "de":
                lang = "de"
            case _:
                raise ValueError("Language not supported.")
        with open(os.path.join("prompts", lang, prompt_name)) as f:
            prompt = PromptTemplate.from_template(f.read(), template_format="jinja2")
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file '{prompt_name}' not found.")

    if query:
        prompt_text = prompt.format(text=text, query=query)
    else:
        prompt_text = prompt.format(text=text)

    return prompt_text


def create_tmp_folder() -> str:
    """Creates a temporary folder for files to store.

    Returns:
        str: The directory name.
    """
    # Create a temporary folder to save the files
    tmp_dir = f"tmp_{str(uuid.uuid4())}"
    os.makedirs(tmp_dir)
    logger.info(f"Created new folder {tmp_dir}.")
    return tmp_dir


def get_token(token: str | None, llm_backend: str | None, aleph_alpha_key: str | None, openai_key: str | None) -> str:
    """Get the token from the environment variables or the parameter.

    Args:
        token (str, optional): Token from the REST service.
        llm_backend (str): LLM provider. Defaults to "openai".

    Returns:
        str: Token for the LLM Provider of choice.

    Raises:
        ValueError: If no token is provided.
    """
    env_token = aleph_alpha_key if llm_backend in {"aleph-alpha", "aleph_alpha", "aa"} else openai_key
    if not env_token and not token:
        raise ValueError("No token provided.")  #

    return token or env_token  # type: ignore


def split_text(text: str, splitter: NLTKTextSplitter):
    """Split the text into chunks.

    Args:
        text (str): input text.

    Returns:
        List: List of splits.
    """
    # define the metadata for the document
    splits = splitter.split_text(text)
    return splits


def count_tokens(text: str, tokenizer):
    """Count the number of tokens in the text.

    Args:
        text (str): The text to count the tokens for.

    Returns:
        int: Number of tokens.
    """
    tokens = tokenizer.encode(text)
    return len(tokens)


if __name__ == "__main__":
    # test the function
    generate_prompt("qa.j2", "This is a test text.", "What is the meaning of life?")
