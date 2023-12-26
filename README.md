# RAG with Ollama

## Installation

Please install Ollama form here: https://ollama.ai.


Then you need to download the model that you want to use. The easiest way is to open your terminal and run the following command:

```bash
ollama run zephyr
```

Then you can install the project with the following command:

```bash
git clone https://github.com/mfmezger/ollama-rag.git
cd ollama-rag

poetry install # if you have not yet installed poetry run: pip install poetry

# you need to start the vector database using docker

docker compose up qdrant

poetry run chainlit run -w  ollama_rag/app.py
```

You can upload your own pdfs in the web interface. They will then be loaded in the Qdrant Vectordatabase.


### Docker setup
Will follow soon.


## Usage
![Screenshot](res/GUI.png)
