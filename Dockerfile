FROM python:3.11

# install poetry and dependencies
# Install Poetry
RUN curl -sSL https://install.python-poetry.org/ | POETRY_HOME=/opt/poetry python && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

# Copy using poetry.lock* in case it doesn't exist yet
COPY ./pyproject.toml ./poetry.lock* ./

RUN poetry install --no-root --no-dev

COPY . .

ENTRYPOINT ["poetry", "run", "chainlit", "run", "-h",  "ollama_rag/app.py"]
# watch the logs
# CMD ["tail", "-f", "/dev/null"]
