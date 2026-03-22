FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

RUN pip install --no-cache-dir .

EXPOSE 8400

ENTRYPOINT ["llm-guard"]
CMD ["--host", "0.0.0.0", "--port", "8400"]
