FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir . && useradd --system --uid 10001 app
USER app

EXPOSE 8000
CMD ["evidenceforge", "serve", "--host", "0.0.0.0", "--port", "8000"]
