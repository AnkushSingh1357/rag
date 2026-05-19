# Deployment

This project can run as a one-shot Docker job that writes the report into
`research_outputs/<AGENT_USER_ID>/<AGENT_THREAD_ID>/final_report.md`.

## 1. Required environment

Keep secrets in `.env`. For Qdrant Cloud, these must be set:

```env
QDRANT_URL=https://your-cluster-url
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION_NAME=financial_docs-qwen
```

The deployed app also needs:

```env
GOOGLE_API_KEY=your-google-api-key
OLLAMA_BASE_URL=http://your-ollama-host:11434
EMBEDDING_MODEL=qwen3-embedding:4b
EMBEDDING_DIMENSION=2560
```

Use the same embedding model and dimension that were used during ingestion.

## 2. Build

```powershell
docker compose build researcher
```

## 3. Check Qdrant access

```powershell
docker compose run --rm researcher python health_check.py
```

The check should show the collection exists and has points.

## 4. Run the deep agent

```powershell
docker compose run --rm researcher
```

To override the question for one run:

```powershell
docker compose run --rm researcher python main_deepagents.py "What was Amazon's revenue in Q1 2024?"
```

## Optional: local Qdrant

Only start the bundled local Qdrant when you are not using Qdrant Cloud:

```powershell
docker compose --profile local-qdrant up -d qdrant
```
