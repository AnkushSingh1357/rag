import os


def env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


DATA_DIR = env_str("RAG_DATA_DIR", "./data/rag-data/")
RAG_FILE_GLOB = env_str("RAG_FILE_GLOB", "markdown/**/*.md")
QDRANT_URL = env_str("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION_NAME = env_str("QDRANT_COLLECTION_NAME", "financial_docs-qwen")

OLLAMA_BASE_URL = env_str("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = env_str("EMBEDDING_MODEL", "qwen3-embedding:4b")
EMBEDDING_DIMENSION = env_int("EMBEDDING_DIMENSION", 2560)
EMBEDDING_NUM_CTX = env_int("EMBEDDING_NUM_CTX", 1024)

INGEST_BATCH_SIZE = env_int("INGEST_BATCH_SIZE", 10)
INGEST_SLEEP_SECONDS = env_float("INGEST_SLEEP_SECONDS", 7.0)
INGEST_MAX_RETRIES = env_int("INGEST_MAX_RETRIES", 5)
MAX_INGEST_CHUNKS = env_int("MAX_INGEST_CHUNKS", 0)
CHUNK_SIZE = env_int("CHUNK_SIZE", 1000)
CHUNK_OVERLAP = env_int("CHUNK_OVERLAP", 200)

GEMINI_CHAT_MODEL = env_str("GEMINI_CHAT_MODEL", "gemini-2.5-pro")
MAIN_CHECKPOINT_DB = env_str("MAIN_CHECKPOINT_DB", "data/deep_finance_researcher.db")
DEEP_AGENT_CHECKPOINT_DB = env_str(
    "DEEP_AGENT_CHECKPOINT_DB",
    "data/deep_agent_finance_researcher.db",
)

AGENT_QUERY = env_str(
    "AGENT_QUERY",
    "Do a detailed analysis of Amazon's financial performance in 2023 and 2024",
)
DEEP_AGENT_QUERY = env_str("DEEP_AGENT_QUERY", "What was Amazon's revenue in Q1 2024?")
AGENT_USER_ID = env_str("AGENT_USER_ID", "default_user")
AGENT_THREAD_ID = env_str("AGENT_THREAD_ID", "default_thread")
RESEARCH_OUTPUT_DIR = env_str("RESEARCH_OUTPUT_DIR", "research_outputs")
