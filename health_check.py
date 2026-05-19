import sys

from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv(override=True)

from scripts.config import (
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_VECTOR_NAME,
    QDRANT_URL,
)


def fail(message: str) -> int:
    print(f"[FAIL] {message}")
    return 1


def main() -> int:
    if not QDRANT_URL:
        return fail("QDRANT_URL is not set.")

    print("[OK] Environment loaded")
    print(f"[OK] Qdrant URL: {QDRANT_URL}")
    print(f"[OK] Qdrant collection: {QDRANT_COLLECTION_NAME}")
    print(f"[OK] Qdrant vector name: {QDRANT_VECTOR_NAME or '<default>'}")
    print(f"[OK] Qdrant API key configured: {'yes' if QDRANT_API_KEY else 'no'}")
    print(f"[OK] Embedding provider: {EMBEDDING_PROVIDER}")
    print(f"[OK] Embedding model: {EMBEDDING_MODEL}")
    print(f"[OK] Ollama base URL: {OLLAMA_BASE_URL}")

    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)

        if not client.collection_exists(QDRANT_COLLECTION_NAME):
            return fail(f"Collection '{QDRANT_COLLECTION_NAME}' does not exist.")

        info = client.get_collection(QDRANT_COLLECTION_NAME)
    except UnexpectedResponse as error:
        return fail(f"Qdrant rejected the request: {error}")
    except ResponseHandlingException as error:
        return fail(f"Could not connect to Qdrant at {QDRANT_URL}: {error}")
    except Exception as error:
        return fail(f"Qdrant health check error: {error}")

    point_count = info.points_count
    vector_size = None

    vectors = info.config.params.vectors
    if hasattr(vectors, "size"):
        vector_size = vectors.size
    elif isinstance(vectors, dict):
        vector_config = vectors.get(QDRANT_VECTOR_NAME) if QDRANT_VECTOR_NAME else next(iter(vectors.values()), None)
        if hasattr(vector_config, "size"):
            vector_size = vector_config.size

    print("[OK] Connected to Qdrant")
    print(f"[OK] Points in collection: {point_count}")
    if vector_size:
        print(f"[OK] Vector size: {vector_size}")

    if point_count == 0:
        return fail("Collection exists but has 0 points. Ingestion did not reach this collection.")

    print("[OK] Deployment health check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
