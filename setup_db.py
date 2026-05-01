import os
import re
import time
import warnings
from dotenv import load_dotenv

# Suppress warnings and load API keys
warnings.filterwarnings('ignore')
load_dotenv()

# Required Imports for Ingestion
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from scripts.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    EMBEDDING_NUM_CTX,
    INGEST_BATCH_SIZE,
    INGEST_MAX_RETRIES,
    INGEST_SLEEP_SECONDS,
    MAX_INGEST_CHUNKS,
    OLLAMA_BASE_URL,
    QDRANT_COLLECTION_NAME,
    QDRANT_URL,
    RAG_FILE_GLOB,
)

print("Starting Qdrant Database Ingestion...")

def enrich_metadata(documents):
    """Add searchable filing metadata parsed from paths like amazon 10-q q1 2024.md."""
    pattern = re.compile(
        r"(?P<company>[a-z]+)\s+(?P<doc_type>10-k|10-q|8-k)(?:\s+(?P<quarter>q[1-4]))?\s+(?P<year>\d{4})",
        re.IGNORECASE,
    )

    for doc in documents:
        source = doc.metadata.get("source", "")
        filename = os.path.splitext(os.path.basename(source))[0].lower()
        match = pattern.search(filename)
        if not match:
            continue

        doc.metadata["company_name"] = match.group("company").lower()
        doc.metadata["doc_type"] = match.group("doc_type").lower()
        doc.metadata["fiscal_year"] = match.group("year")
        if match.group("quarter"):
            doc.metadata["fiscal_quarter"] = match.group("quarter").lower()

    return documents


def retry_delay_from_error(error: Exception) -> float:
    """Extract provider retry delay from the error text, falling back to one minute."""
    match = re.search(r"retryDelay': '(\d+)s'", str(error))
    if match:
        return float(match.group(1)) + 2.0
    return 62.0


def add_documents_with_retry(vector_store, batch, batch_size):
    for attempt in range(INGEST_MAX_RETRIES + 1):
        try:
            vector_store.add_documents(documents=batch, batch_size=batch_size)
            return
        except Exception as error:
            if attempt == INGEST_MAX_RETRIES:
                raise

            if "RESOURCE_EXHAUSTED" in str(error):
                wait_seconds = retry_delay_from_error(error)
            else:
                wait_seconds = min(5.0 * (attempt + 1), 30.0)

            print(
                f"Embedding request failed. Waiting {wait_seconds:.0f}s before retry "
                f"{attempt + 1}/{INGEST_MAX_RETRIES}..."
            )
            time.sleep(wait_seconds)

# 2. Load Documents (FIXED: Explicitly telling it to use UTF-8 encoding)
print(f"Loading documents from {DATA_DIR}...")
loader = DirectoryLoader(
    DATA_DIR, 
    glob=RAG_FILE_GLOB, 
    loader_cls=TextLoader,
    loader_kwargs={'encoding': 'utf-8'} # <-- FIXED THIS LINE
)
documents = loader.load()
documents = enrich_metadata(documents)
print(f"Loaded {len(documents)} documents.")

# 3. Split Text into Chunks
print("Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=True
)
splits = text_splitter.split_documents(documents)
print(f"Created {len(splits)} chunks.")

if MAX_INGEST_CHUNKS > 0 and len(splits) > MAX_INGEST_CHUNKS:
    print(
        f"Limiting ingestion to first {MAX_INGEST_CHUNKS} chunks "
        f"because MAX_INGEST_CHUNKS is set."
    )
    splits = splits[:MAX_INGEST_CHUNKS]

# 4. Initialize Embeddings
print(f"Initializing Ollama embeddings with {EMBEDDING_MODEL}...")
embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL,
    num_ctx=EMBEDDING_NUM_CTX,
)

# 5. Connect to Qdrant & Create Collection (If it doesn't exist)
print(f"Connecting to Qdrant at {QDRANT_URL}...")
client = QdrantClient(url=QDRANT_URL)

if not client.collection_exists(collection_name=QDRANT_COLLECTION_NAME):
    print(f"Creating collection '{QDRANT_COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE),
    )
else:
    print(f"Collection '{QDRANT_COLLECTION_NAME}' already exists.")

# 6. Ingest Data into Qdrant
print("Ingesting data into Vector Store. This may take a moment...")
vector_store = QdrantVectorStore(
    client=client,
    collection_name=QDRANT_COLLECTION_NAME,
    embedding=embeddings,
)

total_batches = (len(splits) + INGEST_BATCH_SIZE - 1) // INGEST_BATCH_SIZE
for batch_number, start in enumerate(range(0, len(splits), INGEST_BATCH_SIZE), start=1):
    batch = splits[start:start + INGEST_BATCH_SIZE]
    print(f"Ingesting batch {batch_number}/{total_batches} ({len(batch)} chunks)...")
    add_documents_with_retry(vector_store, batch, INGEST_BATCH_SIZE)
    time.sleep(INGEST_SLEEP_SECONDS)

print("\nIngestion Complete! The database is populated.")
print("You can now run your main agent script.")
