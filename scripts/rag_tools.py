# scripts/rag_tools.py
import os
import sys
import json
import re
import subprocess
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv(override=True)

from scripts.llm_utils import get_rotating_llm
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.tools import tool
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import CrossEncoder
import logging

# Suppress verbose sentence_transformers output
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# Load lightweight re-ranker globally (loads once on startup)
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)

from scripts.schema import ChunkMetadata
from scripts.config import (
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_VECTOR_NAME,
    QDRANT_URL,
    GROQ_API_KEY,
    GROQ_MODEL_NAME,
)

# ─────────────────────────────────────────────
# LLM — use fast/cheap model for filter extraction only
# ─────────────────────────────────────────────
_fast_llm = get_rotating_llm(
    model_name=os.getenv("GROQ_FAST_MODEL", GROQ_MODEL_NAME),
    temperature=0,
    max_tokens=128,          # ✅ OPT: filter extraction needs very few tokens
)

# Full LLM for research tasks
llm = get_rotating_llm(
    model_name=GROQ_MODEL_NAME,
    temperature=0,
    max_tokens=1024,         # ✅ OPT: cap output to reduce cost
)

# ─────────────────────────────────────────────
# Embeddings + Vector Store (singletons)
# ─────────────────────────────────────────────
if EMBEDDING_PROVIDER.lower() == "fastembed":
    embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
else:
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=QDRANT_COLLECTION_NAME,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY or None,
    vector_name=QDRANT_VECTOR_NAME or None,
)

# ─────────────────────────────────────────────
# Filter extraction with caching
# ─────────────────────────────────────────────
_FILTER_PROMPT = """\
Extract metadata filters from this query. Return None for fields not mentioned.

COMPANY MAP: Amazon/AMZN→amazon, Google/Alphabet→google, Apple/AAPL→apple,
Microsoft/MSFT→microsoft, Tesla/TSLA→tesla, Nvidia/NVDA→nvidia, Meta/FB→meta

DOC TYPES: annual report→10-k, quarterly→10-q, current→8-k

EXAMPLES:
"Amazon Q3 2024 revenue" → company_name=amazon, doc_type=10-q, fiscal_year=2024, fiscal_quarter=q3
"Apple 2023 annual" → company_name=apple, doc_type=10-k, fiscal_year=2023

Query: {query}"""


@lru_cache(maxsize=256)   # ✅ OPT: identical queries skip LLM entirely
def _cached_extract_filters(query: str) -> str:
    """Return JSON string of extracted filters (cached)."""
    structured = _fast_llm.with_structured_output(ChunkMetadata)
    metadata   = structured.invoke(_FILTER_PROMPT.format(query=query))
    filters    = metadata.model_dump(exclude_none=True) if metadata else {}
    return json.dumps(filters, sort_keys=True)


def extract_filters(query: str) -> dict:
    if os.getenv("USE_LLM_FILTERS", "false").lower() == "true":
        return json.loads(_cached_extract_filters(query))

    query_lower = query.lower()
    filters = {}
    company_aliases = {
        "amazon": "amazon",
        "amzn": "amazon",
        "apple": "apple",
        "aapl": "apple",
        "meta": "meta",
        "facebook": "meta",
        "microsoft": "microsoft",
        "msft": "microsoft",
        "google": "google",
        "alphabet": "google",
    }
    for alias, company in company_aliases.items():
        if alias in query_lower:
            filters["company_name"] = company
            break

    year_match = re.search(r"\b(20\d{2})\b", query_lower)
    if year_match:
        filters["fiscal_year"] = year_match.group(1)

    quarter_match = re.search(r"\bq([1-4])\b", query_lower)
    if quarter_match:
        filters["fiscal_quarter"] = f"q{quarter_match.group(1)}"

    if "10-k" in query_lower or "annual" in query_lower:
        filters["doc_type"] = "10-k"
    elif "10-q" in query_lower or "quarter" in query_lower:
        filters["doc_type"] = "10-q"
    elif "8-k" in query_lower or "current report" in query_lower:
        filters["doc_type"] = "8-k"

    return filters


def _build_qdrant_filter(filters: dict) -> Filter | None:
    if not filters:
        return None

    if QDRANT_COLLECTION_NAME == "ank-pdf":
        field_map = {
            "company_name": ("metadata.company_name", lambda value: value),
            "doc_type": ("metadata.report_type", lambda value: str(value).upper()),
            "fiscal_year": ("metadata.report_year", lambda value: int(value)),
            "fiscal_quarter": ("metadata.report_quarter", lambda value: str(value).upper()),
        }
    else:
        field_map = {
            "company_name": ("metadata.company_name", lambda value: value),
            "doc_type": ("metadata.doc_type", lambda value: value),
            "fiscal_year": ("metadata.fiscal_year", lambda value: value),
            "fiscal_quarter": ("metadata.fiscal_quarter", lambda value: value),
        }

    conditions = []
    for key, value in filters.items():
        if value is None or key not in field_map:
            continue

        field_name, normalize = field_map[key]
        conditions.append(FieldCondition(key=field_name, match=MatchValue(value=normalize(value))))

    if not conditions:
        return None

    return Filter(must=conditions)


# ─────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────

def core_hybrid_search(query: str, k: int = 1, retrieve_k: int = None, return_metadata: bool = False) -> str | tuple[str, dict]:
    filters        = extract_filters(query)
    qdrant_filter  = _build_qdrant_filter(filters)
    
    # Stage 1: Dense Retrieval (Fetch retrieve_k chunks)
    actual_retrieve_k = retrieve_k if retrieve_k is not None else min(k * 5, 50)
    results    = vector_store.similarity_search(query=query, k=actual_retrieve_k, filter=qdrant_filter)

    if not results:
        empty_msg = "No historical documents found for this query."
        return (empty_msg, {"top_chunk_score": 0.0, "source_count": 0}) if return_metadata else empty_msg

    # Stage 2: Cross-Encoder Re-ranking
    pairs = [[query, doc.page_content] for doc in results]
    scores = reranker_model.predict(pairs)
    
    # Sort descending by relevance score and slice top K
    scored_results = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
    top_results = [(score, doc) for score, doc in scored_results[:k]]

    snippets = []
    unique_sources = set()
    for idx, (score, doc) in enumerate(top_results, start=1):
        source = doc.metadata.get("source", "SEC Filing")
        unique_sources.add(source)
        snippet = doc.page_content[:500].strip()
        snippets.append(f"[{idx}] Source: {source}\nContent: {snippet}")

    final_snippets = "\n\n".join(snippets)
    if return_metadata:
        top_score = _sigmoid(float(top_results[0][0])) if top_results else 0.0
        min_score = _sigmoid(float(top_results[-1][0])) if top_results else 0.0
        return final_snippets, {
            "top_chunk_score": top_score,
            "min_chunk_score": min_score,
            "source_count": len(unique_sources)
        }
    return final_snippets

@tool
def hybrid_search(query: str, k: int = 1, retrieve_k: int = None) -> str:
    """Search historical SEC filings (10-K, 10-Q) for financial data.

    Args:
        query: Specific financial question or metric to look up.
        k:     Number of documents to output (default 1 to minimise tokens).
        retrieve_k: Optional custom number of documents to pull from Qdrant before re-ranking.

    Returns:
        Source citation + short content snippet.
    """
    return core_hybrid_search(query, k, retrieve_k, return_metadata=False)

import math

def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))

def calculate_coherence(text: str) -> float:
    """Calculate answer coherence using the cross-encoder on adjacent sentences."""
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text) if len(s.strip()) > 10]
    if len(sentences) < 2:
        return 1.0 # default for very short answers
    
    pairs = [[sentences[i], sentences[i+1]] for i in range(len(sentences)-1)]
    scores = reranker_model.predict(pairs)
    normalized_scores = [_sigmoid(float(s)) for s in scores]
    return float(sum(normalized_scores) / len(normalized_scores))

def calculate_info_density(text: str) -> float:
    """Calculate the ratio of numbers, percentages, and money symbols to words."""
    words = text.split()
    if not words:
        return 0.0
    
    data_points = 0
    for word in words:
        if any(char.isdigit() for char in word) or "$" in word or "%" in word:
            data_points += 1
            
    return float(data_points / len(words))

def calculate_readability(text: str) -> float:
    """Calculate an approximate Flesch Reading Ease score. Lower means harder/more complex."""
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if s.strip()]
    
    if not words or not sentences:
        return 0.0
        
    num_words = len(words)
    num_sentences = len(sentences)
    
    # Very crude syllable counting approximation (vowels)
    num_syllables = sum(len(re.findall(r'[aeiouy]+', word, re.IGNORECASE)) for word in words)
    # Ensure at least 1 syllable per word
    num_syllables = max(num_syllables, num_words)
    
    # Flesch Reading Ease formula
    score = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)
    # Normalize to roughly 0.0 - 1.0 (typical score is 0 to 100)
    normalized = max(0.0, min(100.0, score)) / 100.0
    return float(normalized)


@tool
def live_finance_researcher(query: str) -> str:
    """Get live stock data, news, and analyst ratings from Yahoo Finance.

    Use only when hybrid_search returns no data or user asks for real-time info.

    Args:
        query: Financial research question about current market data.

    Returns:
        Research results from Yahoo Finance MCP.
    """
    code = (
        "import asyncio\n"
        "from scripts.yahoo_mcp import finance_research\n"
        f'asyncio.run(finance_research("{query}"))\n'
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)

    if result.returncode != 0 or not result.stdout.strip():
        return f"Yahoo Finance unavailable.\nError: {result.stderr.strip()[:300]}"

    # ✅ OPT: cap live response to 1 200 chars to avoid huge tool messages
    return result.stdout.strip()[:1200]


@tool
def think_tool(reflection: str) -> str:
    """Record a research reflection — use after each search to decide next steps.

    Args:
        reflection: What you found, what's missing, whether to continue searching.

    Returns:
        Confirmation that reflection was recorded.
    """
    return f"Reflection recorded: {reflection}"


@tool
def format_chart_data(data_json: str, title: str, chart_type: str = "bar") -> str:
    """Validate and format financial data so the Streamlit UI can render a chart.

    Args:
        data_json:  JSON array like '[{"Label":"Q1","Value":100},...]'
        title:      Chart heading.
        chart_type: "bar" or "line".

    Returns:
        Pipe-delimited string consumed by app.py, or an error message.
    """
    try:
        # ✅ FIX: json was used but never imported in original file
        json.loads(data_json)
        return f"CHART_DATA|{chart_type}|{title}|{data_json}"
    except Exception as e:
        return f"Error formatting chart data: {e}"
