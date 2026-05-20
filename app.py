import streamlit as st
import uuid
import os
import glob
import re
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import orchestrator from main.py
from main import orchestrator_agent, SUPPORTED_COMPANIES, checkpointer
from scripts.config import MAIN_CHECKPOINT_DB
from scripts.llm_utils import get_rotating_llm
from scripts.rag_tools import hybrid_search
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import sqlite3

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Finance Copilot — Apple · Amazon · Meta · Microsoft",
    page_icon="📈",
    layout="wide",
)

# ── Custom dark CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global dark background */
    [data-testid="stAppViewContainer"] { background-color: #0E1117; }
    [data-testid="stSidebar"]          { background-color: #161B22; }

    /* Chat bubbles */
    [data-testid="stChatMessage"] {
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 6px;
    }

    /* Company badge pills */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px 3px;
    }
    .badge-apple     { background:#555;   color:#fff; }
    .badge-amazon    { background:#FF9900; color:#000; }
    .badge-meta      { background:#1877F2; color:#fff; }
    .badge-microsoft { background:#00A4EF; color:#fff; }

    /* Section divider */
    .section-divider {
        border-top: 1px solid #30363D;
        margin: 18px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HEADER
# ==========================================
st.title("📈 Finance Copilot")
st.markdown(
    "Trained on **Apple · Amazon · Meta · Microsoft** filings. "
    "Ask complex financial questions — the Orchestrator plans, delegates to "
    "Research sub-agents, and auto-generates charts."
)

company_badges = "".join([
    '<span class="badge badge-apple">🍎 Apple</span>',
    '<span class="badge badge-amazon">📦 Amazon</span>',
    '<span class="badge badge-meta">📘 Meta</span>',
    '<span class="badge badge-microsoft">🪟 Microsoft</span>',
])
st.markdown(company_badges, unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ==========================================
# 3. SESSION STATE
# ==========================================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "user_id" not in st.session_state:
    st.session_state.user_id = "streamlit_user"

if "fast_rag_k" not in st.session_state:
    st.session_state.fast_rag_k = int(os.getenv("FAST_RAG_K", "2"))

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I'm your Finance Copilot, trained on Apple, Amazon, Meta, "
                "and Microsoft data.\n\n"
                "Try asking:\n"
                "- *Compare revenue growth of Apple and Microsoft from 2021–2023*\n"
                "- *What were Meta's net income trends over the last 4 quarters?*\n"
                "- *Show Amazon's AWS vs retail segment breakdown as a pie chart*"
            ),
            "charts": [],
        }
    ]

# ── Helper: load recent threads from SQLite ─────────────────────────────────
def load_recent_threads() -> list[str]:
    if not os.path.exists(MAIN_CHECKPOINT_DB):
        return []
    try:
        conn = sqlite3.connect(MAIN_CHECKPOINT_DB)
        cursor = conn.cursor()
        cursor.execute("SELECT thread_id FROM checkpoints GROUP BY thread_id ORDER BY MAX(rowid) DESC LIMIT 15;")
        threads = [row[0] for row in cursor.fetchall()]
        conn.close()
        return threads
    except Exception:
        return []

# ── Helper: load full chat history from LangGraph checkpointer ──────────────
def load_thread_history(thread_id: str) -> list[dict] | None:
    config = {"configurable": {"thread_id": thread_id}}
    state = checkpointer.get(config)
    if not state:
        return None
        
    msgs = []
    raw_msgs = state["channel_values"].get("messages", [])
    chart_paths = state["channel_values"].get("chart_paths", [])
    
    for m in raw_msgs:
        if isinstance(m, HumanMessage):
             msgs.append({"role": "user", "content": str(m.content), "charts": []})
        elif isinstance(m, AIMessage):
             content = _extract_text(m) if hasattr(m, "content") else str(m.content)
             if content and content.strip():
                 msgs.append({"role": "assistant", "content": content, "charts": []})
                 
    if msgs and chart_paths and msgs[-1]["role"] == "assistant":
         msgs[-1]["charts"] = chart_paths
         
    return msgs

# ── Helper: collect chart PNGs for a session ────────────────────────────────
def _session_chart_dir() -> str:
    return os.path.join(
        "agent_files",
        st.session_state.user_id,
        st.session_state.thread_id,
        "charts",
    )

def _new_charts_since(known_paths: list[str]) -> list[str]:
    """Return PNG paths generated after the last message."""
    chart_dir = _session_chart_dir()
    if not os.path.isdir(chart_dir):
        return []
    all_charts = sorted(glob.glob(os.path.join(chart_dir, "*.png")))
    return [p for p in all_charts if p not in known_paths]

# ── Helper: extract clean text from a LangGraph response message ─────────────
def _extract_text(last_message) -> str:
    content = last_message.content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts)
    return str(content)

# ── Helper: inject report.md content if agent references it ─────────────────
def _maybe_append_report(text: str) -> str:
    if "report.md" not in text:
        return text
    report_path = os.path.join(
        "agent_files",
        st.session_state.user_id,
        st.session_state.thread_id,
        "report.md",
    )
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report_content = f.read()
        text = f"{text}\n\n---\n\n{report_content}"
    return text

@st.cache_resource
def _fast_answer_llm():
    return get_rotating_llm(
        model_name=os.getenv("GROQ_FAST_MODEL", os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant")),
        temperature=0,
        max_tokens=650,
    )

@st.cache_resource
def _intent_llm():
    return get_rotating_llm(
        model_name=os.getenv("GROQ_FAST_MODEL", os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant")),
        temperature=0,
        max_tokens=160,
    )

def _rule_intent(prompt: str) -> dict:
    prompt_lower = prompt.lower()
    companies = _find_requested_companies(prompt)
    years = _find_requested_years(prompt)
    quarters = _find_requested_quarters(prompt)
    chart_type = _chart_type(prompt) if _wants_chart(prompt) else None

    metric = None
    if "net income" in prompt_lower or "profit" in prompt_lower:
        metric = "net_income"
    elif "revenue" in prompt_lower or "sales" in prompt_lower:
        metric = "revenue"

    breakdown = None
    if any(word in prompt_lower for word in ("product", "category", "iphone", "ipad", "mac", "wearables", "services")):
        breakdown = "product_category"
    elif "segment" in prompt_lower or "aws" in prompt_lower:
        breakdown = "segment"

    is_live = any(word in prompt_lower for word in ("current", "today", "live", "price", "now"))
    is_direct = any(phrase in prompt_lower for phrase in ("best financial year", "best year", "highest year"))

    intent = "rag_lookup"
    data_source = "qdrant"
    confidence = 0.65
    if is_live:
        intent = "live_market"
        data_source = "yahoo"
        confidence = 0.95
    elif is_direct:
        intent = "direct_financial"
        data_source = "yahoo"
        confidence = 0.95
    elif chart_type:
        intent = "chart"
        confidence = 0.85 if companies and (metric or breakdown) else 0.55
    elif companies and metric:
        confidence = 0.8

    return {
        "intent": intent,
        "data_source": data_source,
        "companies": companies,
        "years": years,
        "quarters": quarters,
        "metric": metric,
        "chart_type": chart_type,
        "breakdown": breakdown,
        "confidence": confidence,
    }

def _classify_intent(prompt: str) -> dict:
    rule_intent = _rule_intent(prompt)
    if (
        os.getenv("HYBRID_INTENT_CLASSIFIER", "false").lower() != "true"
        or rule_intent["confidence"] >= float(os.getenv("INTENT_RULE_CONFIDENCE", "0.75"))
    ):
        return rule_intent

    try:
        response = _intent_llm().invoke([
            SystemMessage(content=(
                "Classify the finance app request. Return only compact JSON with keys: "
                "intent, data_source, companies, years, quarters, metric, chart_type, breakdown, confidence. "
                "Allowed intent: live_market, direct_financial, chart, rag_lookup. "
                "Allowed data_source: yahoo, qdrant, either. Use null for unknown."
            )),
            HumanMessage(content=prompt),
        ])
        parsed = json.loads(_extract_text(response))
        if isinstance(parsed, dict):
            return {**rule_intent, **parsed}
    except Exception:
        return rule_intent

    return rule_intent

def _find_requested_companies(prompt: str) -> list[str]:
    prompt_lower = prompt.lower()
    companies = []
    aliases = {
        "Apple": ("apple", "aapl"),
        "Amazon": ("amazon", "amzn"),
        "Meta": ("meta", "facebook"),
        "Microsoft": ("microsoft", "msft"),
    }
    for company, names in aliases.items():
        if any(re.search(rf"\b{re.escape(name)}\b", prompt_lower) for name in names):
            companies.append(company)
    return companies

def _find_requested_years(prompt: str) -> list[str]:
    years = sorted(set(re.findall(r"\b(20\d{2})\b", prompt)))
    range_match = re.search(r"\b(20\d{2})\s*[-–]\s*(20\d{2})\b", prompt)
    if range_match:
        start, end = map(int, range_match.groups())
        if 0 <= end - start <= 6:
            years = [str(year) for year in range(start, end + 1)]
    return years

def _find_requested_quarters(prompt: str) -> list[str]:
    quarters = sorted(set(re.findall(r"\bq([1-4])\b", prompt.lower())))
    return [f"Q{quarter}" for quarter in quarters]

def _build_fast_queries(prompt: str, intent: dict | None = None) -> list[str]:
    intent = intent or _rule_intent(prompt)
    companies = intent.get("companies") or _find_requested_companies(prompt)
    years = intent.get("years") or _find_requested_years(prompt)
    quarters = intent.get("quarters") or _find_requested_quarters(prompt)
    prompt_lower = prompt.lower()
    metric_key = intent.get("metric") or ("net_income" if "income" in prompt_lower or "profit" in prompt_lower else "revenue")
    metric = "net income" if metric_key == "net_income" else "revenue"

    # Always include the exact raw prompt for semantic search!
    queries = [prompt]
    if intent.get("breakdown") == "product_category" and companies and years:
        for company in companies[:2]:
            for year in years[:3]:
                queries.append(f"{company} {year} net sales by category iPhone Mac iPad Wearables Services 10-K")
    elif intent.get("breakdown") == "segment" and companies and years:
        for company in companies[:2]:
            for year in years[:3]:
                queries.append(f"{company} {year} revenue by segment 10-K")
    elif companies and quarters and years:
        year = years[-1]
        for company in companies[:2]:
            for quarter in quarters[:4]:
                queries.append(f"{company} {metric} {quarter} {year}")
    elif companies and years:
        for company in companies[:2]:
            queries.append(f"{company} {metric} {' '.join(years[:4])}")
    elif companies:
        for company in companies[:2]:
            queries.append(f"{company} {metric} {prompt}")

    return queries[:8] or [prompt]

def _yahoo_financial_table(prompt: str, intent: dict | None = None) -> str | None:
    intent = intent or _rule_intent(prompt)
    companies = intent.get("companies") or _find_requested_companies(prompt)
    years = intent.get("years") or _find_requested_years(prompt)
    prompt_lower = prompt.lower()
    metric_key = intent.get("metric")
    metric_name = "Net Income" if metric_key == "net_income" or "net income" in prompt_lower or "profit" in prompt_lower else "Revenue"
    row_name = "Net Income" if metric_name == "Net Income" else "Total Revenue"
    tickers = {
        "Apple": "AAPL",
        "Amazon": "AMZN",
        "Meta": "META",
        "Microsoft": "MSFT",
    }

    broad_financial_lookup = any(
        phrase in prompt_lower
        for phrase in ("best financial year", "best year", "highest year", "annual financial")
    )
    if not companies or metric_name == "Revenue" and "revenue" not in prompt_lower and not broad_financial_lookup:
        return None

    try:
        import yfinance as yf
    except ImportError:
        return None

    rows = []
    for company in companies:
        ticker = tickers.get(company)
        if not ticker:
            continue

        financials = yf.Ticker(ticker).financials
        if financials.empty or row_name not in financials.index:
            continue

        series = financials.loc[row_name]
        for col, raw_value in series.items():
            year = str(getattr(col, "year", col))[:4]
            if years and year not in years:
                continue
            if raw_value is None:
                continue
            if raw_value != raw_value:
                continue
            value_millions = float(raw_value) / 1_000_000
            rows.append((f"{company} {year}", value_millions))

    if not rows:
        return None

    rows.sort(key=lambda item: item[0])
    table = ["| Period | Value |", "|---|---:|"]
    for period, value in rows:
        table.append(f"| {period} | {value:,.0f} |")

    return (
        f"Yahoo Finance annual {metric_name.lower()} data, values in millions USD.\n\n"
        + "\n".join(table)
        + "\n\nSource: Yahoo Finance via yfinance."
    )

def _yahoo_quarterly_financial_table(prompt: str, intent: dict | None = None) -> str | None:
    intent = intent or _rule_intent(prompt)
    companies = intent.get("companies") or _find_requested_companies(prompt)
    if len(companies) != 1:
        return None

    prompt_lower = prompt.lower()
    quarters = intent.get("quarters") or _find_requested_quarters(prompt)
    years = intent.get("years") or _find_requested_years(prompt)
    wants_quarterly = "quarterly" in prompt_lower or "quaterly" in prompt_lower or bool(quarters)
    if not wants_quarterly:
        return None

    metric_name = "Net Income" if intent.get("metric") == "net_income" or "income" in prompt_lower or "profit" in prompt_lower else "Revenue"
    row_name = "Net Income" if metric_name == "Net Income" else "Total Revenue"
    tickers = {
        "Apple": "AAPL",
        "Amazon": "AMZN",
        "Meta": "META",
        "Microsoft": "MSFT",
    }
    ticker = tickers.get(companies[0])
    if not ticker:
        return None

    try:
        import yfinance as yf
        financials = yf.Ticker(ticker).quarterly_financials
    except Exception:
        return None

    if financials.empty or row_name not in financials.index:
        return None

    rows = []
    series = financials.loc[row_name]
    for col, raw_value in series.items():
        if raw_value is None or raw_value != raw_value:
            continue
        year = str(getattr(col, "year", col))[:4]
        quarter = f"Q{((getattr(col, 'month', 1) - 1) // 3) + 1}"
        if years and year not in years:
            continue
        if quarters and quarter not in quarters:
            continue
        rows.append((f"{quarter} {year}", float(raw_value) / 1_000_000))

    if not rows:
        return None

    # Keep a compact recent trend if the user did not specify exact periods.
    rows = rows[:4] if not years and not quarters else rows
    rows.sort(key=lambda item: (item[0].split()[1], item[0].split()[0]))

    table = ["| Period | Value |", "|---|---:|"]
    for period, value in rows:
        table.append(f"| {period} | {value:,.0f} |")

    return (
        f"Yahoo Finance quarterly {metric_name.lower()} data for {companies[0]}, values in millions USD.\n\n"
        + "\n".join(table)
        + "\n\nSource: Yahoo Finance via yfinance."
    )

def _amazon_aws_vs_retail_answer(prompt: str, intent: dict | None = None) -> str | None:
    prompt_lower = prompt.lower()
    if "amazon" not in prompt_lower or "aws" not in prompt_lower:
        return None
    if not any(word in prompt_lower for word in ("retail", "stores", "segment", "pie")):
        return None

    years = _find_requested_years(prompt)
    year = years[-1] if years else "2023"
    k_val = max(2, min(int(st.session_state.get("fast_rag_k", os.getenv("FAST_RAG_K", "2"))), 20))
    retrieve_k_val = int(st.session_state.get("fast_rag_retrieve_k", k_val * 5))
    result = hybrid_search.invoke({
        "query": f"Amazon {year} net sales Online stores Physical stores AWS 10-K",
        "k": k_val,
        "retrieve_k": retrieve_k_val
    })

    aws_match = re.search(r"AWS\s+\$?\s*[\d,]+\s+\$?\s*[\d,]+\s+\$?\s*([\d,]+)", result, re.IGNORECASE)
    online_match = re.search(r"Online stores.*?\$?\s*[\d,]+\s+\$?\s*[\d,]+\s+\$?\s*([\d,]+)", result, re.IGNORECASE | re.DOTALL)
    physical_match = re.search(r"Physical stores.*?\$?\s*[\d,]+\s+\$?\s*[\d,]+\s+\$?\s*([\d,]+)", result, re.IGNORECASE | re.DOTALL)

    if not aws_match or not online_match:
        return None

    aws = float(aws_match.group(1).replace(",", ""))
    online = float(online_match.group(1).replace(",", ""))
    physical = float(physical_match.group(1).replace(",", "")) if physical_match else 0.0
    retail = online + physical

    return (
        f"Amazon AWS vs retail revenue for {year}, values in millions USD. "
        f"Retail is Online stores plus Physical stores.\n\n"
        "| Period | Value |\n"
        "|---|---:|\n"
        f"| AWS | {aws:,.0f} |\n"
        f"| Retail | {retail:,.0f} |\n\n"
        f"Source: Amazon {year} 10-K via Qdrant."
    )

def _direct_financial_answer(prompt: str, intent: dict | None = None) -> str | None:
    intent = intent or _rule_intent(prompt)
    prompt_lower = prompt.lower()
    if intent.get("intent") != "direct_financial" and not any(phrase in prompt_lower for phrase in ("best financial year", "best year", "highest year")):
        return None

    table_answer = _yahoo_financial_table(prompt, intent)
    if not table_answer:
        return None

    labels, values = _parse_markdown_table(table_answer)
    if not labels or not values:
        return table_answer

    best_index = max(range(len(values)), key=lambda idx: values[idx])
    best_period = labels[best_index]
    best_value = values[best_index]
    metric = "net income" if "net income" in prompt_lower or "profit" in prompt_lower else "revenue"

    return (
        f"Assuming \"best\" means highest annual {metric}, {best_period} was the best financial year "
        f"with {best_value:,.0f} million USD ({best_value / 1000:,.3f} billion USD).\n\n"
        f"{table_answer}"
    )

def _live_market_answer(prompt: str, intent: dict | None = None) -> str | None:
    intent = intent or _rule_intent(prompt)
    if intent.get("intent") != "live_market":
        return None

    prompt_lower = prompt.lower()
    ticker = None
    label = None
    if "nifty" in prompt_lower or "nsei" in prompt_lower:
        ticker, label = "^NSEI", "NIFTY 50"
    elif "sensex" in prompt_lower:
        ticker, label = "^BSESN", "SENSEX"
    else:
        ticker_map = {
            "apple": ("AAPL", "Apple"),
            "aapl": ("AAPL", "Apple"),
            "amazon": ("AMZN", "Amazon"),
            "amzn": ("AMZN", "Amazon"),
            "meta": ("META", "Meta"),
            "microsoft": ("MSFT", "Microsoft"),
            "msft": ("MSFT", "Microsoft"),
            "google": ("GOOGL", "Alphabet"),
        }
        for token, mapped in ticker_map.items():
            if re.search(rf"\b{re.escape(token)}\b", prompt_lower):
                ticker, label = mapped
                break

    if not ticker:
        return None

    try:
        import yfinance as yf
        history = yf.Ticker(ticker).history(period="5d", interval="1d")
        if history.empty:
            return None
        last = history.iloc[-1]
        price = float(last["Close"])
        date = str(history.index[-1].date())
        return f"{label} ({ticker}) last close was {price:,.2f} as of {date}.\n\nSource: Yahoo Finance via yfinance."
    except Exception:
        return None

def _fast_rag_answer(prompt: str) -> str:
    intent = _classify_intent(prompt)

    live_answer = _live_market_answer(prompt, intent)
    if live_answer:
        return live_answer

    direct_answer = _direct_financial_answer(prompt, intent)
    if direct_answer:
        return direct_answer

    amazon_segment_answer = _amazon_aws_vs_retail_answer(prompt, intent)
    if amazon_segment_answer:
        return amazon_segment_answer

    quarterly_answer = _yahoo_quarterly_financial_table(prompt, intent)
    if quarterly_answer and (intent.get("chart_type") or "quarterly" in prompt.lower()):
        return quarterly_answer

    yahoo_answer = _yahoo_financial_table(prompt, intent)
    if yahoo_answer and "microsoft" in prompt.lower():
        return yahoo_answer

    snippet_parts = []
    per_query_k = max(1, min(int(st.session_state.get("fast_rag_k", os.getenv("FAST_RAG_K", "2"))), 20))
    retrieve_k_val = int(st.session_state.get("fast_rag_retrieve_k", per_query_k * 5))
    for query in _build_fast_queries(prompt, intent):
        result = hybrid_search.invoke({"query": query, "k": per_query_k, "retrieve_k": retrieve_k_val})
        if "No historical documents found" not in result:
            snippet_parts.append(f"Query: {query}\n{result}")

    snippets = "\n\n---\n\n".join(snippet_parts)
    if "No historical documents found" in snippets:
        return snippets
    if not snippets:
        return "No historical documents found for this query."

    llm = _fast_answer_llm()
    response = llm.invoke([
        SystemMessage(content=(
            "Answer the user's finance question using only the retrieved SEC filing snippets. "
            "Be concise. Include numbers only if present in the snippets; never fill missing periods from memory. "
            "You MUST append a bold **Sources:** section at the very end of your response, listing the exact source filenames provided in the snippets. "
            "If a chart is requested, output a pipe-delimited markdown table with exactly two columns "
            "before any notes: Period and Value. Use one row per line, including the separator row. "
            "Example:\n| Period | Value |\n|---|---:|\n| Q1 2024 | 123 |\nDo not use tab-separated tables."
        )),
        HumanMessage(content=f"User question:\n{prompt}\n\nRetrieved snippets:\n{snippets}"),
    ])
    base_response = _extract_text(response)
    
    # Safely format the raw chunks into an HTML expander
    safe_snippets = snippets.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
    expander_html = (
        f"\n\n<details><summary>🔍 <b>View Sources</b></summary>"
        f"<p style='font-size: 0.85em; color: gray; margin-top: 10px;'>{safe_snippets}</p></details>"
    )
    return base_response + expander_html

def _wants_chart(prompt: str) -> bool:
    prompt_lower = prompt.lower()
    return any(word in prompt_lower for word in ("chart", "graph", "plot", "line", "bar", "pie"))

def _chart_type(prompt: str) -> str:
    prompt_lower = prompt.lower()
    if "pie" in prompt_lower:
        return "pie"
    if "bar" in prompt_lower:
        return "bar"
    return "line"

def _number_from_text(value: str) -> float | None:
    cleaned = re.sub(r"[^0-9.\-]", "", value)
    if not cleaned or cleaned in {"-", ".", "-."}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None

def _parse_markdown_table(text: str) -> tuple[list[str], list[float]]:
    lines = [line.strip() for line in text.splitlines() if line.strip().startswith("|")]
    for idx in range(len(lines) - 2):
        header = [cell.strip().lower() for cell in lines[idx].strip("|").split("|")]
        separator = lines[idx + 1]
        if not all("-" in cell for cell in separator.strip("|").split("|")):
            continue
        if len(header) < 2:
            continue

        labels = []
        values = []
        for row in lines[idx + 2:]:
            cells = [cell.strip() for cell in row.strip("|").split("|")]
            if len(cells) < 2:
                break
            value = _number_from_text(cells[1])
            if value is None:
                continue
            if "billion" in cells[1].lower():
                value *= 1000
            labels.append(cells[0])
            values.append(value)

        if labels and values:
            return labels, values

    tabular_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("|"):
            continue
        cells = re.split(r"\t+|\s{2,}", stripped)
        if len(cells) >= 2:
            tabular_lines.append(cells)

    labels = []
    values = []
    for cells in tabular_lines:
        first = cells[0].strip()
        second = cells[1].strip()
        if first.lower() in {"quarter", "period", "year"}:
            continue
        value = _number_from_text(second)
        if value is None:
            continue
        if "billion" in second.lower():
            value *= 1000
        labels.append(first)
        values.append(value)

    if labels and values:
        return labels, values

    return [], []

def _generate_fast_chart(prompt: str, answer: str) -> str | None:
    if not _wants_chart(prompt):
        return None

    labels, values = _parse_markdown_table(answer)
    if not labels or not values:
        return None

    chart_type = _chart_type(prompt)
    chart_dir = _session_chart_dir()
    os.makedirs(chart_dir, exist_ok=True)
    chart_path = os.path.join(chart_dir, f"{uuid.uuid4().hex[:10]}_{chart_type}.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#1C1E26")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    if chart_type == "pie":
        ax.pie(values, labels=labels, autopct="%1.1f%%", textprops={"color": "white"})
        ax.set_title(prompt, color="white", fontsize=13, fontweight="bold")
    else:
        x = range(len(labels))
        if chart_type == "bar":
            ax.bar(x, values, color="#1877F2", alpha=0.9)
        else:
            ax.plot(x, values, marker="o", color="#1877F2", linewidth=2.5)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=25, ha="right", color="white")
        ax.set_ylabel("Value", color="white")
        ax.set_title(prompt, color="white", fontsize=13, fontweight="bold")
        ax.grid(color="#333333", linestyle="--", linewidth=0.5, axis="y")

    plt.tight_layout()
    plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return chart_path

# ==========================================
# 4. SIDEBAR
# ==========================================
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown(f"**Session ID**\n`{st.session_state.thread_id[:18]}…`")

    st.session_state.fast_mode = st.toggle(
        "⚡ Fast Mode (Quick RAG)",
        value=os.getenv("FAST_UI_MODE", "true").lower() == "true",
        help="Turn off to enable Deep Research. Deep Research saves conversations to History."
    )
    
    st.divider()
    st.header("💬 Chat History")
    recent_threads = load_recent_threads()
    
    if recent_threads:
        options = ["Current Session"] + recent_threads
        selected = st.selectbox("Past Conversations", options, label_visibility="collapsed")
        if selected != "Current Session" and selected != st.session_state.thread_id:
            st.session_state.thread_id = selected
            history = load_thread_history(selected)
            if history:
                st.session_state.messages = history
            st.rerun()
    else:
        st.caption("No history found. Turn off Fast Mode to save conversations.")

    st.divider()
    current_k = int(st.session_state.fast_rag_k)
    new_k = st.slider(
        "Final K (Chunks sent to LLM)",
        min_value=1,
        max_value=20,
        value=current_k,
        step=1,
        help="Number of chunks fed to the LLM. Lower = better quality/faster.",
    )
    
    # If user changed Final K just now, auto-update Top K to 5:1 ratio
    if new_k != current_k:
        st.session_state.fast_rag_k = new_k
        st.session_state.fast_rag_retrieve_k = min(new_k * 5, 50)
        st.rerun()
        
    current_retrieve_k = int(st.session_state.get("fast_rag_retrieve_k", new_k * 5))
    
    st.session_state.fast_rag_retrieve_k = st.slider(
        "Top K (Pulled from Database)",
        min_value=1,
        max_value=50,
        value=current_retrieve_k,
        step=1,
        help="Number of raw chunks to pull before re-ranking. Auto-scales to 5:1 ratio.",
    )
    
    st.caption(f"DB pulls {st.session_state.fast_rag_retrieve_k} → Re-ranker keeps best {st.session_state.fast_rag_k}")

    st.markdown("**Covered Companies**")
    for c in SUPPORTED_COMPANIES:
        st.markdown(f"- {c}")

    st.divider()
    if st.button("🗑️ Clear Chat & Reset Memory"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Memory cleared! What would you like to research next?",
                "charts": [],
            }
        ]
        st.rerun()

    st.divider()
    st.caption("Powered by Groq · LangGraph · LlamaIndex · Qdrant")

# ==========================================
# 5. RENDER CHAT HISTORY
# ==========================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

        # Render any charts that were saved with this message
        charts = message.get("charts", [])
        if charts:
            cols = st.columns(min(len(charts), 2))
            for idx, chart_path in enumerate(charts):
                if os.path.exists(chart_path):
                    with cols[idx % 2]:
                        st.image(chart_path, use_container_width=True)

# ==========================================
# 6. CHAT INPUT
# ==========================================
EXAMPLE_QUERIES = [
    "Compare Apple vs Microsoft revenue 2021-2023 (bar chart)",
    "Meta's quarterly net income — line chart",
    "Amazon AWS vs retail revenue pie chart",
    "Show Microsoft cloud vs on-prem breakdown",
]

# Quick-start buttons (only before first user message)
if len(st.session_state.messages) == 1:
    st.markdown("**Quick start:**")
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUERIES):
        if cols[i % 2].button(q, key=f"qs_{i}"):
            st.session_state["pending_query"] = q
            st.rerun()

# Handle pending quick-start query
prompt = st.chat_input(
    "E.g., Compare Amazon and Microsoft revenues for 2022–2023 with a bar chart"
)
if not prompt and "pending_query" in st.session_state:
    prompt = st.session_state.pop("pending_query")

# ==========================================
# 7. AGENT INVOCATION
# ==========================================
if prompt:
    # Track charts already on disk before this call
    chart_dir = _session_chart_dir()
    existing_charts = sorted(glob.glob(os.path.join(chart_dir, "*.png"))) if os.path.isdir(chart_dir) else []

    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt, "charts": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run the agent
    with st.chat_message("assistant"):
        status_box = st.empty()
        fast_mode = st.session_state.get("fast_mode", True)
        if fast_mode:
            status_box.info("Fast RAG lookup is searching Qdrant and drafting a concise answer...")
        else:
            status_box.info("Orchestrator is planning and delegating to Research sub-agents...")

        try:
            if fast_mode:
                final_reply = _fast_rag_answer(prompt)
                fast_chart = _generate_fast_chart(prompt, final_reply)
            else:
                fast_chart = None
                initial_state = {
                    "messages": [HumanMessage(content=prompt)],
                    "user_id": st.session_state.user_id,
                    "thread_id": st.session_state.thread_id,
                    "chart_paths": existing_charts,
                }
                config = {
                    "configurable": {"thread_id": st.session_state.thread_id},
                    "recursion_limit": int(os.getenv("AGENT_RECURSION_LIMIT", "10")),
                }

                response = orchestrator_agent.invoke(initial_state, config=config)
                last_message = response["messages"][-1]
                final_reply = _extract_text(last_message)
                final_reply = _maybe_append_report(final_reply)

            # ── Collect newly created charts ────────────────────────────────
            new_charts = _new_charts_since(existing_charts)
            if fast_chart and fast_chart not in new_charts:
                new_charts.append(fast_chart)

            status_box.empty()

            # Render text
            st.markdown(final_reply, unsafe_allow_html=True)

            # Render charts in a responsive 2-column grid
            if new_charts:
                st.markdown("---")
                st.markdown("### 📊 Generated Charts")
                chart_cols = st.columns(min(len(new_charts), 2))
                for idx, chart_path in enumerate(new_charts):
                    with chart_cols[idx % 2]:
                        st.image(chart_path, use_container_width=True)

            # ── Save to session history ─────────────────────────────────────
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_reply,
                "charts": new_charts,
            })

        except Exception as e:
            status_box.empty()
            st.error(f"❌ An error occurred: {e}")
            st.exception(e)
