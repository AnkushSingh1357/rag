import streamlit as st
import uuid
import os
import glob
import re

# Import orchestrator from main.py
from main import orchestrator_agent, SUPPORTED_COMPANIES
from scripts.llm_utils import get_rotating_llm
from scripts.rag_tools import hybrid_search
from langchain_core.messages import HumanMessage, SystemMessage

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

def _fast_rag_answer(prompt: str) -> str:
    snippets = hybrid_search.invoke({
        "query": prompt,
        "k": int(os.getenv("FAST_RAG_K", "3")),
    })
    if "No historical documents found" in snippets:
        return snippets

    llm = _fast_answer_llm()
    response = llm.invoke([
        SystemMessage(content=(
            "Answer the user's finance question using only the retrieved SEC filing snippets. "
            "Be concise. Include numbers only if present in the snippets. Cite sources briefly. "
            "If a chart is requested, provide a compact markdown table of chart-ready values when possible."
        )),
        HumanMessage(content=f"User question:\n{prompt}\n\nRetrieved snippets:\n{snippets}"),
    ])
    return _extract_text(response)

# ==========================================
# 4. SIDEBAR
# ==========================================
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown(f"**Session ID**\n`{st.session_state.thread_id[:18]}…`")

    st.divider()
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
        st.markdown(message["content"])

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
        fast_mode = os.getenv("FAST_UI_MODE", "true").lower() == "true"
        if fast_mode:
            status_box.info("Fast RAG lookup is searching Qdrant and drafting a concise answer...")
        else:
            status_box.info("Orchestrator is planning and delegating to Research sub-agents...")

        try:
            if fast_mode:
                final_reply = _fast_rag_answer(prompt)
            else:
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

            status_box.empty()

            # Render text
            st.markdown(final_reply)

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
