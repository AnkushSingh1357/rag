import os
import sys
import time
import warnings
import json
import groq
from typing import Annotated
from dotenv import load_dotenv

load_dotenv(override=True)
warnings.filterwarnings('ignore')

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.prebuilt import create_react_agent, InjectedState
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from langchain_groq import ChatGroq

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from scripts.rag_tools import hybrid_search, live_finance_researcher
from scripts.file_tools import (
    DeepAgentState,
    ls,
    read_file,
    write_file,
    cleanup_files,
    generate_hash,
    _disk_path
)
from scripts.prompts import (
    ORCHESTRATOR_PROMPT,
    RESEARCHER_PROMPT,
    EDITOR_PROMPT,
)
from scripts.agent_utils import stream_agent_response
from scripts.config import (
    AGENT_QUERY,
    AGENT_THREAD_ID,
    AGENT_USER_ID,
    MAIN_CHECKPOINT_DB,
)

# ==========================================
# SUPPORTED COMPANIES
# ==========================================
SUPPORTED_COMPANIES = ["Apple", "Amazon", "Meta", "Microsoft"]

from scripts.llm_utils import get_rotating_llm

llm = get_rotating_llm(
    model_name=os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile"),
    temperature=0,
    max_tokens=900,
)

import psycopg

DB_URI = os.getenv("POSTGRES_DB_URI", "postgresql://rag_user:rag_password@localhost:5433/rag_db")

# 1. Create tables using an autocommit connection
with psycopg.connect(DB_URI, autocommit=True) as _conn:
    PostgresSaver(_conn).setup()
    with _conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag_evaluations (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                query TEXT,
                coherence_score REAL,
                top_chunk_score REAL,
                source_count INTEGER,
                response_length INTEGER,
                info_density REAL,
                context_spread REAL,
                readability REAL
            )
        """)
        # In case the table already exists from a previous run, add the new columns
        try:
            cur.execute("ALTER TABLE rag_evaluations ADD COLUMN IF NOT EXISTS info_density REAL")
            cur.execute("ALTER TABLE rag_evaluations ADD COLUMN IF NOT EXISTS context_spread REAL")
            cur.execute("ALTER TABLE rag_evaluations ADD COLUMN IF NOT EXISTS readability REAL")
        except Exception:
            pass

# 2. Initialize the production connection pool
pool = ConnectionPool(conninfo=DB_URI, max_size=20, kwargs={"autocommit": True})
checkpointer = PostgresSaver(pool)

# ==========================================
# 2. CHART COLORS
# ==========================================
CHART_COLORS = {
    "Apple":     "#A8A8A8",
    "Amazon":    "#FF9900",
    "Meta":      "#1877F2",
    "Microsoft": "#00A4EF",
}

# Used for pie slices and any label not matching a company name
PIE_PALETTE = [
    "#FF9900", "#00A4EF", "#1877F2", "#34A853",
    "#EA4335", "#FBBC04", "#9C27B0", "#00BCD4",
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
]


def _get_chart_dir(state: DeepAgentState) -> str:
    base = _disk_path(state, "charts")
    os.makedirs(base, exist_ok=True)
    return base


# ==========================================
# 3. GENERATE CHART TOOL
# ==========================================
@tool
def generate_chart(
    chart_type: str,
    title: str,
    x_labels: list[str],
    datasets: list[dict],
    x_axis_label: str,
    y_axis_label: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """
    Render a bar, line, or pie chart and save it as a PNG.

    Args:
        chart_type:   "bar" | "line" | "pie"
        title:        Chart title string.
        x_labels:     For bar/line: time periods. For pie: slice/segment names.
        datasets:     List of dicts with keys "label" (str) and "values" (list of numbers).
                      PIE RULE: pass exactly ONE dataset with ALL slice values.
                      BAR/LINE RULE: one dataset per company/series.
        x_axis_label: X-axis label (ignored for pie).
        y_axis_label: Y-axis label (ignored for pie).

    IMPORTANT for pie charts:
        CORRECT: x_labels=["AWS","Retail"], datasets=[{"label":"Revenue","values":[90757,404885]}]
        WRONG:   datasets=[{"label":"AWS","values":[90757]},{"label":"Retail","values":[404885]}]
        Values must be actual dollar amounts in millions, NOT fractions (not 0.3, not 30).
    """
    chart_dir = _get_chart_dir(state)
    file_hash = generate_hash(title)
    filename  = f"{file_hash}_{chart_type}.png"
    full_path = os.path.join(chart_dir, filename)

    n_series = len(datasets)

    # ─────────────────────────────────────────────
    # PIE CHART
    # ─────────────────────────────────────────────
    if chart_type == "pie":
        # Auto-merge: agent may pass one dataset per slice instead of one dataset total
        # e.g. [{"label":"AWS","values":[90757]}, {"label":"Retail","values":[404885]}]
        if n_series > 1 and all(len(ds.get("values", [])) == 1 for ds in datasets):
            pie_labels = [ds["label"] for ds in datasets]
            pie_values = [ds["values"][0] for ds in datasets]
        else:
            pie_labels = x_labels
            pie_values = datasets[0]["values"] if datasets else []

        # Guard: if all values are fractions (<=1), agent passed percentages — scale up
        if pie_values and all(isinstance(v, (int, float)) and 0 < v <= 1 for v in pie_values):
            pie_values = [v * 100 for v in pie_values]

        # Guard: filter out zero/negative values
        filtered = [(lbl, val) for lbl, val in zip(pie_labels, pie_values) if val > 0]
        if not filtered:
            msg = "[CHART ERROR] All pie values are zero or invalid. Cannot render pie chart."
            return Command(update={"messages": [ToolMessage(msg, tool_call_id=tool_call_id)]})
        pie_labels, pie_values = zip(*filtered)
        pie_labels, pie_values = list(pie_labels), list(pie_values)

        palette = [
            CHART_COLORS.get(lbl, PIE_PALETTE[i % len(PIE_PALETTE)])
            for i, lbl in enumerate(pie_labels)
        ]

        fig, ax = plt.subplots(figsize=(9, 6))
        fig.patch.set_facecolor("#0E1117")
        ax.set_facecolor("#0E1117")

        wedges, texts, autotexts = ax.pie(
            pie_values,
            labels=pie_labels,
            autopct="%1.1f%%",
            colors=palette,
            explode=[0.03] * len(pie_labels),
            textprops={"color": "white", "fontsize": 11},
            startangle=140,
            pctdistance=0.80,
        )
        for at in autotexts:
            at.set_color("white")
            at.set_fontsize(10)
            at.set_fontweight("bold")
        for t in texts:
            t.set_color("white")
            t.set_fontsize(11)

        ax.set_title(title, fontsize=14, fontweight="bold", color="white", pad=16)
        plt.tight_layout()
        plt.savefig(full_path, dpi=150, bbox_inches="tight", facecolor="#0E1117")
        plt.close(fig)

    # ─────────────────────────────────────────────
    # BAR CHART
    # ─────────────────────────────────────────────
    elif chart_type == "bar":
        fig, ax = plt.subplots(figsize=(11, 5))
        fig.patch.set_facecolor("#0E1117")
        ax.set_facecolor("#1C1E26")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

        x = np.arange(len(x_labels))
        width = 0.7 / max(n_series, 1)
        offsets = np.linspace(-(n_series - 1) / 2, (n_series - 1) / 2, n_series) * width

        for i, (ds, offset) in enumerate(zip(datasets, offsets)):
            color = CHART_COLORS.get(ds["label"], PIE_PALETTE[i % len(PIE_PALETTE)])
            bars = ax.bar(x + offset, ds["values"], width, label=ds["label"], color=color, alpha=0.88)
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.annotate(
                        f"{h:,.0f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7, color="white"
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=30, ha="right", color="white")
        ax.set_xlabel(x_axis_label, color="white")
        ax.set_ylabel(y_axis_label, color="white")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda val, _: f"{val:,.0f}"))
        ax.set_title(title, fontsize=13, fontweight="bold", color="white", pad=12)
        ax.legend(facecolor="#1C1E26", labelcolor="white", framealpha=0.7)
        plt.tight_layout()
        plt.savefig(full_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    # ─────────────────────────────────────────────
    # LINE CHART
    # ─────────────────────────────────────────────
    elif chart_type == "line":
        fig, ax = plt.subplots(figsize=(11, 5))
        fig.patch.set_facecolor("#0E1117")
        ax.set_facecolor("#1C1E26")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

        x = np.arange(len(x_labels))
        for i, ds in enumerate(datasets):
            color = CHART_COLORS.get(ds["label"], PIE_PALETTE[i % len(PIE_PALETTE)])
            ax.plot(x, ds["values"], marker="o", label=ds["label"], color=color, linewidth=2.5)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=30, ha="right", color="white")
        ax.set_xlabel(x_axis_label, color="white")
        ax.set_ylabel(y_axis_label, color="white")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda val, _: f"{val:,.0f}"))
        ax.grid(color="#333333", linestyle="--", linewidth=0.5)
        ax.set_title(title, fontsize=13, fontweight="bold", color="white", pad=12)
        ax.legend(facecolor="#1C1E26", labelcolor="white", framealpha=0.7)
        plt.tight_layout()
        plt.savefig(full_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    else:
        msg = f"[CHART ERROR] Unknown chart_type '{chart_type}'. Use 'bar', 'line', or 'pie'."
        return Command(update={"messages": [ToolMessage(msg, tool_call_id=tool_call_id)]})

    relative_path = os.path.join("charts", filename)
    msg = f"[CHART SAVED] {relative_path} — Title: {title}"
    return Command(
        update={
            "messages": [ToolMessage(msg, tool_call_id=tool_call_id)],
            "chart_paths": state.get("chart_paths", []) + [full_path],
        }
    )


# ==========================================
# 4. CREATE WORKER AGENTS
# ==========================================
researcher_agent = create_react_agent(
    model=llm,
    tools=[ls, write_file, read_file, hybrid_search, live_finance_researcher, generate_chart],
    prompt=RESEARCHER_PROMPT,
    state_schema=DeepAgentState,
)

editor_agent = create_react_agent(
    model=llm,
    tools=[ls, read_file, cleanup_files, generate_chart],
    prompt=EDITOR_PROMPT,
    state_schema=DeepAgentState,
)


# ==========================================
# 5. ORCHESTRATOR TOOLS
# ==========================================
@tool
def write_research_plan(
    thematic_questions: list[str],
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """Write the high-level research plan with major thematic questions."""
    last_msg = state["messages"][-1]
    query_text = (
        last_msg.content
        if isinstance(last_msg.content, str)
        else " ".join(
            b["text"] for b in last_msg.content if isinstance(b, dict) and "text" in b
        )
    )

    content  = "# Research Plan\n\n## User Query\n"
    content += query_text + "\n\n## Thematic Questions\n\n"
    for i, question in enumerate(thematic_questions, 1):
        content += f"{i}. {question}\n"

    path = _disk_path(state, "research_plan.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    msg = f"[RESEARCH PLAN WRITTEN] research_plan.md with {len(thematic_questions)} thematic questions"
    return Command(update={"messages": [ToolMessage(msg, tool_call_id=tool_call_id)]})


@tool
def run_researcher(
    theme_id: int,
    thematic_question: str,
    state: Annotated[DeepAgentState, InjectedState],
    max_retries: int = 0,
):
    """Run a single Research agent for ONE thematic question."""
    file_hash = generate_hash(f"{theme_id}_{thematic_question}")

    instruction = (
        f"[THEME {theme_id}] {thematic_question}\n\n"
        f"Save your research findings to: researcher/{file_hash}_theme.md\n"
        f"Save your sources list to: researcher/{file_hash}_sources.txt\n\n"
        f"Search rules:\n"
        f"1. Use hybrid_search with 2-4 specific queries (include company name, metric, year).\n"
        f"2. If hybrid_search returns nothing, use live_finance_researcher.\n"
        f"3. If you find numeric data, call generate_chart to visualise it.\n"
        f"   PIE CHART: use ONE dataset with all values. x_labels = segment names.\n"
        f"   Values must be in millions USD (actual amounts, not fractions).\n"
        f"   CRITICAL: All values MUST be raw numbers. NEVER use strings in values list.\n"
        f"Supported companies: {', '.join(SUPPORTED_COMPANIES)}."
    )

    # Only pass the initial user query and the instruction. 
    # Do NOT pass the full Orchestrator history to avoid INVALID_CHAT_HISTORY errors from unresolved tool calls.
    initial_user_msg = state["messages"][0] if state["messages"] else HumanMessage(content=thematic_question)
    
    sub_state: DeepAgentState = {
        "messages": [initial_user_msg, HumanMessage(content=instruction)],
        "user_id":  state.get("user_id"),
        "thread_id": state.get("thread_id"),
        "chart_paths": state.get("chart_paths", []),
    }

    for attempt in range(max_retries + 1):
        try:
            result = researcher_agent.invoke(sub_state, config={"recursion_limit": 8})
            new_charts = result.get("chart_paths", [])
            
            # Extract findings if available so Orchestrator can see them
            output_path = _disk_path(state, f"researcher/{file_hash}_theme.md")
            content = "No findings written to file."
            if os.path.exists(output_path):
                with open(output_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    
            return f"✓ Theme {theme_id} completed (hash: {file_hash}) | charts: {new_charts}\n\nFindings:\n{content}"
        except Exception as e:
            err = str(e)
            is_rate_limit = "429" in err or "rate_limit" in err.lower() or "rate limit" in err.lower()
            if is_rate_limit and attempt < max_retries:
                wait = 15 * (attempt + 1)   # 15s, 30s, 45s
                print(f"[Rate limit] Theme {theme_id} attempt {attempt+1} — retrying in {wait}s…")
                time.sleep(wait)
            else:
                print(f"[Researcher error] Theme {theme_id} attempt {attempt+1}: {e}")

    return f"✗ Theme {theme_id} failed after {max_retries + 1} attempts"


@tool
def run_editor(state: Annotated[DeepAgentState, InjectedState]) -> str:
    """Run the Editor agent to synthesise all research into a final report."""
    sub_state: DeepAgentState = {
        "messages": [HumanMessage(content=(
            "Read research_plan.md and every file inside the researcher/ folder. "
            "Then synthesise all findings into a comprehensive report. "
            "Use markdown tables for all numeric data. "
            "Include inline citations and a References section. "
            "Reference chart filenames from the charts/ folder so the UI can render them."
        ))],
        "user_id":   state.get("user_id"),
        "thread_id": state.get("thread_id"),
        "chart_paths": state.get("chart_paths", []),
    }
    result = editor_agent.invoke(sub_state, config={"recursion_limit": 6})
    
    final_report = result["messages"][-1].content
    if isinstance(final_report, list):
        final_report = " ".join(b["text"] for b in final_report if isinstance(b, dict) and "text" in b)
    
    path = _disk_path(state, "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(final_report)
        
    return f"Editor completed. Final report written to report.md. Contents:\n\n{final_report}"


# ==========================================
# 6. HISTORY TRIMMER
# ==========================================
# Keeps system prompt + last N messages to cap token usage.
# Trimmed messages are still saved in PostgreSQL — they just aren't sent to the LLM.
MAX_HISTORY_MESSAGES = 10  # tune this: lower = fewer tokens, less cross-query memory

def _trim_orchestrator_messages(state: DeepAgentState) -> list:
    """Prepend the system prompt and keep only the last MAX_HISTORY_MESSAGES messages."""
    messages = state["messages"]
    if len(messages) > MAX_HISTORY_MESSAGES:
        # Always keep the very first human message (current query context)
        # and the most recent MAX_HISTORY_MESSAGES messages
        trimmed = messages[-MAX_HISTORY_MESSAGES:]
        # If the first trimmed message isn't HumanMessage, prepend the original query
        if trimmed and not isinstance(trimmed[0], HumanMessage):
            trimmed = [messages[0]] + trimmed
        messages = trimmed
    return [SystemMessage(content=ORCHESTRATOR_PROMPT)] + list(messages)


# ==========================================
# 6. CREATE ORCHESTRATOR AGENT
# ==========================================
orchestrator_agent = create_react_agent(
    model=llm,
    tools=[write_research_plan, run_researcher, run_editor, cleanup_files, generate_chart],
    prompt=_trim_orchestrator_messages,
    state_schema=DeepAgentState,
    checkpointer=checkpointer,
)


# ==========================================
# 7. CLI ENTRY POINT
# ==========================================
if __name__ == "__main__":
    print("Starting Multi-Agent Deep AI Finance Researcher…\n")
    print(f"Supported companies: {', '.join(SUPPORTED_COMPANIES)}\n")
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else AGENT_QUERY
    stream_agent_response(
        orchestrator_agent,
        query,
        thread_id=AGENT_THREAD_ID,
        user_id=AGENT_USER_ID,
    )
