import warnings
import sqlite3
import sys
from typing import Annotated
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
warnings.filterwarnings('ignore')

# LangChain & LangGraph imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langgraph.checkpoint.sqlite import SqliteSaver

# Custom local imports
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
    GEMINI_CHAT_MODEL,
    MAIN_CHECKPOINT_DB,
)

# ==========================================
# 1. INITIALIZE COMPONENTS
# ==========================================
llm = ChatGoogleGenerativeAI(model=GEMINI_CHAT_MODEL)

conn = sqlite3.connect(MAIN_CHECKPOINT_DB, check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# ==========================================
# 2. CREATE WORKER AGENTS
# ==========================================
researcher_agent = create_agent(
    model=llm,
    tools=[ls, write_file, read_file, hybrid_search, live_finance_researcher],
    system_prompt=RESEARCHER_PROMPT,
    state_schema=DeepAgentState,
)

editor_agent = create_agent(
    model=llm,
    tools=[ls, read_file, write_file, cleanup_files],
    system_prompt=EDITOR_PROMPT,
    state_schema=DeepAgentState,
)

# ==========================================
# 3. DEFINE ORCHESTRATOR TOOLS
# ==========================================
@tool
def write_research_plan(
    thematic_questions: list[str],
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """Write the high-level research plan with major thematic questions."""
    content = "# Research Plan\n\n"
    content = content + "## User Query\n"
    content = content + state["messages"][-1].text + "\n\n"
    content = content + "## Thematic Questions\n\n"
    
    for i, question in enumerate(thematic_questions, 1):
        content = content + f"{i}. {question}\n"

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
            max_retries: int = 2
        ):
    """Run a single Research agent for ONE thematic question."""
    file_hash = generate_hash(f"{theme_id}_{thematic_question}")

    ai_message_instruction = f"""[THEME {theme_id}] {thematic_question}

                        Save research to: researcher/{file_hash}_theme.md
                        Save sources to: researcher/{file_hash}_sources.txt
                        """

    sub_state: DeepAgentState = {
        "messages": state["messages"] + [AIMessage(ai_message_instruction)],
        "user_id": state.get("user_id"),
        "thread_id": state.get("thread_id"),
    }

    for attempt in range(max_retries + 1):
        try:
            researcher_agent.invoke(sub_state)
            return f"✓ Theme {theme_id} research completed (hash: {file_hash})"
        except Exception:
            print(f"Failed. Trying #{attempt} times")

    return f"✗ Theme {theme_id} failed after {max_retries + 1} attempts"

@tool
def run_editor(state: Annotated[DeepAgentState, InjectedState]) -> str:
    """Run the Editor agent to synthesize all research into final report."""
    sub_state: DeepAgentState = {
        "messages": [HumanMessage(content="Read research_plan.md and all files in the researcher/ folder, then synthesize everything into a comprehensive report.md file.")],
        "user_id": state.get("user_id"),
        "thread_id": state.get("thread_id"),
    }
    editor_agent.invoke(sub_state)
    return "Editor completed. Final report is written to report.md."

# ==========================================
# 4. CREATE ORCHESTRATOR AGENT
# ==========================================
orchestrator_agent = create_agent(
    model=llm,
    tools=[write_research_plan, run_researcher, run_editor, cleanup_files],
    system_prompt=ORCHESTRATOR_PROMPT,
    state_schema=DeepAgentState,
    checkpointer=checkpointer
)

# ==========================================
# 5. EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    print("Starting Multi-Agent Deep AI Finance Researcher...\n")
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else AGENT_QUERY
    
    stream_agent_response(
        orchestrator_agent,
        query,
        thread_id=AGENT_THREAD_ID,
        user_id=AGENT_USER_ID
    )
