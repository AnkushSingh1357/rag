import os
import sqlite3
import sys
import warnings
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)
warnings.filterwarnings('ignore')

from langchain_core.messages import ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from scripts.llm_utils import RotatingChatGroq, get_rotating_llm

# DeepAgent imports
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

# Custom local imports
from scripts.rag_tools import hybrid_search, live_finance_researcher, think_tool, format_chart_data
from scripts.deep_prompts import DEEP_RESEARCHER_INSTRUCTIONS, DEEP_ORCHESTRATOR_INSTRUCTIONS
from scripts.agent_utils import stream_agent_response

from scripts.config import (
    AGENT_THREAD_ID,
    AGENT_USER_ID,
    DEEP_AGENT_CHECKPOINT_DB,
    DEEP_AGENT_QUERY,
    RESEARCH_OUTPUT_DIR,
)

# ==========================================
# SAFE GROQ WRAPPER (Prevents tool crashes)
# ==========================================
class SafeChatGroq(RotatingChatGroq):
    def invoke(self, input, *args, **kwargs):
        if isinstance(input, list):
            for msg in input:
                if isinstance(msg, ToolMessage) and not msg.content:
                    msg.content = "Action completed successfully."
        return super().invoke(input, *args, **kwargs)

    def stream(self, input, *args, **kwargs):
        if isinstance(input, list):
            for msg in input:
                if isinstance(msg, ToolMessage) and not msg.content:
                    msg.content = "Action completed successfully."
        return super().stream(input, *args, **kwargs)

# ==========================================
# 1. SETUP FILE BACKEND
# ==========================================
RESEARCH_OUTPUT_DIR = os.path.abspath(RESEARCH_OUTPUT_DIR)

def get_research_backend(user_id, thread_id):
    USER_OUTPUT_DIR = os.path.join(RESEARCH_OUTPUT_DIR, user_id, thread_id)
    os.makedirs(USER_OUTPUT_DIR, exist_ok=True)
    return FilesystemBackend(root_dir=USER_OUTPUT_DIR, virtual_mode=True)

# ==========================================
# 2. INITIALIZE GROQ LLMS 
# ==========================================
# 🧠 Heavy Brain (Orchestrator) - Running safely on Groq
llm = get_rotating_llm(
    model_name=os.getenv("GROQ_FAST_MODEL", "llama-3.3-70b-versatile"),
    temperature=0,
    llm_cls=SafeChatGroq
)

# ⚡ Fast Worker (Researcher) 
fast_llm = get_rotating_llm(
    model_name=os.getenv("GROQ_FAST_MODEL", "llama-3.3-70b-versatile"),
    temperature=0,
    llm_cls=SafeChatGroq
)

# ==========================================
# 3. CREATE RESEARCH SUB-AGENT
# ==========================================
current_date = datetime.now().strftime("%Y-%m-%d")

tools = [hybrid_search, live_finance_researcher, think_tool, format_chart_data]

research_sub_agent = {
    "name": "financial-research-agent",
    "description": "Delegate financial research and chart formatting to this sub-agent.",
    "system_prompt": DEEP_RESEARCHER_INSTRUCTIONS.format(date=current_date),
    "tools": tools,
    "model": fast_llm,
}

# ==========================================
# 4. INITIALIZE DEEP AGENT ORCHESTRATOR
# ==========================================
def get_deep_agent(user_id, thread_id):
    conn = sqlite3.connect(DEEP_AGENT_CHECKPOINT_DB, check_same_thread=False)
    checkpointer = SqliteSaver(conn=conn)
    backend = get_research_backend(user_id, thread_id)

    agent = create_deep_agent(
        model=llm, 
        tools=tools,
        system_prompt=DEEP_ORCHESTRATOR_INSTRUCTIONS,
        subagents=[research_sub_agent],
        checkpointer=checkpointer, 
        backend=backend, 
    )
    return agent