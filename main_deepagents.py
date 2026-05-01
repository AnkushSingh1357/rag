import os
import sqlite3
import sys
import warnings
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
warnings.filterwarnings('ignore')

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver

# DeepAgent imports
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

# Custom local imports
from scripts.rag_tools import hybrid_search, live_finance_researcher, think_tool
from scripts.deep_prompts import DEEP_RESEARCHER_INSTRUCTIONS, DEEP_ORCHESTRATOR_INSTRUCTIONS
from scripts.agent_utils import stream_agent_response
from scripts.config import (
    AGENT_THREAD_ID,
    AGENT_USER_ID,
    DEEP_AGENT_CHECKPOINT_DB,
    DEEP_AGENT_QUERY,
    GEMINI_CHAT_MODEL,
    RESEARCH_OUTPUT_DIR,
)

# ==========================================
# 1. SETUP FILE BACKEND
# ==========================================
RESEARCH_OUTPUT_DIR = os.path.abspath(RESEARCH_OUTPUT_DIR)

def get_research_backend(user_id, thread_id):
    USER_OUTPUT_DIR = os.path.join(RESEARCH_OUTPUT_DIR, user_id, thread_id)
    os.makedirs(USER_OUTPUT_DIR, exist_ok=True)
    print(f"Writing research files to: {USER_OUTPUT_DIR}")

    # Create filesystem backend with virtual_mode=True for security
    backend = FilesystemBackend(
        root_dir=USER_OUTPUT_DIR,
        virtual_mode=True 
    )
    return backend

# ==========================================
# 2. CREATE RESEARCH SUB-AGENT
# ==========================================
current_date = datetime.now().strftime("%Y-%m-%d")

research_sub_agent = {
    "name": "financial-research-agent",
    "description": "Delegate financial research to this sub-agent. Give it one specific research task at a time.",
    "system_prompt": DEEP_RESEARCHER_INSTRUCTIONS.format(date=current_date),
    "tools": [hybrid_search, live_finance_researcher, think_tool],
}

# ==========================================
# 3. INITIALIZE DEEP AGENT ORCHESTRATOR
# ==========================================
model = ChatGoogleGenerativeAI(model=GEMINI_CHAT_MODEL)
tools = [hybrid_search, live_finance_researcher, think_tool]

def get_deep_agent(user_id, thread_id):
    # SQLite checkpointer for agent memory
    conn = sqlite3.connect(DEEP_AGENT_CHECKPOINT_DB, check_same_thread=False)
    checkpointer = SqliteSaver(conn=conn)
    backend = get_research_backend(user_id, thread_id)

    # Create the deep agent with memory and secure file backend
    agent = create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=DEEP_ORCHESTRATOR_INSTRUCTIONS,
        subagents=[research_sub_agent],
        checkpointer=checkpointer, 
        backend=backend, 
    )
    return agent

# ==========================================
# 4. EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    print("Starting LangChain DeepAgent Finance Researcher...\n")
    
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEEP_AGENT_QUERY
    user_id = AGENT_USER_ID
    thread_id = AGENT_THREAD_ID

    agent = get_deep_agent(user_id, thread_id)
    stream_agent_response(agent, query, thread_id)
