"""Yahoo Finance MCP module with LangGraph integration."""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import asyncio
import shutil

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# ✅ FIX: create_react_agent from langgraph, not create_agent from langchain.agents
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from scripts.config import GROQ_API_KEY


def _uvx_command() -> str:
    venv_uvx = os.path.join(os.path.dirname(sys.executable), "uvx.exe" if sys.platform == "win32" else "uvx")
    if os.path.exists(venv_uvx):
        return venv_uvx
    return shutil.which("uvx") or "uvx"

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile"),
    temperature=0,
    max_tokens=1024,   # ✅ OPT: cap tokens for live finance answers
)

# ✅ OPT: Trimmed system prompt — same instructions, ~60% fewer tokens
_SYSTEM_PROMPT = """\
You are a financial research assistant using Yahoo Finance data.

Tools available:
- get_historical_stock_prices(ticker, period='1mo', interval='1d')
- get_stock_info(ticker) — price, metrics, financials
- get_yahoo_finance_news(ticker)
- get_stock_actions(ticker) — dividends, splits
- get_financial_statement(ticker, financial_type) — income_stmt, balance_sheet, cashflow (quarterly variants too)
- get_holder_info(ticker, holder_type) — major_holders, institutional_holders, etc.
- get_option_expiration_dates(ticker)
- get_option_chain(ticker, expiration_date, option_type) — 'calls' or 'puts'
- get_recommendations(ticker, recommendation_type, months_back=12)

Rules:
1. Always call at least one tool before responding.
2. Extract the ticker from the query (AAPL, MSFT, AMZN, META).
3. For general queries, start with get_stock_info.
4. Present key numbers and trends clearly. Be concise.
"""


async def _get_tools():
    client = MultiServerMCPClient(
        {
            "yahoo-finance": {
                "command": _uvx_command(),
                "args": ["yahoo-finance-mcp-server"],
                "transport": "stdio",
            }
        }
    )
    return await client.get_tools()


async def finance_research(query: str) -> str:
    tools = await _get_tools()

    # ✅ FIX: create_react_agent uses `prompt=` not `system_prompt=`
    agent  = create_react_agent(model=llm, tools=tools, prompt=_SYSTEM_PROMPT)
    result = await agent.ainvoke({"messages": [HumanMessage(query)]})

    last   = result["messages"][-1]
    # ✅ FIX: AIMessage uses .content, not .text
    content = last.content
    if isinstance(content, list):
        text = "".join(b.get("text", "") for b in content if isinstance(b, dict))
    else:
        text = str(content)

    print(text)
    return text


if __name__ == "__main__":
    query = "Current stock price and recent news for Apple (AAPL)."
    asyncio.run(finance_research(query))
