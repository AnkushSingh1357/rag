from __future__ import annotations

import subprocess
import sys
import uuid
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
import fastapi.middleware.cors
from langchain.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

load_dotenv()

from scripts.config import AGENT_THREAD_ID, AGENT_USER_ID


app = FastAPI(
    title="Finance RAG API",
    description="API wrapper for the multi-agent finance RAG researcher.",
    version="1.0.0",
)

app.add_middleware(
    fastapi.middleware.cors.CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Financial research question")
    user_id: str = Field(default=AGENT_USER_ID, description="User/session owner")
    thread_id: str | None = Field(
        default=None,
        description="Conversation thread. Leave empty to create a fresh thread.",
    )


class ResearchResponse(BaseModel):
    answer: str
    user_id: str
    thread_id: str


class IngestResponse(BaseModel):
    status: str
    output: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part)

    return str(content) if content else ""


def _run_research(query: str, user_id: str, thread_id: str) -> str:
    from main import orchestrator_agent

    state = {
        "messages": [HumanMessage(content=query)],
        "user_id": user_id,
        "thread_id": thread_id,
    }
    config = {"configurable": {"user_id": user_id, "thread_id": thread_id}}

    result = orchestrator_agent.invoke(state, config=config)
    messages = result.get("messages", [])

    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = _message_text(message).strip()
            if text:
                return text

    return "Research completed, but no final answer was returned."


@app.post("/ingest", response_model=IngestResponse)
async def ingest() -> IngestResponse:
    def run_ingestion() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "setup_db.py"],
            capture_output=True,
            text=True,
            check=False,
        )

    result = await run_in_threadpool(run_ingestion)
    output = (result.stdout + "\n" + result.stderr).strip()

    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=output)

    return IngestResponse(status="completed", output=output)


@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest) -> ResearchResponse:
    thread_id = request.thread_id or f"{AGENT_THREAD_ID}-{uuid.uuid4().hex[:8]}"

    try:
        answer = await run_in_threadpool(
            _run_research,
            request.query,
            request.user_id,
            thread_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ResearchResponse(answer=answer, user_id=request.user_id, thread_id=thread_id)


