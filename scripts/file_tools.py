# scripts/file_tools.py
import os
import hashlib
from typing import Annotated
from typing_extensions import NotRequired

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId

from langgraph.prebuilt import InjectedState
from langgraph.graph import MessagesState
from langgraph.types import Command

BASE_FILE_DIR = os.getenv("AGENT_FILE_BASE_DIR", "agent_files")

# ─────────────────────────────────────────────
# Shared Agent State
# ─────────────────────────────────────────────

class DeepAgentState(MessagesState):
    """
    Shared state for all agents (orchestrator, researcher, editor).
    - user_id:         separates users
    - thread_id:       separates conversations per user
    - chart_paths:     PNG chart paths generated this session
    - remaining_steps: required by create_react_agent (LangGraph >= 0.2)
    """
    user_id: NotRequired[str]
    thread_id: NotRequired[str]
    chart_paths: NotRequired[list[str]]
    remaining_steps: NotRequired[int]  # ✅ Required by create_react_agent


# ─────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────

def _thread_folder(state: DeepAgentState) -> str:
    user   = state.get("user_id")   or "default_user"
    thread = state.get("thread_id") or "default_thread"
    folder = os.path.join(BASE_FILE_DIR, user, thread)
    os.makedirs(folder, exist_ok=True)
    return folder


def generate_hash(text: str, length: int = 6) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:length]


def _disk_path(state: DeepAgentState, file_path: str) -> str:
    folder    = _thread_folder(state)
    safe_path = file_path.lstrip("/\\")
    full      = os.path.join(folder, safe_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    return full


def _safe_content(text: str) -> str:
    """
    Groq rejects ToolMessage with content='' or content=[].
    Always return a non-empty string so the API never sees a blank tool result.
    """
    stripped = (text or "").strip()
    return stripped if stripped else "[no output]"


# ─────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────

@tool(parse_docstring=True)
def ls(
    state: Annotated[DeepAgentState, InjectedState],
    path: str = "",
) -> str:
    """List files for this user/thread on the real filesystem.

    Args:
        state: Injected agent state (provides user_id/thread_id).
        path:  Optional sub-directory (e.g. "researcher"). Lists root if empty.

    Returns:
        Newline-separated filenames, or a descriptive message if empty.
    """
    # ✅ FIX: return str, NOT list[str].
    # Groq rejects ToolMessage with content=[] (empty list), which happened
    # whenever ls() returned [] for an empty/missing folder.
    folder = _thread_folder(state)
    if path:
        folder = os.path.join(folder, path.lstrip("/\\"))
    if not os.path.exists(folder):
        return f"Folder '{path or 'root'}' does not exist yet."
    files = sorted(os.listdir(folder))
    if not files:
        return f"Folder '{path or 'root'}' is empty."
    return "\n".join(files)


@tool(parse_docstring=True)
def read_file(
    file_path: str,
    state: Annotated[DeepAgentState, InjectedState],
    offset: int = 0,
    limit: int = 2000,
) -> str:
    """Read a file from the real filesystem.

    Args:
        file_path: Relative path under this user/thread folder.
        state:     Injected agent state.
        offset:    Line number to start from (0-based).
        limit:     Maximum number of lines to return.

    Returns:
        Numbered file content, or an error message if not found.
    """
    path = _disk_path(state, file_path)
    if not os.path.exists(path):
        return f"Error: File '{file_path}' does not exist."
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    if not lines:
        return f"File '{file_path}' exists but is empty."
    end = min(offset + limit, len(lines))
    return "\n".join(
        f"{i + 1:5d}  {line}" for i, line in enumerate(lines[offset:end])
    )


@tool(parse_docstring=True)
def write_file(
    file_path: str,
    content: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Write content to a file on the real filesystem (overwrites if exists).

    Args:
        file_path:    Relative path (e.g. "plan.md", "notes/sources.txt").
        content:      Text content to write.
        state:        Injected agent state.
        tool_call_id: Attached to the confirmation ToolMessage.

    Returns:
        Command with a ToolMessage confirming the write.
    """
    path = _disk_path(state, file_path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    msg = _safe_content(f"[FILE WRITTEN] {file_path} ({len(content)} chars)")
    return Command(
        update={"messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]}
    )


@tool(parse_docstring=True)
def cleanup_files(
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Delete ALL files for this user/thread. DESTRUCTIVE — only call when user explicitly asks.

    Args:
        state:        Injected agent state with user_id/thread_id.
        tool_call_id: Used to attach the result ToolMessage.

    Returns:
        Command with a ToolMessage summarising what was deleted.
    """
    folder = _thread_folder(state)
    if not os.path.exists(folder):
        msg = _safe_content("[CLEANUP] No workspace folder — nothing to delete.")
        return Command(update={"messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]})

    deleted, errors = [], []
    for name in os.listdir(folder):
        full = os.path.join(folder, name)
        if os.path.isfile(full):
            try:
                os.remove(full)
                deleted.append(name)
            except Exception as e:
                errors.append(f"{name}: {e}")

    parts = []
    if deleted:
        parts.append(f"Deleted {len(deleted)} file(s): {', '.join(deleted)}")
    if errors:
        parts.append(f"Errors: {'; '.join(errors)}")
    if not parts:
        parts.append("No files to delete.")

    msg = _safe_content("[CLEANUP] " + " | ".join(parts))
    return Command(update={"messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]})
