"""Utility functions for agent operations."""

# ✅ FIX: langchain_core.messages, not langchain.messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


def stream_agent_response(agent, query: str, thread_id: str = "default", user_id: str = None):
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    state = {"messages": [HumanMessage(query)], "thread_id": thread_id, "user_id": user_id}

    for chunk in agent.stream(state, stream_mode="messages", config=config):
        message = chunk[0] if isinstance(chunk, tuple) else chunk

        if isinstance(message, AIMessage) and message.tool_calls:
            for tc in message.tool_calls:
                print(f"\n  Tool Called : {tc['name']}")
                print(f"  Args        : {tc['args']}\n")

        elif isinstance(message, ToolMessage):
            # ✅ FIX: ToolMessage exposes content, not .text
            length = len(message.content) if isinstance(message.content, str) else 0
            print(f"\n  Tool Result : {length} chars\n")

        elif isinstance(message, AIMessage):
            # ✅ FIX: AIMessage uses .content, not .text
            content = message.content
            if isinstance(content, list):
                text = "".join(b.get("text", "") for b in content if isinstance(b, dict))
            else:
                text = str(content)
            if text:
                print(text, end="", flush=True)
