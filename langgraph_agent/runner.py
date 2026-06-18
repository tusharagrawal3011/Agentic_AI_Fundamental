import os
from langchain_core.messages import HumanMessage
from .graph import agent_app
from .state import AgentState


def extract_text(message) -> str:
    """
    Extract clean text from LangChain message.
    Handles both string content and list-of-dicts content
    (Gemini thinking models return list format).
    """
    content = message.content

    # Simple string — most models
    if isinstance(content, str):
        return content

    # List of content blocks — Gemini thinking models
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block["text"])
        return "\n".join(text_parts) if text_parts else str(content)

    return str(content)


def run_agent(user_input: str, thread_id: str = "default") -> str:
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_input)],
        "tool_results": [],
        "step_count": 0
    }

    print(f"\nUser: {user_input}")
    print("-" * 50)

    result = agent_app.invoke(initial_state, config=config)

    # Extract clean text from final message
    final_message = result["messages"][-1]
    answer = extract_text(final_message)

    print(f"Agent: {answer}")
    print(f"Steps taken: {result['step_count']}")

    return answer


def run_agent_streaming(user_input: str, thread_id: str = "default"):
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_input)],
        "tool_results": [],
        "step_count": 0
    }

    print(f"\nUser: {user_input}")
    print("Agent: ", end="", flush=True)

    for message_chunk, metadata in agent_app.stream(
        initial_state,
        config=config,
        stream_mode="messages"
    ):
        if hasattr(message_chunk, "content"):
            content = message_chunk.content
            # Extract text from list format
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        print(block["text"], end="", flush=True)
            elif isinstance(content, str) and content:
                print(content, end="", flush=True)

    print()