from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    The state that flows through the entire graph.
    Every node reads from and writes to this state.

    messages uses add_messages annotation — this means LangGraph
    automatically APPENDS new messages rather than replacing the list.
    So nodes just return new messages, not the full history.
    """
    messages: Annotated[list, add_messages]
    tool_results: list      # stores all tool execution results
    step_count: int         # safety counter — prevents infinite loops