from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode
from .state import AgentState
from .tools import ALL_TOOLS

# LLM setup — bind tools so LLM knows what's available
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
llm_with_tools = llm.bind_tools(ALL_TOOLS)


def agent_node(state: AgentState) -> dict:
    """
    The brain of the agent.
    Receives full conversation history, decides next action.
    Returns either a tool call or a final text answer.
    """
    response = llm_with_tools.invoke(state["messages"])

    return {
        "messages": [response],
        "step_count": state["step_count"] + 1
    }


# ToolNode is a LangGraph built-in that:
# 1. Reads tool_calls from the last message
# 2. Executes the matching tool function
# 3. Returns ToolMessage with the result
# We don't need to write dispatcher logic manually!
tools_node = ToolNode(ALL_TOOLS)


def should_continue(state: AgentState) -> str:
    """
    Conditional edge function — decides where to go after agent node.

    Returns:
        "tools" — if LLM made a tool call
        "end"   — if LLM gave a final text answer
    """
    last_message = state["messages"][-1]

    # Safety limit — prevent infinite loops
    if state["step_count"] >= 5:
        return "end"

    # Does the last message have tool calls?
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "end"