from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from .state import AgentState
from .nodes import agent_node, tools_node, should_continue


def build_graph():
    """
    Assembles and compiles the agent graph.

    Graph structure:
        START → agent → (conditional) → tools → agent → ... → END
                                      ↘ end → END
    """
    # Initialize graph with our state schema
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)

    # Entry point — graph always starts at agent node
    graph.set_entry_point("agent")

    # Conditional edge — after agent, where do we go?
    graph.add_conditional_edges(
        "agent",          # source node
        should_continue,  # routing function
        {
            "tools": "tools",  # tool call detected → execute tools
            "end": END         # no tool call → graph ends
        }
    )

    # Fixed edge — after tools, always go back to agent
    # Agent sees tool results and decides next step
    graph.add_edge("tools", "agent")

    # Checkpointer — saves state after every node
    # Enables cross-session memory and resume capability
    checkpointer = MemorySaver()

    return graph.compile(checkpointer=checkpointer)


# Build once, reuse everywhere
agent_app = build_graph()