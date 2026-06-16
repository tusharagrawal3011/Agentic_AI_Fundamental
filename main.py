import os
from llm import get_llm, Message, ToolDispatcher
from tools import ALL_TOOLS
from memory import EpisodicMemory

# Provider setup
llm = get_llm(provider="gemini")
dispatcher = ToolDispatcher(ALL_TOOLS)
memory = EpisodicMemory(storage_path="memory_store.json")

print(f"Available tools: {dispatcher.list_tools()}")
print(f"Past sessions in memory: {len(memory.memories)}\n")


def generate_summary(conversation: list[Message]) -> str:
    """Ask LLM to summarize the conversation for episodic memory."""
    transcript = "\n".join([
        f"{m.role.upper()}: {m.content}"
        for m in conversation
        if isinstance(m.content, str)
    ])

    summary_prompt = f"""Summarize this conversation in 1-2 sentences.
Focus on: who the user is, what they asked, any personal details mentioned.
Be concise — this will be used as context in future sessions.

Conversation:
{transcript}

Summary:"""

    response = llm.chat(
        messages=[Message(role="user", content=summary_prompt)]
    )
    return response.text.strip()


def run_agent(user_input: str, conversation_history: list[Message]) -> list[Message]:
    """
    Multi-step ReAct loop.

    The agent keeps reasoning and calling tools until it has
    enough information to give a final answer — then it stops.

    Each tool result is added back to conversation history so
    the LLM can see what it already found and decide what to do next.
    """
    # Add user message to working memory
    conversation_history.append(Message(role="user", content=user_input))

    step = 1  # track how many reasoning steps we take

    # ReAct loop — runs until LLM gives a final answer (no tool call)
    while True:
        print(f"  [Step {step}] Thinking...")

        # Send full conversation history to LLM
        # LLM sees: past context + user question + all previous tool results
        response = llm.chat(conversation_history, tools=ALL_TOOLS)

        if response.has_tool_calls:
            # LLM decided it needs more information — execute the tool
            for tool_call in response.tool_calls:
                print(f"  [Step {step}] Tool called: {tool_call.name}({tool_call.arguments})")

                # Execute the actual function via dispatcher
                result = dispatcher.execute(tool_call.name, tool_call.arguments)
                print(f"  [Step {step}] Result: {result}")

                # Get final answer after tool result
                final_answer = llm.send_tool_result(
                    user_question=user_input,
                    tool_call=tool_call,
                    result=result
                )

                # Add tool result to conversation history as assistant message
                # This is how LLM "remembers" what it already found
                tool_summary = f"[Tool: {tool_call.name}] Result: {result}"
                conversation_history.append(
                    Message(role="assistant", content=tool_summary)
                )

            step += 1

            # Safety limit — prevent infinite loops
            # In production this would be configurable
            if step > 10:
                print("  [Warning] Max steps reached, stopping loop.")
                final_answer = "I was unable to complete the task within the allowed steps."
                break

        else:
            # LLM gave a direct answer — no more tools needed
            # This is the exit condition of the ReAct loop
            final_answer = response.text
            break

    print(f"Agent: {final_answer}\n")
    conversation_history.append(
        Message(role="assistant", content=final_answer)
    )

    return conversation_history


def start_session():
    """
    Run an interactive multi-turn conversation.
    Loads past memory at start, saves summary at end.
    """
    print("=" * 50)
    print("Agent started. Type 'quit' to end session.")
    print("=" * 50 + "\n")

    conversation_history: list[Message] = []

    # Inject episodic memory as context at session start
    if memory.has_memories():
        past_context = memory.build_context(max_sessions=3)
        conversation_history.append(
            Message(role="user", content=past_context)
        )
        conversation_history.append(
            Message(role="assistant", content="Understood. I have context from our previous conversations.")
        )
        print(f"[Memory loaded]\n{past_context}\n")
    else:
        print("[No past memory — fresh start]\n")

    # Main conversation loop
    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("\n[Saving session to memory...]")
            summary = generate_summary(conversation_history)
            memory.save_session(summary)
            print("Session ended.\n")
            break

        conversation_history = run_agent(user_input, conversation_history)


if __name__ == "__main__":
    start_session()