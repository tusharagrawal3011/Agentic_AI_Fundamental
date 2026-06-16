import os
from llm import get_llm, Message, ToolDispatcher
from tools import ALL_TOOLS
from memory import EpisodicMemory

# Provider setup
llm = get_llm(provider="gemini")
dispatcher = ToolDispatcher(ALL_TOOLS)

# Memory setup
memory = EpisodicMemory(storage_path="memory_store.json")

print(f"Available tools: {dispatcher.list_tools()}")
print(f"Past sessions in memory: {len(memory.memories)}\n")


def generate_summary(conversation: list[Message]) -> str:
    """
    At the end of a session, ask the LLM to summarize
    what it learned about the user and what was discussed.
    This summary gets stored in episodic memory.
    """
    # Build a readable transcript
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
    Process one user message and return updated conversation history.
    Handles tool calls automatically via dispatcher.
    """
    # Add user message to history
    conversation_history.append(Message(role="user", content=user_input))

    # Send full history to LLM (working memory = entire history)
    response = llm.chat(conversation_history, tools=ALL_TOOLS)

    if response.has_tool_calls:
        for tool_call in response.tool_calls:
            print(f"  Tool: {tool_call.name}({tool_call.arguments})")
            result = dispatcher.execute(tool_call.name, tool_call.arguments)
            print(f"  Result: {result}")

            final_answer = llm.send_tool_result(
                user_question=user_input,
                tool_call=tool_call,
                result=result
            )
            print(f"Agent: {final_answer}\n")
            conversation_history.append(
                Message(role="assistant", content=final_answer)
            )
    else:
        print(f"Agent: {response.text}\n")
        conversation_history.append(
            Message(role="assistant", content=response.text)
        )

    return conversation_history


def start_session():
    """
    Run an interactive multi-turn conversation session.
    Loads past memory at start, saves summary at end.
    """
    print("=" * 50)
    print("Agent started. Type 'quit' to end session.")
    print("=" * 50 + "\n")

    # Build initial context from past sessions
    conversation_history: list[Message] = []

    if memory.has_memories():
        past_context = memory.build_context(max_sessions=3)
        # Inject as system-style context at the start
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
            # Generate and save session summary
            print("\n[Saving session to memory...]")
            summary = generate_summary(conversation_history)
            memory.save_session(summary)
            print("Session ended.\n")
            break

        conversation_history = run_agent(user_input, conversation_history)


if __name__ == "__main__":
    start_session()