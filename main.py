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
    conversation_history.append(Message(role="user", content=user_input))

    step = 1
    final_answer = ""
    tool_results = []

    # ── Phase 1: ReAct loop ──
    while True:
        print(f"  [Step {step}] Thinking...")
        response = llm.chat(conversation_history, tools=ALL_TOOLS)

        if response.has_tool_calls:
            for tool_call in response.tool_calls:
                print(f"  [Step {step}] Tool called: {tool_call.name}({tool_call.arguments})")
                result = dispatcher.execute(tool_call.name, tool_call.arguments)
                print(f"  [Step {step}] Result: {result}")

                tool_results.append({
                    "tool": tool_call.name,
                    "args": tool_call.arguments,
                    "result": result
                })

            step += 1

            # After all tool calls in this step, ask LLM for next action
            # Build a summary of what we know so far
            results_so_far = "\n".join([
                f"- {t['tool']}({t['args']}) = {t['result']}"
                for t in tool_results
            ])

            # Add tool results as a user message — tell LLM "here's what you found"
            conversation_history.append(Message(
                role="user",
                content=f"Tool results so far:\n{results_so_far}\n\nDo you need more tools, or can you give the final answer now?"
            ))

            if step > 5:
                print("  [Warning] Max steps reached.")
                final_answer = f"Based on tool results: {results_so_far}"
                break

        else:
            # LLM gave text answer — this is the final answer
            final_answer = response.text
            break

    # ── Phase 2: Self-correction (only for calculations) ──
    calc_used = any(t["tool"] == "calculator" for t in tool_results)

    if calc_used:
        print(f"  [Verifying...]")

        results_summary = "\n".join([
            f"- {t['tool']}({t['args']}) = {t['result']}"
            for t in tool_results
        ])

        verify_messages = [
            Message(
                role="user",
                content=f"""Question was: {user_input}

Tool results:
{results_summary}

Agent's answer: {final_answer}

Is the answer consistent with the tool results?
Reply ONLY with:
CORRECT: [clean final answer]
OR
CORRECTION: [corrected answer]

Do not call any tools."""
            )
        ]

        verify_response = llm.chat(verify_messages, tools=None)
        verify_text = verify_response.text.strip()

        if verify_text.upper().startswith("CORRECTION:"):
            print(f"  [Self-correction triggered]")
            final_answer = verify_text.split(":", 1)[-1].strip()
        elif verify_text.upper().startswith("CORRECT:"):
            final_answer = verify_text.split(":", 1)[-1].strip()

    print(f"Agent: {final_answer}\n")
    conversation_history.append(Message(role="assistant", content=final_answer))
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