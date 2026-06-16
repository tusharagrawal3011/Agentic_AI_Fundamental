import os
from llm import get_llm, Message, ToolDispatcher
from tools import ALL_TOOLS

# Change provider here — "gemini" or "anthropic"
llm = get_llm(provider="gemini")

# Dispatcher automatically routes any tool call to the right function
dispatcher = ToolDispatcher(ALL_TOOLS)

print(f"Available tools: {dispatcher.list_tools()}\n")


def run_agent(user_question: str):
    print(f"User: {user_question}")
    print("-" * 50)

    messages = [Message(role="user", content=user_question)]

    # Step 1: Send question + all tools to LLM
    response = llm.chat(messages, tools=ALL_TOOLS)

    # Step 2: Handle tool calls
    if response.has_tool_calls:
        for tool_call in response.tool_calls:
            print(f"Tool called : {tool_call.name}")
            print(f"Arguments  : {tool_call.arguments}")

            # Step 3: Dispatcher executes the correct function automatically
            result = dispatcher.execute(tool_call.name, tool_call.arguments)
            print(f"Result     : {result}")

            # Step 4: Send result back, get final answer
            final_answer = llm.send_tool_result(
                user_question=user_question,
                tool_call=tool_call,
                result=result
            )
            print(f"Answer     : {final_answer}")
    else:
        print(f"Answer     : {response.text}")

    print()


if __name__ == "__main__":
    run_agent("What is 1234 multiplied by 567?")
    run_agent("What is the weather in Bangalore?")
    run_agent("How many words are in the sentence: The quick brown fox jumps over the lazy dog")
    run_agent("What is the capital of Japan?")