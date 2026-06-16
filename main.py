import os
from llm import get_llm, Message
from tools import calculator

# API key is set in terminal as environment variable:
# $env:GEMINI_API_KEY="your-key-here"
# No .env file needed — os.environ reads it directly
llm = get_llm(provider="gemini")
tools = [calculator]


def run_agent(user_question: str):
    print(f"User: {user_question}\n")

    messages = [Message(role="user", content=user_question)]

    # Step 1: Send question + available tools to LLM
    response = llm.chat(messages, tools=tools)

    print(f"DEBUG - has_tool_calls: {response.has_tool_calls}")
    print(f"DEBUG - tool_calls: {response.tool_calls}")
    print()

    # Step 2: Check if LLM decided to call a tool
    if response.has_tool_calls:
        for tool_call in response.tool_calls:
            print(f"LLM wants to call: {tool_call.name}({tool_call.arguments})")

            # Step 3: Execute the tool (your code runs the actual function)
            if tool_call.name == "calculator":
                result = calculator(**tool_call.arguments)
                print(f"Tool result: {result}\n")

            # Step 4: Send result back to LLM and get the final natural language answer
            final_answer = llm.send_tool_result(
                user_question=user_question,
                tool_call=tool_call,
                result=result
            )
            print(f"Final answer: {final_answer}")

    else:
        # LLM answered directly without needing any tool
        print(f"Answer: {response.text}")

    print()


if __name__ == "__main__":
    run_agent("Use the calculator tool to multiply 847 by 23")
    print("---")
    run_agent("What is the capital of France?")