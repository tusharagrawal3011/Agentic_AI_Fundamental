import os
from langgraph_agent.runner import run_agent, run_agent_streaming

# Test 1: Simple question — no tool needed
run_agent(
    "What is the capital of Japan?",
    thread_id="test-1"
)

print("\n" + "="*50 + "\n")

# Test 2: Calculator tool
run_agent(
    "What is 847 multiplied by 23?",
    thread_id="test-2"
)

print("\n" + "="*50 + "\n")

# Test 3: Multi-tool — weather + calculator
run_agent(
    "What is the weather in Bangalore, and calculate 28 times 350?",
    thread_id="test-3"
)

print("\n" + "="*50 + "\n")

# Test 4: Checkpointer demo — same thread_id, two separate calls
# First call
run_agent(
    "My name is Tushar and I am learning LangGraph",
    thread_id="memory-test"
)

print()

# Second call — SAME thread_id
# LangGraph will have full history from first call
run_agent(
    "What did I just tell you about myself?",
    thread_id="memory-test"
)

print("\n" + "="*50 + "\n")

# Test 5: Streaming
run_agent_streaming(
    "What is the weather in Delhi?",
    thread_id="streaming-test"
)