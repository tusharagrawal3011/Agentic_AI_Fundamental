# Agentic AI Fundamentals

Hands-on learning repo for Agentic AI concepts — building from raw tool calling and memory systems toward a production-grade AI agent, implemented in Python.

This repo is part of a parallel learning track alongside [kafka-go-fundamentals](https://github.com/your-username/kafka-go-fundamentals). Both tracks will eventually merge into a single project: an **AI Agent Event Pipeline** (Go microservices + Kafka event bus + LLM agents).

## Goals

- Understand what makes AI "agentic" — the Perceive → Think → Act → Observe loop
- Implement tool calling from scratch, without frameworks, to understand the underlying mechanics
- Build a provider-agnostic LLM wrapper so switching between Gemini, Anthropic, and OpenAI is a one-line change
- Progress through memory systems, planning patterns, multi-agent orchestration, and production deployment

## Tech stack

- **Python** (3.13+)
- **Google Gemini API** (`gemini-3.5-flash`) via `google-genai` SDK
- **Anthropic API** (Claude) — provider ready, plug in API key to activate
- Provider-agnostic wrapper architecture — no LangChain dependency at this stage

## Project structure

```
agentic-ai-fundamentals/
├── llm/
│   ├── __init__.py          # exports BaseLLM, Message, LLMResponse, ToolCall, get_llm
│   ├── base.py              # abstract interface all providers implement
│   ├── gemini.py            # Gemini implementation (gemini-3.5-flash, thinking model)
│   ├── anthropic_llm.py     # Anthropic implementation (ready, needs API key)
│   └── factory.py           # get_llm("gemini") or get_llm("anthropic")
├── tools/
│   ├── __init__.py
│   └── calculator.py        # sample tool — arithmetic operations
├── main.py                  # agent loop entry point
└── requirements.txt
```

## Setup

### 1. Clone and create virtual environment

```bash
git clone https://github.com/your-username/agentic-ai-fundamentals.git
cd agentic-ai-fundamentals

# Windows
py -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set API key

```bash
# Windows PowerShell
$env:GEMINI_API_KEY="your-gemini-api-key"

# Mac/Linux
export GEMINI_API_KEY="your-gemini-api-key"
```

Get a free Gemini API key at: https://aistudio.google.com/apikey

### 4. Run

```bash
python main.py
```

## How provider switching works

The entire codebase talks to a single interface (`BaseLLM`) — never directly to a provider SDK. To switch providers, change one line in `main.py`:

```python
llm = get_llm(provider="gemini")     # uses Gemini 3.5 Flash
llm = get_llm(provider="anthropic")  # uses Claude Sonnet (needs ANTHROPIC_API_KEY)
```

No other code changes needed.

## How tool calling works

This is implemented from scratch — no framework magic:

```
1. Define tool as a Python function with type hints + docstring
2. Wrapper auto-generates the JSON schema from the function signature
3. LLM receives the question + tool schema, decides whether to call a tool
4. If tool call: your code executes the actual function
5. Result is sent back to LLM in the next API call
6. LLM generates the final natural language answer
```

Key insight: **the LLM never executes anything — it only decides. Your code always executes.**

## Concepts covered

| Concept | Where it's applied |
|---|---|
| ReAct loop (Reason + Act) | `main.py` — run_agent() function |
| Tool schema auto-generation | `llm/gemini.py` — `_python_fn_to_declaration()` |
| Tool call detection | `llm/gemini.py` — parsing `function_call` from response parts |
| thought_signature handling | `llm/gemini.py` — `send_tool_result()` passes original model content back |
| Provider abstraction | `llm/base.py` — `BaseLLM` abstract class |
| Factory pattern | `llm/factory.py` — `get_llm(provider)` |

## Roadmap

- [x] Stage 1 — What makes AI agentic (ReAct loop, LLM as decision engine)
- [x] Stage 2 — Tool calling (single tool, provider-agnostic wrapper)
- [ ] Stage 2 — Multiple tools, tool dispatcher
- [ ] Stage 2 — Memory systems (working, episodic, semantic)
- [ ] Stage 3 — Planning and reasoning patterns
- [ ] Stage 4 — Agent frameworks (LangGraph, custom orchestrator)
- [ ] Stage 5 — Multi-agent systems
- [ ] Stage 6 — Production deployment + Kafka integration

## Related repos

- [`kafka-go-fundamentals`](https://github.com/your-username/kafka-go-fundamentals) — Kafka + Go learning track
- `ai-agent-event-pipeline` — final integrated project (coming soon)