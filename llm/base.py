from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    role: str      # "user" or "assistant"
    content: str


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    text: str | None             # final text answer when no tool call was made
    tool_calls: list[ToolCall]   # list of tool calls the LLM wants to make
    raw: Any                     # raw provider response for debugging

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class BaseLLM(ABC):
    """
    Abstract base class that every LLM provider must implement.
    The rest of the codebase only talks to this interface —
    never directly to Gemini, Anthropic, or OpenAI SDKs.
    """

    @abstractmethod
    def chat(self, messages: list[Message], tools: list = None) -> LLMResponse:
        """Send messages to the LLM and return a structured response."""
        pass

    @abstractmethod
    def build_tool_result_message(
        self,
        tool_call: ToolCall,
        result: str,
        original_response: Any
    ) -> dict:
        """
        Wrap a tool execution result in the provider-specific format
        so it can be sent back to the LLM in the next API call.
        """
        pass