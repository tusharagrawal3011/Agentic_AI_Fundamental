import os
import anthropic
from .base import BaseLLM, Message, LLMResponse, ToolCall


class AnthropicLLM(BaseLLM):

    def __init__(self, model: str = "claude-sonnet-4-5"):
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    def chat(self, messages: list[Message], tools: list = None) -> LLMResponse:
        anthropic_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
            if isinstance(m.content, str)
        ]

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            tools=tools or [],
            messages=anthropic_messages
        )

        tool_calls = []
        for block in response.content:
            if block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input
                ))

        text = None
        for block in response.content:
            if hasattr(block, "text"):
                text = block.text
                break

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            raw=response
        )

    def build_tool_result_message(self, tool_call, result, original_response):
        return {
            "type": "anthropic_tool_result",
            "tool_use_id": tool_call.id,
            "result": result,
            "original_response": original_response
        }