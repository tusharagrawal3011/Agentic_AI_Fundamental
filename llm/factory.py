from .base import BaseLLM
from .gemini import GeminiLLM
from .anthropic_llm import AnthropicLLM


def get_llm(provider: str = "gemini") -> BaseLLM:
    """
    Factory function — returns the correct LLM provider instance.
    Switch providers by changing a single string: "gemini", "anthropic"
    """
    providers = {
        "gemini":    GeminiLLM,
        "anthropic": AnthropicLLM,
    }

    if provider not in providers:
        raise ValueError(
            f"Unknown provider: '{provider}'. "
            f"Available providers: {list(providers.keys())}"
        )

    return providers[provider]()