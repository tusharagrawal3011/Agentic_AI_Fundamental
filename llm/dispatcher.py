from typing import Callable


class ToolDispatcher:
    """
    Automatically routes tool call requests from the LLM
    to the correct Python function — no if/elif chains needed.
    
    How it works:
    - Register functions by name in a dict
    - When LLM requests tool_name with arguments,
      dispatcher finds the function and calls it
    """

    def __init__(self, tools: list[Callable]):
        # Build a name -> function mapping
        self.registry: dict[str, Callable] = {
            fn.__name__: fn for fn in tools
        }

    def execute(self, tool_name: str, arguments: dict) -> str:
        """
        Execute the requested tool and return result as string.
        Raises ValueError if tool is not registered.
        """
        if tool_name not in self.registry:
            available = list(self.registry.keys())
            raise ValueError(
                f"Tool '{tool_name}' not found. "
                f"Available tools: {available}"
            )

        fn = self.registry[tool_name]

        try:
            result = fn(**arguments)
            return str(result)
        except TypeError as e:
            return f"Error calling {tool_name}: {e}"

    def list_tools(self) -> list[str]:
        """Returns names of all registered tools."""
        return list(self.registry.keys())