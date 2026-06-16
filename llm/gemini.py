import os
from google import genai
from google.genai import types
from .base import BaseLLM, Message, LLMResponse, ToolCall


class GeminiLLM(BaseLLM):

    def __init__(self, model: str = "gemini-3.5-flash"):
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model = model
        self._tools = None
        self._last_raw_response = None  # stores last response to preserve thought_signature

    def chat(self, messages: list[Message], tools: list = None) -> LLMResponse:
        self._tools = tools

        # Convert messages to Gemini-compatible Content format
        gemini_contents = []
        for m in messages:
            if isinstance(m.content, str):
                role = "user" if m.role == "user" else "model"
                gemini_contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=m.content)]
                    )
                )

        # Build config with tool declarations if tools are provided
        if tools:
            config = types.GenerateContentConfig(
                tools=[types.Tool(function_declarations=[
                    self._python_fn_to_declaration(fn) for fn in tools
                ])],
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode="AUTO"  # LLM decides whether to call a tool or answer directly
                    )
                )
            )
        else:
            config = None

        response = self.client.models.generate_content(
            model=self.model,
            contents=gemini_contents,
            config=config
        )

        # Store raw response — needed to preserve thought_signature for thinking models
        self._last_raw_response = response

        # Parse tool calls from response parts
        tool_calls = []
        candidate = response.candidates[0]

        for part in candidate.content.parts:
            if hasattr(part, 'function_call') and part.function_call is not None:
                fn = part.function_call
                if fn.name:  # skip empty function call objects
                    tool_calls.append(ToolCall(
                        id=fn.name,
                        name=fn.name,
                        arguments=dict(fn.args) if fn.args else {}
                    ))

        # Parse text response (only when no tool calls were made)
        text = None
        if not tool_calls:
            try:
                text = response.text
            except Exception:
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                        text = part.text
                        break

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            raw=response
        )

    def build_tool_result_message(self, tool_call, result, original_response):
        return {
            "type": "gemini_tool_result",
            "tool_name": tool_call.name,
            "result": result,
            "original_response": original_response
        }

    def send_tool_result(self, user_question: str, tool_call: ToolCall, result: str) -> str:
        """
        Send tool execution result back to Gemini.
        For thinking models (3.x series), the original model response content
        must be passed back AS-IS to preserve its internal thought_signature.
        """
        # Original model response (contains thought_signature internally)
        original_model_content = self._last_raw_response.candidates[0].content

        contents = [
            # 1. Original user question
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_question)]
            ),
            # 2. Original model response AS-IS — preserves thought_signature
            original_model_content,
            # 3. Tool execution result
            types.Content(
                role="user",
                parts=[types.Part.from_function_response(
                    name=tool_call.name,
                    response={"result": result}
                )]
            )
        ]

        config = types.GenerateContentConfig(
            tools=[types.Tool(function_declarations=[
                self._python_fn_to_declaration(fn) for fn in (self._tools or [])
            ])]
        ) if self._tools else None

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config
        )

        return response.text

    def _python_fn_to_declaration(self, fn) -> types.FunctionDeclaration:
        """
        Convert a Python function into a Gemini FunctionDeclaration.
        Extracts name, description, and parameter types from the function's
        docstring and type annotations — no manual JSON schema needed.
        """
        import inspect

        sig = inspect.signature(fn)
        doc = inspect.getdoc(fn) or ""

        lines = doc.split('\n')
        description = lines[0] if lines else fn.__name__

        # Parse parameter descriptions from the Args section of the docstring
        param_docs = {}
        in_args = False
        for line in lines:
            line = line.strip()
            if line == "Args:":
                in_args = True
                continue
            if line in ("Returns:", "Raises:"):
                in_args = False
                continue
            if in_args and ":" in line:
                param_name, param_desc = line.split(":", 1)
                param_docs[param_name.strip()] = param_desc.strip()

        type_map = {
            str: "string",
            float: "number",
            int: "integer",
            bool: "boolean",
        }

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            param_type = type_map.get(param.annotation, "string")

            properties[param_name] = types.Schema(
                type=param_type,
                description=param_docs.get(param_name, param_name)
            )

            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        return types.FunctionDeclaration(
            name=fn.__name__,
            description=description,
            parameters=types.Schema(
                type="object",
                properties=properties,
                required=required
            )
        )