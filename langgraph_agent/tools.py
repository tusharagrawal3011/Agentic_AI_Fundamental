from langchain_core.tools import tool


# LangGraph uses LangChain's @tool decorator
# This auto-generates the schema AND makes the function callable by LLM

@tool
def calculator(operation: str, a: float, b: float) -> str:
    """
    Performs basic arithmetic operations.

    Args:
        operation: The operation to perform — one of: add, subtract, multiply, divide
        a: First number
        b: Second number
    """
    if operation == "add":
        return str(a + b)
    elif operation == "subtract":
        return str(a - b)
    elif operation == "multiply":
        return str(a * b)
    elif operation == "divide":
        if b == 0:
            return "Error: division by zero"
        return str(a / b)
    return f"Error: unknown operation {operation}"


@tool
def get_weather(city: str) -> str:
    """
    Get the current weather for a given city.

    Args:
        city: Name of the city to get weather for
    """
    weather_data = {
        "bangalore": "28°C, partly cloudy",
        "mumbai": "32°C, humid and sunny",
        "delhi": "38°C, hot and hazy",
        "pune": "27°C, clear skies",
        "hyderabad": "31°C, mostly sunny",
    }
    city_lower = city.lower()
    if city_lower in weather_data:
        return f"Weather in {city}: {weather_data[city_lower]}"
    return f"Weather data not available for {city}"


@tool
def word_count(text: str) -> str:
    """
    Count the number of words in a given text.

    Args:
        text: The input text to count words in
    """
    count = len(text.split())
    return f"The text contains {count} words"


# All tools in one list — passed to LLM and graph
ALL_TOOLS = [calculator, get_weather, word_count]