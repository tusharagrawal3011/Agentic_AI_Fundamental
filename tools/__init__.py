from .calculator import calculator
from .weather import get_weather
from .string_tools import word_count, reverse_string

# All available tools in one list
# Add any new tool here — dispatcher picks it up automatically
ALL_TOOLS = [calculator, get_weather, word_count, reverse_string]