def get_weather(city: str) -> str:
    """
    Get the current weather for a given city.

    Args:
        city: Name of the city to get weather for

    Returns:
        Weather description as a string
    """
    # Simulated weather data — in production this would call a real weather API
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