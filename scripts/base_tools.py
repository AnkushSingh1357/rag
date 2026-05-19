import os
import requests

# ✅ FIX: langchain_core.tools, not langchain.tools
from langchain_core.tools import tool
import ollama


@tool
def web_search(query: str) -> str:
    """Perform a live web search for real-time information.

    Args:
        query: Search query string.

    Returns:
        JSON string of top results (max 2).
    """
    response = ollama.web_search(query=query, max_results=2)
    return str(response.results)


@tool
def get_weather(location: str) -> dict:
    """Get current weather for a city using WeatherAPI.

    Args:
        location: City name (e.g. 'New York', 'London').

    Returns:
        Current weather data dict.
    """
    url = (
        f"http://api.weatherapi.com/v1/current.json"
        f"?key={os.getenv('WEATHER_API_KEY')}&q={location}&aqi=no"
    )
    response = requests.get(url=url, timeout=10)
    response.raise_for_status()
    return response.json()
