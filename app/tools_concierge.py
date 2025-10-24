import requests
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from langchain.tools import tool

try:
    from langchain_community.tools import DuckDuckGoSearchResults
except Exception:
    DuckDuckGoSearchResults = None

try:
    from langchain_community.utilities import WikipediaAPIWrapper
except Exception:
    WikipediaAPIWrapper = None

class CityInput(BaseModel):
    city_name: str = Field(..., description="City name")

@tool(description="General web search via DuckDuckGo")
def web_search(query: str) -> str:
    try:
        if DuckDuckGoSearchResults is None:
            return "DuckDuckGo search unavailable: install duckduckgo-search"
        ddg = DuckDuckGoSearchResults(num_results=5)
        return ddg.run(query)
    except Exception as e:
        return f"DuckDuckGo search error: {e}"

@tool(description="Retrieve concise encyclopedic info from Wikipedia")
def wiki_search(query: str) -> str:
    try:
        if WikipediaAPIWrapper is None:
            return "Wikipedia unavailable: install wikipedia"
        wiki = WikipediaAPIWrapper(lang="en")
        return wiki.run(query)
    except Exception as e:
        return f"Wikipedia error: {e}"

@tool(args_schema=CityInput, description="Get the current temperature in a city using Open-Meteo")
def current_weather(city_name: str) -> str:
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city_name, "count": 1},
            timeout=20,
        )
        geo.raise_for_status()
        data = geo.json()
        if not data.get("results"):
            return f"Could not find coordinates for '{city_name}'."
        lat = data["results"][0]["latitude"]
        lon = data["results"][0]["longitude"]
        wx = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "hourly": "temperature_2m", "forecast_days": 1},
            timeout=20,
        )
        wx.raise_for_status()
        res = wx.json()
        now = datetime.now(timezone.utc)
        times = [datetime.fromisoformat(t.replace("Z", "+00:00")).astimezone(timezone.utc)
                 for t in res["hourly"]["time"]]
        temps = res["hourly"]["temperature_2m"]
        idx = min(range(len(times)), key=lambda i: abs(times[i] - now))
        return f"The current temperature in {city_name} is {temps[idx]}°C."
    except Exception as e:
        return f"Weather error for '{city_name}': {e}"

class CurrencyInput(BaseModel):
    amount: float = Field(..., description="Amount to convert")
    from_code: str = Field(..., description="ISO currency code, e.g., CHF")
    to_code: str = Field(..., description="ISO currency code, e.g., EUR")

@tool(args_schema=CurrencyInput, description="Convert currencies via exchangerate.host")
def fx_convert(amount: float, from_code: str, to_code: str) -> str:
    try:
        r = requests.get(
            "https://api.exchangerate.host/convert",
            params={"from": from_code.upper(), "to": to_code.upper(), "amount": amount},
            timeout=20,
        )
        r.raise_for_status()
        j = r.json()
        if j.get("result") is None:
            return f"Conversion not available {from_code}->{to_code}."
        return f"{amount} {from_code.upper()} ≈ {j['result']:.2f} {to_code.upper()} (exchangerate.host)."
    except Exception as e:
        return f"FX error: {e}"