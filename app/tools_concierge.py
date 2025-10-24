import os
import requests
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

try:
    from langchain_community.tools import DuckDuckGoSearchResults
except Exception:
    DuckDuckGoSearchResults = None

try:
    from langchain_community.utilities import WikipediaAPIWrapper
except Exception:
    WikipediaAPIWrapper = None

USER_AGENT = os.getenv("USER_AGENT", "hotel-concierge-bot/1.0")
_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT})
_DEFAULT_TIMEOUT = 20

def _http_get(url: str, **kwargs) -> requests.Response:
    timeout = kwargs.pop("timeout", _DEFAULT_TIMEOUT)
    r = _session.get(url, timeout=timeout, **kwargs)
    r.raise_for_status()
    return r

def _web_search(query: str) -> str:
    """Allgemeine Websuche über DuckDuckGo. Liefert eine kurze Ergebnisliste als Text."""
    try:
        if DuckDuckGoSearchResults is None:
            return "Suche nicht verfügbar: Bitte 'duckduckgo-search' installieren."
        ddg = DuckDuckGoSearchResults(num_results=5)
        return ddg.run(query)
    except Exception as e:
        return f"Fehler bei der Websuche: {e}"

web_search = StructuredTool.from_function(
    func=_web_search,
    name="web_search",
    description="Allgemeine Websuche (DuckDuckGo). Input ist eine Suchanfrage (String).",
)

def _wiki_search(query: str) -> str:
    """Kurzinfo aus Wikipedia. Versucht zuerst Deutsch, fällt sonst auf Englisch zurück."""
    try:
        if WikipediaAPIWrapper is None:
            return "Wikipedia nicht verfügbar: Bitte 'wikipedia' installieren."
        try:
            wiki_de = WikipediaAPIWrapper(lang="de")
            ans = wiki_de.run(query)
            if isinstance(ans, str) and ans.strip():
                return ans
        except Exception:
            pass
        wiki_en = WikipediaAPIWrapper(lang="en")
        return wiki_en.run(query)
    except Exception as e:
        return f"Fehler bei Wikipedia: {e}"

wiki_search = StructuredTool.from_function(
    func=_wiki_search,
    name="wiki_search",
    description="Kurzinfo aus Wikipedia. Input ist eine Frage/Begriff (String).",
)

class CityInput(BaseModel):
    city_name: str = Field(..., description="City name, e.g., Basel")

def _current_weather(city_name: str) -> str:
    """Aktuelle Temperatur in einer Stadt über Open-Meteo (current_weather, timezone=auto)."""
    try:
        geo = _http_get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city_name, "count": 1},
        ).json()
        if not geo.get("results"):
            return f"Konnte keine Koordinaten für {city_name} finden."
        lat = geo["results"][0]["latitude"]
        lon = geo["results"][0]["longitude"]
        wx = _http_get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current_weather": True,
                "timezone": "auto",
            },
        ).json()
        cw = wx.get("current_weather") or {}
        temp = cw.get("temperature")
        if temp is None:
            return f"Leider keine aktuelle Temperatur für {city_name} verfügbar."
        return f"Aktuelle Temperatur in {city_name}: {float(temp):.1f} °C."
    except Exception as e:
        return f"Wetter-Fehler für {city_name}: {e}"

current_weather = StructuredTool.from_function(
    func=_current_weather,
    name="current_weather",
    args_schema=CityInput,
    description="Aktuelle Temperatur in einer Stadt (Open-Meteo). Input: {city_name: str}.",
)

class CurrencyInput(BaseModel):
    amount: float = Field(..., description="Betrag")
    from_code: str = Field(..., description="ISO-Währungscode, z. B. CHF")
    to_code: str = Field(..., description="ISO-Währungscode, z. B. EUR")

def _fx_exchangerate_host(amount: float, from_code: str, to_code: str):
    j = _http_get(
        "https://api.exchangerate.host/convert",
        params={"from": from_code.upper(), "to": to_code.upper(), "amount": amount},
    ).json()
    return j.get("result")

def _fx_frankfurter(amount: float, from_code: str, to_code: str):
    j = _http_get(
        "https://api.frankfurter.app/latest",
        params={"amount": amount, "from": from_code.upper(), "to": to_code.upper()},
    ).json()
    rates = j.get("rates") or {}
    return rates.get(to_code.upper())

def _fx_open_erapi(amount: float, from_code: str, to_code: str):
    base = from_code.upper()
    j = _http_get(f"https://open.er-api.com/v6/latest/{base}").json()
    rate = (j.get("rates") or {}).get(to_code.upper())
    if rate is None:
        return None
    return float(amount) * float(rate)

def _fx_convert(amount: float, from_code: str, to_code: str) -> str:
    """Währungsumrechnung mit Fallback (exchangerate.host → frankfurter.app → open.er-api.com)."""
    for fn in (_fx_exchangerate_host, _fx_frankfurter, _fx_open_erapi):
        try:
            result = fn(amount, from_code, to_code)
            if result is not None:
                return f"{amount} {from_code.upper()} ≈ {float(result):.2f} {to_code.upper()}."
        except Exception:
            continue
    return f"Keine Umrechnung möglich {from_code}->{to_code}."

fx_convert = StructuredTool.from_function(
    func=_fx_convert,
    name="fx_convert",
    args_schema=CurrencyInput,
    description="Währungsumrechnung. Input: {amount: float, from_code: str, to_code: str}.",
)