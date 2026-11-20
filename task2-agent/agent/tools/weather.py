"""Current weather for a place, from Open-Meteo (free, no API key)."""

from agent.tools.base import Tool, ToolError, get_json

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo reports the sky as a WMO weather code.
WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "freezing drizzle",
    57: "heavy freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "heavy freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light rain showers",
    81: "rain showers",
    82: "heavy rain showers",
    85: "light snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "thunderstorm with heavy hail",
}


def _find_place(location, client):
    """Look up a place name. 'Paris, Texas' uses the part after the comma to pick between matches."""
    name, _, hint = location.partition(",")
    data = get_json(
        GEOCODING_URL,
        {"name": name.strip(), "count": 5, "language": "en", "format": "json"},
        client,
    )
    places = data.get("results") or []
    if not places:
        raise ToolError(f"I could not find a place called '{location}'.")

    hint = hint.strip().lower()
    if hint:
        for place in places:
            labels = {
                place.get("country", "").lower(),
                place.get("admin1", "").lower(),
                place.get("country_code", "").lower(),
            }
            if hint in labels:
                return place
    return places[0]


def get_weather(location, units="celsius", client=None):
    location = location.strip()
    if not location:
        raise ToolError("Please give a city name.")

    place = _find_place(location, client)
    data = get_json(
        FORECAST_URL,
        {
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,weather_code",
            "temperature_unit": "fahrenheit" if units == "fahrenheit" else "celsius",
            "timezone": "auto",
        },
        client,
    )

    current = data.get("current")
    if not current:
        raise ToolError("The weather service returned no current data for that place.")
    unit_labels = data.get("current_units", {})
    try:
        temperature = f"{current['temperature_2m']}{unit_labels.get('temperature_2m', '')}"
        feels_like = f"{current['apparent_temperature']}{unit_labels.get('apparent_temperature', '')}"
        humidity = current["relative_humidity_2m"]
        wind = f"{current['wind_speed_10m']} {unit_labels.get('wind_speed_10m', '')}".strip()
    except KeyError:
        raise ToolError("The weather service reply was missing some values.")

    sky = WEATHER_CODES.get(current.get("weather_code"), "unknown conditions")
    place_name = ", ".join(
        part for part in (place["name"], place.get("admin1"), place.get("country")) if part
    )
    return (
        f"Current weather in {place_name}: {sky}, {temperature} "
        f"(feels like {feels_like}), humidity {humidity}%, wind {wind}."
    )


TOOL = Tool(
    name="get_weather",
    description=(
        "Get the CURRENT weather for a city: temperature, how it feels, humidity, wind and sky. "
        "Use it whenever the user asks about the weather or temperature somewhere right now. "
        "It does not give forecasts for future days."
    ),
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name, optionally with a country or state, for example 'Ahmedabad' or 'Paris, Texas'.",
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit. Defaults to celsius.",
            },
        },
        "required": ["location"],
    },
    func=get_weather,
)