import httpx
import pytest

from agent.tools.base import ToolError
from agent.tools.weather import get_weather

PARIS_FR = {"name": "Paris", "latitude": 48.85, "longitude": 2.35, "country": "France",
            "country_code": "FR", "admin1": "Île-de-France"}
PARIS_TX = {"name": "Paris", "latitude": 33.66, "longitude": -95.55, "country": "United States",
            "country_code": "US", "admin1": "Texas"}

FORECAST = {
    "current_units": {"temperature_2m": "°C", "apparent_temperature": "°C", "wind_speed_10m": "km/h"},
    "current": {"temperature_2m": 21.5, "apparent_temperature": 20.1,
                "relative_humidity_2m": 55, "wind_speed_10m": 12.3, "weather_code": 2},
}


def make_client(places=None, forecast=FORECAST, seen=None):
    """A fake internet: geocoding returns `places`, the forecast returns `forecast`."""

    def handler(request):
        if seen is not None:
            seen.append(dict(request.url.params))
        if "geocoding" in request.url.host:
            body = {"results": places} if places else {}
            return httpx.Response(200, json=body)
        return httpx.Response(200, json=forecast)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_report_contains_the_important_facts():
    text = get_weather("Paris", client=make_client([PARIS_FR]))
    assert "Paris, Île-de-France, France" in text
    assert "partly cloudy" in text
    assert "21.5°C" in text
    assert "feels like 20.1°C" in text
    assert "humidity 55%" in text
    assert "12.3 km/h" in text


def test_fahrenheit_is_passed_to_the_service():
    seen = []
    get_weather("Paris", units="fahrenheit", client=make_client([PARIS_FR], seen=seen))
    assert seen[-1]["temperature_unit"] == "fahrenheit"


def test_celsius_is_the_default():
    seen = []
    get_weather("Paris", client=make_client([PARIS_FR], seen=seen))
    assert seen[-1]["temperature_unit"] == "celsius"


def test_unknown_place():
    with pytest.raises(ToolError, match="could not find a place called 'Atlantis'"):
        get_weather("Atlantis", client=make_client(places=None))


def test_text_after_the_comma_picks_between_matches():
    text = get_weather("Paris, Texas", client=make_client([PARIS_FR, PARIS_TX]))
    assert "Texas" in text


def test_first_match_is_used_without_a_hint():
    text = get_weather("Paris", client=make_client([PARIS_FR, PARIS_TX]))
    assert "France" in text


def test_only_the_city_part_is_searched():
    seen = []
    get_weather("Paris, Texas", client=make_client([PARIS_TX], seen=seen))
    assert seen[0]["name"] == "Paris"


def test_empty_location():
    with pytest.raises(ToolError, match="city name"):
        get_weather("  ")


def test_missing_current_data():
    with pytest.raises(ToolError, match="no current data"):
        get_weather("Paris", client=make_client([PARIS_FR], forecast={}))


def test_missing_values_in_reply():
    broken = {"current": {"temperature_2m": 20}}
    with pytest.raises(ToolError, match="missing some values"):
        get_weather("Paris", client=make_client([PARIS_FR], forecast=broken))


def test_unknown_weather_code_does_not_crash():
    forecast = {"current_units": {}, "current": dict(FORECAST["current"], weather_code=1234)}
    assert "unknown conditions" in get_weather("Paris", client=make_client([PARIS_FR], forecast=forecast))


def test_timeout_is_reported():
    def handler(request):
        raise httpx.ReadTimeout("slow")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(ToolError, match="too long"):
        get_weather("Paris", client=client)