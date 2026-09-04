"""Прогноз погоди на день гонки (Open-Meteo, без ключа)."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import httpx

from ..config import CACHE_DIR, HTTP_TIMEOUT

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TTL = 3 * 3600


def race_day_forecast(lat: float, lon: float, date: str) -> dict:
    """Повертає прогноз на дату гонки або пояснення, чому його немає.

    Open-Meteo дає прогноз приблизно на 16 днів уперед; далі — порожньо.
    """
    cache_dir = Path(CACHE_DIR) / "weather"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{lat},{lon},{date}".encode()).hexdigest()[:24]
    cached = cache_dir / f"{key}.json"
    if cached.exists() and (time.time() - cached.stat().st_mtime) < TTL:
        return json.loads(cached.read_text())

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date,
        "end_date": date,
        "timezone": "UTC",
        "daily": "temperature_2m_max,precipitation_sum,precipitation_probability_max,wind_speed_10m_max",
    }
    try:
        response = httpx.get(FORECAST_URL, params=params, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        daily = response.json().get("daily", {})
        if not daily.get("time"):
            result = {"available": False, "reason": "дата поза горизонтом прогнозу (>16 днів)"}
        else:
            rain_mm = daily["precipitation_sum"][0] or 0.0
            rain_prob = daily["precipitation_probability_max"][0] or 0
            result = {
                "available": True,
                "date": date,
                "temp_max_c": daily["temperature_2m_max"][0],
                "precipitation_mm": rain_mm,
                "rain_probability_pct": rain_prob,
                "wind_max_kmh": daily["wind_speed_10m_max"][0],
                "wet_race_likely": bool(rain_prob >= 50 or rain_mm >= 1.0),
            }
    except Exception as exc:  # noqa: BLE001 — погода не критична, агент має працювати й без неї
        result = {"available": False, "reason": f"сервіс погоди недоступний: {exc}"}

    cached.write_text(json.dumps(result))
    return result
