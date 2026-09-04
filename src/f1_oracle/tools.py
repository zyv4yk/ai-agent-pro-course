"""Інструменти агента.

Кожен інструмент — звичайна функція з тайп-хінтами і докстрінгом:
саме з них LangChain будує схему, яку бачить модель.
Уся арифметика лишається тут; LLM не рахує ймовірності, а інтерпретує їх.
"""
from __future__ import annotations

import json
from typing import Any

from langchain.tools import tool

from .config import SIMULATIONS
from .data.jolpica import JolpicaClient
from .data.weather import race_day_forecast
from .evaluation import backtest
from .predictor import predict_upcoming
from .season import Snapshot, load_snapshot

_client: JolpicaClient | None = None
_snapshot: Snapshot | None = None


def get_snapshot(refresh: bool = False) -> Snapshot:
    """Знімок сезону тягнеться один раз на процес (усередині — ще й дисковий кеш)."""
    global _client, _snapshot
    if _client is None:
        _client = JolpicaClient()
    if _snapshot is None or refresh:
        _snapshot = load_snapshot(_client)
    return _snapshot


def get_client() -> JolpicaClient:
    get_snapshot()
    assert _client is not None
    return _client


def reset_cache() -> None:
    """Скидає кеш процесу (потрібно тестам і евалам)."""
    global _client, _snapshot
    _client = None
    _snapshot = None


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


@tool
def get_next_race() -> str:
    """Найближча гонка календаря Формули-1: назва, траса, дата, скільки днів лишилось.

    Викликай першим — усі інші інструменти працюють саме з цією гонкою.
    """
    snapshot = get_snapshot()
    if snapshot.upcoming is None:
        return _json({"error": "у календарі сезону більше немає майбутніх гонок"})
    race = snapshot.upcoming
    from datetime import datetime, timezone

    days = (race.starts_at - datetime.now(timezone.utc)).days
    return _json(
        {
            "season": snapshot.season,
            "round": race.round,
            "race": race.name,
            "circuit": race.circuit_name,
            "location": f"{race.locality}, {race.country}",
            "date": race.date,
            "time_utc": race.time,
            "days_until_race": days,
            "rounds_completed": len(snapshot.completed),
            "qualifying_results_available": bool(snapshot.upcoming_qualifying()),
            "data_as_of": snapshot.as_of,
        }
    )


@tool
def get_championship_standings(top: int = 10) -> str:
    """Поточна таблиця чемпіонату: пілоти та конструктори.

    Args:
        top: скільки рядків повернути в кожній таблиці
    """
    snapshot = get_snapshot()
    drivers = [
        {
            "position": int(row["position"]),
            "driver": f"{row['Driver']['givenName']} {row['Driver']['familyName']}",
            "driver_id": row["Driver"]["driverId"],
            "team": row["Constructors"][-1]["name"],
            "points": float(row["points"]),
            "wins": int(row["wins"]),
        }
        for row in snapshot.driver_standings[:top]
    ]
    constructors = [
        {
            "position": int(row["position"]),
            "team": row["Constructor"]["name"],
            "points": float(row["points"]),
            "wins": int(row["wins"]),
        }
        for row in snapshot.constructor_standings[:top]
    ]
    return _json({"season": snapshot.season, "drivers": drivers, "constructors": constructors})


@tool
def get_recent_form(races: int = 4) -> str:
    """Результати останніх гонок сезону: хто виграв, хто на подіумі, хто зійшов.

    Args:
        races: скільки останніх гонок показати (1-8)
    """
    snapshot = get_snapshot()
    races = max(1, min(races, 8))
    out = []
    for race in snapshot.completed[-races:]:
        out.append(
            {
                "round": race.round,
                "race": race.name,
                "date": race.date,
                "top5": [
                    {
                        "position": r.position,
                        "driver": r.name,
                        "team": r.constructor_name,
                        "grid": r.grid,
                    }
                    for r in race.results[:5]
                ],
                "retirements": [
                    {"driver": r.name, "team": r.constructor_name, "status": r.status}
                    for r in race.results
                    if not r.classified
                ],
            }
        )
    return _json({"season": snapshot.season, "races": out})


@tool
def get_circuit_history() -> str:
    """Історія найближчої траси: переможці й подіуми попередніх сезонів.

    Показує, кому ця конфігурація траси історично підходить.
    """
    snapshot = get_snapshot()
    if snapshot.upcoming is None:
        return _json({"error": "немає майбутньої гонки"})
    history = []
    for year in sorted(snapshot.circuit_history, reverse=True):
        race = snapshot.circuit_history[year]
        history.append(
            {
                "season": year,
                "race": race.name,
                "podium": [
                    {"position": r.position, "driver": r.name, "team": r.constructor_name}
                    for r in race.results[:3]
                ],
                "pole": next((r.name for r in race.results if r.grid == 1), None),
            }
        )
    return _json(
        {
            "circuit": snapshot.upcoming.circuit_name,
            "circuit_id": snapshot.upcoming.circuit_id,
            "history": history,
        }
    )


@tool
def get_qualifying_grid() -> str:
    """Стартова сітка найближчої гонки, якщо кваліфікація вже відбулася.

    Якщо кваліфікації ще не було — повертає ознаку, що прогноз робиться до неї.
    """
    snapshot = get_snapshot()
    grid = snapshot.upcoming_qualifying()
    if not grid:
        return _json(
            {
                "available": False,
                "note": "кваліфікація ще не відбулася — модель спирається на темп у попередніх кваліфікаціях",
            }
        )
    by_position = sorted(grid.items(), key=lambda kv: kv[1])
    return _json(
        {
            "available": True,
            "race": snapshot.upcoming.name if snapshot.upcoming else None,
            "grid": [
                {"position": pos, "driver": snapshot.name_of(driver_id)}
                for driver_id, pos in by_position
            ],
        }
    )


@tool
def get_race_weather() -> str:
    """Прогноз погоди на день найближчої гонки: температура, опади, вітер.

    Дощ радикально змінює ієрархію — це найважливіший позамодельний фактор.
    """
    snapshot = get_snapshot()
    if snapshot.upcoming is None:
        return _json({"error": "немає майбутньої гонки"})
    race = snapshot.upcoming
    forecast = race_day_forecast(race.lat, race.lon, race.date)
    return _json({"race": race.name, "date": race.date, **forecast})


@tool
def run_prediction_model(top: int = 8, simulations: int = SIMULATIONS) -> str:
    """Головний інструмент: детермінована модель ймовірностей перемоги.

    Рахує чотири нормалізовані фічі (форма, темп сезону, кваліфікаційний темп,
    темп команди, історія траси), зводить їх у "силу" пілота і розігрує
    Монте-Карло порядок фінішу. Повертає win/podium/points у відсотках.
    Ці цифри не можна вигадувати самому — бери тільки звідси.

    Args:
        top: скільки пілотів повернути
        simulations: кількість симуляцій Монте-Карло
    """
    snapshot = get_snapshot()
    prediction = predict_upcoming(snapshot, simulations=simulations)
    if prediction is None:
        return _json({"error": "недостатньо даних: немає майбутньої гонки або проведених гонок"})
    return _json(prediction.as_dict(top=top))


@tool
def get_model_backtest(start_round: int = 4) -> str:
    """Якість моделі на вже проведених гонках цього сезону.

    Показує top-1 accuracy, влучність по подіуму, Brier і log-loss, а також
    наївні бейзлайни. Використовуй, щоб чесно калібрувати впевненість прогнозу.

    Args:
        start_round: з якого раунду починати бектест
    """
    snapshot = get_snapshot()
    report = backtest(snapshot, get_client(), start_round=start_round)
    return _json(report.summary())


ALL_TOOLS = [
    get_next_race,
    get_championship_standings,
    get_recent_form,
    get_circuit_history,
    get_qualifying_grid,
    get_race_weather,
    run_prediction_model,
    get_model_backtest,
]
