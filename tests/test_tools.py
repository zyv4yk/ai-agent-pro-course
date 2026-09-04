"""Тести інструментів: агент має отримувати валідний JSON, а не сюрпризи."""
from __future__ import annotations

import json

import pytest

from f1_oracle import tools


@pytest.fixture(autouse=True)
def patched_snapshot(monkeypatch, snapshot):
    """Підміняємо знімок сезону — жодного мережевого запиту в тестах."""
    monkeypatch.setattr(tools, "_snapshot", snapshot)
    monkeypatch.setattr(tools, "_client", object())
    yield
    tools.reset_cache()


def test_get_next_race_returns_upcoming_round():
    payload = json.loads(tools.get_next_race.invoke({}))
    assert payload["round"] == 7
    assert payload["race"] == "Upcoming Grand Prix"
    assert payload["qualifying_results_available"] is False


def test_run_prediction_model_returns_sorted_probabilities():
    payload = json.loads(tools.run_prediction_model.invoke({"top": 3, "simulations": 2000}))
    drivers = payload["drivers"]
    assert len(drivers) == 3
    assert drivers == sorted(drivers, key=lambda d: d["win_pct"], reverse=True)
    assert payload["used_qualifying_results"] is False
    assert payload["monte_carlo_stderr_pp"] > 0


def test_recent_form_lists_retirements():
    payload = json.loads(tools.get_recent_form.invoke({"races": 2}))
    assert len(payload["races"]) == 2
    assert "retirements" in payload["races"][0]


def test_circuit_history_returns_podiums():
    payload = json.loads(tools.get_circuit_history.invoke({}))
    assert payload["circuit"] == "Test Ring"
    assert len(payload["history"][0]["podium"]) == 3


def test_qualifying_grid_reports_absence_explicitly():
    payload = json.loads(tools.get_qualifying_grid.invoke({}))
    assert payload["available"] is False
    assert "кваліфікація" in payload["note"]


def test_every_tool_has_description_for_the_model():
    """Докстрінг — це не коментар для колеги, а частина промпту."""
    for tool in tools.ALL_TOOLS:
        assert tool.description and len(tool.description) > 30, tool.name
