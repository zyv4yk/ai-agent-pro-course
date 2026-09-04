"""Тести обвʼязки агента — без жодного справжнього виклику LLM."""
from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage

from f1_oracle import tools
from f1_oracle.agent import RaceForecast, build_agent
from tests.conftest import ScriptedChatModel

FORECAST_ARGS = {
    "race": "Upcoming Grand Prix",
    "circuit": "Test Ring",
    "date": "2026-09-01",
    "predicted_winner": "Fast Driver",
    "confidence": "medium",
    "top5": [
        {
            "driver": "Fast Driver",
            "team": "Alpha",
            "win_probability_pct": 42.0,
            "podium_probability_pct": 80.0,
            "reason": "найкраща форма",
        }
    ],
    "podium": ["Fast Driver", "Second Alpha", "Mid Driver"],
    "key_factors": ["темп Alpha"],
    "risks": ["кваліфікації ще не було"],
    "model_agreement": "згоден з моделлю",
    "data_as_of": "2026-08-25",
}


@pytest.fixture(autouse=True)
def patched_snapshot(monkeypatch, snapshot):
    monkeypatch.setattr(tools, "_snapshot", snapshot)
    monkeypatch.setattr(tools, "_client", object())
    yield
    tools.reset_cache()


def _scripted(*responses: AIMessage) -> ScriptedChatModel:
    return ScriptedChatModel(responses=list(responses), calls=[])


def test_agent_calls_tools_and_returns_structured_forecast():
    model = _scripted(
        AIMessage(
            content="",
            tool_calls=[
                {"name": "get_next_race", "args": {}, "id": "1"},
                {"name": "run_prediction_model", "args": {"top": 5}, "id": "2"},
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "RaceForecast", "args": FORECAST_ARGS, "id": "3"}],
        ),
    )
    agent = build_agent(model=model)
    result = agent.invoke({"messages": [{"role": "user", "content": "Хто переможе?"}]})

    forecast = result["structured_response"]
    assert isinstance(forecast, RaceForecast)
    assert forecast.predicted_winner == "Fast Driver"

    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    names = {m.name for m in tool_messages}
    assert {"get_next_race", "run_prediction_model"} <= names
    # інструмент справді відпрацював, а не був проігнорований
    payload = next(json.loads(m.content) for m in tool_messages if m.name == "get_next_race")
    assert payload["round"] == 7


def test_tracer_middleware_records_every_tool_call():
    from f1_oracle.agent import TraceMiddleware

    tracer = TraceMiddleware()
    model = _scripted(
        AIMessage(content="", tool_calls=[{"name": "get_next_race", "args": {}, "id": "1"}]),
        AIMessage(content="", tool_calls=[{"name": "RaceForecast", "args": FORECAST_ARGS, "id": "2"}]),
    )
    build_agent(model=model, tracer=tracer).invoke(
        {"messages": [{"role": "user", "content": "прогноз"}]}
    )
    assert [c["tool"] for c in tracer.calls] == ["get_next_race"]
    assert tracer.calls[0]["status"] == "ok"


def test_model_call_limit_stops_a_looping_agent(monkeypatch):
    """Модель, що нескінченно кличе інструмент, має впертись у ліміт, а не крутитись вічно."""
    monkeypatch.setattr("f1_oracle.agent.MODEL_CALL_LIMIT", 3)
    looping = AIMessage(content="", tool_calls=[{"name": "get_next_race", "args": {}, "id": "x"}])
    model = _scripted(looping)
    result = build_agent(model=model).invoke(
        {"messages": [{"role": "user", "content": "прогноз"}]}
    )
    assert len([m for m in result["messages"] if m.type == "tool"]) <= 3
    assert result.get("structured_response") is None


def test_system_prompt_forbids_inventing_numbers():
    from f1_oracle.agent import SYSTEM_PROMPT

    assert "run_prediction_model" in SYSTEM_PROMPT
    assert "Жодного відсотка з голови" in SYSTEM_PROMPT
