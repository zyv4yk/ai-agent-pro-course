"""Синтетичний сезон і фейкова LLM — щоб тести не ходили в мережу і не палили токени."""
from __future__ import annotations

from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from f1_oracle.season import DriverResult, Entry, Race, Snapshot

DRIVERS = [
    # driver_id, імʼя, код, команда
    ("fast", "Fast Driver", "FAS", "alpha", "Alpha"),
    ("fast2", "Second Alpha", "SEC", "alpha", "Alpha"),
    ("mid", "Mid Driver", "MID", "beta", "Beta"),
    ("mid2", "Second Beta", "SB2", "beta", "Beta"),
    ("slow", "Slow Driver", "SLO", "gamma", "Gamma"),
    ("slow2", "Second Gamma", "SG2", "gamma", "Gamma"),
]
POINTS = {1: 25.0, 2: 18.0, 3: 15.0, 4: 12.0, 5: 10.0, 6: 8.0}


def make_race(
    round_no: int,
    order: list[str],
    circuit_id: str = "testring",
    dnf: set[str] | None = None,
    season: int = 2026,
    grid: list[str] | None = None,
) -> Race:
    """Гонка з заданим порядком фінішу (order — список driver_id від першого місця)."""
    dnf = dnf or set()
    grid = grid or order
    by_id = {d[0]: d for d in DRIVERS}
    results = []
    for position, driver_id in enumerate(order, start=1):
        _, name, code, constructor_id, constructor_name = by_id[driver_id]
        retired = driver_id in dnf
        results.append(
            DriverResult(
                driver_id=driver_id,
                name=name,
                code=code,
                constructor_id=constructor_id,
                constructor_name=constructor_name,
                position=position,
                position_text="R" if retired else str(position),
                grid=grid.index(driver_id) + 1,
                points=0.0 if retired else POINTS.get(position, 0.0),
                status="Engine" if retired else ("Finished" if position == 1 else "+1 Lap"),
            )
        )
    return Race(
        season=season,
        round=round_no,
        name=f"Test Grand Prix {round_no}",
        date=f"2026-0{min(round_no, 9)}-01",
        time="13:00:00Z",
        circuit_id=circuit_id,
        circuit_name="Test Ring",
        locality="Testville",
        country="Testland",
        lat=50.0,
        lon=30.0,
        results=tuple(results),
    )


@pytest.fixture
def entries() -> list[Entry]:
    return [Entry(d[0], d[1], d[2], d[3], d[4]) for d in DRIVERS]


@pytest.fixture
def races() -> list[Race]:
    """Шість гонок, де alpha стабільно швидша за beta, а gamma — аутсайдер."""
    orders = [
        ["fast", "fast2", "mid", "mid2", "slow", "slow2"],
        ["fast2", "fast", "mid", "slow", "mid2", "slow2"],
        ["fast", "mid", "fast2", "mid2", "slow2", "slow"],
        ["fast", "fast2", "mid2", "mid", "slow", "slow2"],
        ["fast2", "fast", "mid", "mid2", "slow2", "slow"],
        ["fast", "mid", "fast2", "mid2", "slow", "slow2"],
    ]
    return [make_race(i + 1, order) for i, order in enumerate(orders)]


@pytest.fixture
def upcoming() -> Race:
    return Race(
        season=2026,
        round=7,
        name="Upcoming Grand Prix",
        date="2026-09-01",
        time="13:00:00Z",
        circuit_id="testring",
        circuit_name="Test Ring",
        locality="Testville",
        country="Testland",
        lat=50.0,
        lon=30.0,
        results=(),
    )


@pytest.fixture
def snapshot(races, upcoming) -> Snapshot:
    return Snapshot(
        season=2026,
        as_of="2026-08-25T00:00:00+00:00",
        completed=races,
        upcoming=upcoming,
        qualifying={},
        circuit_history={2025: make_race(1, ["fast", "mid", "fast2", "mid2", "slow", "slow2"], season=2025)},
        driver_standings=[],
        constructor_standings=[],
    )


class ScriptedChatModel(BaseChatModel):
    """Фейкова LLM: віддає заздалегідь записані відповіді по черзі.

    Потрібна, щоб перевірити обвʼязку агента (виклик інструментів, структурований
    вихід, межі циклу) без жодного запиту до справжньої моделі.
    """

    responses: list[AIMessage]
    calls: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":  # noqa: ARG002
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: CallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> ChatResult:
        self.calls.append(messages)
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return ChatResult(generations=[ChatGeneration(message=self.responses[index])])
