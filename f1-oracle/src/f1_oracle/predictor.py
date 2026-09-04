"""Склеює знімок сезону з детермінованою моделлю."""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import BETA, SEED, SIMULATIONS, TRACK_HISTORY_YEARS
from .data.jolpica import JolpicaClient
from .scoring import DriverProbability, Weights, mc_stderr_pp, predict
from .season import Race, Snapshot


@dataclass
class RacePrediction:
    season: int
    round: int
    race: str
    circuit: str
    date: str
    country: str
    locality: str
    used_qualifying: bool
    races_used: int
    simulations: int
    track_history_years: list[int]
    data_as_of: str
    drivers: list[DriverProbability] = field(default_factory=list)

    def as_dict(self, top: int | None = 10) -> dict:
        drivers = self.drivers[:top] if top else self.drivers
        return {
            "race": self.race,
            "round": self.round,
            "season": self.season,
            "circuit": self.circuit,
            "location": f"{self.locality}, {self.country}",
            "date": self.date,
            "used_qualifying_results": self.used_qualifying,
            "monte_carlo_stderr_pp": (
                mc_stderr_pp(self.drivers[0].win_pct, self.simulations) if self.drivers else 0.0
            ),
            "races_in_form_window": self.races_used,
            "track_history_seasons": self.track_history_years,
            "data_as_of": self.data_as_of,
            "drivers": [d.as_dict() for d in drivers],
        }


def predict_upcoming(
    snapshot: Snapshot, simulations: int = SIMULATIONS, seed: int = SEED
) -> RacePrediction | None:
    """Прогноз на найближчу гонку календаря."""
    if snapshot.upcoming is None or not snapshot.completed:
        return None
    race = snapshot.upcoming
    grid = snapshot.upcoming_qualifying()
    circuit_races = list(snapshot.circuit_history.values())
    drivers = predict(
        snapshot.completed, circuit_races, snapshot.entries, grid, simulations, seed
    )
    return RacePrediction(
        season=snapshot.season,
        round=race.round,
        race=race.name,
        circuit=race.circuit_name,
        date=race.date,
        country=race.country,
        locality=race.locality,
        used_qualifying=bool(grid) and any(d.grid for d in drivers),
        races_used=min(len(snapshot.completed), 5),
        simulations=simulations,
        track_history_years=sorted(snapshot.circuit_history, reverse=True),
        data_as_of=snapshot.as_of,
        drivers=drivers,
    )


def predict_past_round(
    snapshot: Snapshot,
    target_round: int,
    client: JolpicaClient | None = None,
    use_qualifying: bool = True,
    simulations: int = SIMULATIONS,
    seed: int = SEED,
    beta: float = BETA,
    weights: Weights | None = None,
) -> tuple[RacePrediction | None, Race | None]:
    """Ретроспективний прогноз: рахуємо раунд target_round, знаючи лише попередні.

    Використовується бектестом — це та сама функція, що й для майбутньої гонки,
    просто з обрізаною історією.
    """
    target = next((r for r in snapshot.completed if r.round == target_round), None)
    if target is None:
        return None, None
    history = [r for r in snapshot.completed if r.round < target_round]
    if not history:
        return None, target

    entries = [
        e
        for e in _entries_from(history[-1])
        if any(r.driver_id == e.driver_id for r in target.results)
    ] or _entries_from(target)

    circuit_races: list[Race] = []
    if client is not None:
        for year in range(snapshot.season - 1, snapshot.season - 1 - TRACK_HISTORY_YEARS, -1):
            try:
                races = client.circuit_results(year, target.circuit_id, snapshot.season)
            except Exception:  # noqa: BLE001
                continue
            if races:
                from .season import _parse_race  # локальний імпорт: уникаємо циклу

                circuit_races.append(_parse_race(races[0]))

    grid = snapshot.qualifying.get(target_round, {}) if use_qualifying else {}
    drivers = predict(history, circuit_races, entries, grid, simulations, seed, beta, weights)
    prediction = RacePrediction(
        season=snapshot.season,
        round=target.round,
        race=target.name,
        circuit=target.circuit_name,
        date=target.date,
        country=target.country,
        locality=target.locality,
        used_qualifying=bool(grid),
        races_used=min(len(history), 5),
        simulations=simulations,
        track_history_years=[r.season for r in circuit_races],
        data_as_of=f"after round {target_round - 1}",
        drivers=drivers,
    )
    return prediction, target


def _entries_from(race: Race):
    from .season import Entry

    return [
        Entry(r.driver_id, r.name, r.code, r.constructor_id, r.constructor_name)
        for r in race.results
    ]
