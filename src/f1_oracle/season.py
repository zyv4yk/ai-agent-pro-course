"""Доменна модель сезону: сирі відповіді Jolpica → типізований знімок даних."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as Date
from datetime import datetime, timezone

from .config import TRACK_HISTORY_YEARS
from .data.jolpica import JolpicaClient

FINISHED_STATUSES = {"Finished"}


def _is_classified(status: str, position_text: str) -> bool:
    """Гонщик доїхав: або 'Finished', або відставання в колах ('+1 Lap')."""
    return position_text.isdigit() and (status in FINISHED_STATUSES or status.startswith("+"))


@dataclass(frozen=True)
class DriverResult:
    driver_id: str
    name: str
    code: str
    constructor_id: str
    constructor_name: str
    position: int
    position_text: str
    grid: int
    points: float
    status: str

    @property
    def classified(self) -> bool:
        return _is_classified(self.status, self.position_text)


@dataclass(frozen=True)
class Race:
    season: int
    round: int
    name: str
    date: str
    time: str | None
    circuit_id: str
    circuit_name: str
    locality: str
    country: str
    lat: float
    lon: float
    results: tuple[DriverResult, ...] = ()

    @property
    def starts_at(self) -> datetime:
        raw = f"{self.date}T{self.time or '12:00:00Z'}".replace("Z", "+00:00")
        return datetime.fromisoformat(raw)

    @property
    def winner(self) -> DriverResult | None:
        return next((r for r in self.results if r.position == 1), None)

    @property
    def podium(self) -> list[str]:
        return [r.driver_id for r in sorted(self.results, key=lambda r: r.position)[:3]]


@dataclass(frozen=True)
class Entry:
    driver_id: str
    name: str
    code: str
    constructor_id: str
    constructor_name: str


@dataclass
class Snapshot:
    """Усе, що агент знає про сезон на конкретний момент часу."""

    season: int
    as_of: str
    completed: list[Race]
    upcoming: Race | None
    qualifying: dict[int, dict[str, int]] = field(default_factory=dict)
    circuit_history: dict[int, Race] = field(default_factory=dict)
    driver_standings: list[dict] = field(default_factory=list)
    constructor_standings: list[dict] = field(default_factory=list)

    @property
    def entries(self) -> list[Entry]:
        """Актуальний склад пілотів — за останньою проведеною гонкою."""
        seen: dict[str, Entry] = {}
        for race in reversed(self.completed):
            for result in race.results:
                seen.setdefault(
                    result.driver_id,
                    Entry(
                        result.driver_id,
                        result.name,
                        result.code,
                        result.constructor_id,
                        result.constructor_name,
                    ),
                )
            if seen:
                break
        return list(seen.values())

    def name_of(self, driver_id: str) -> str:
        for entry in self.entries:
            if entry.driver_id == driver_id:
                return entry.name
        return driver_id

    def upcoming_qualifying(self) -> dict[str, int]:
        if not self.upcoming:
            return {}
        return self.qualifying.get(self.upcoming.round, {})


def _parse_race(raw: dict, with_results: bool = True) -> Race:
    circuit = raw["Circuit"]
    location = circuit["Location"]
    results: list[DriverResult] = []
    if with_results:
        for item in raw.get("Results", []):
            results.append(
                DriverResult(
                    driver_id=item["Driver"]["driverId"],
                    name=f"{item['Driver']['givenName']} {item['Driver']['familyName']}",
                    code=item["Driver"].get("code", ""),
                    constructor_id=item["Constructor"]["constructorId"],
                    constructor_name=item["Constructor"]["name"],
                    position=int(item["position"]),
                    position_text=item["positionText"],
                    grid=int(item.get("grid", 0)),
                    points=float(item.get("points", 0)),
                    status=item.get("status", ""),
                )
            )
    return Race(
        season=int(raw["season"]),
        round=int(raw["round"]),
        name=raw["raceName"],
        date=raw["date"],
        time=raw.get("time"),
        circuit_id=circuit["circuitId"],
        circuit_name=circuit["circuitName"],
        locality=location["locality"],
        country=location["country"],
        lat=float(location["lat"]),
        lon=float(location["long"]),
        results=tuple(sorted(results, key=lambda r: r.position)),
    )


def load_snapshot(
    client: JolpicaClient | None = None,
    now: datetime | None = None,
    season: int | None = None,
) -> Snapshot:
    """Збирає знімок сезону: проведені гонки, найближча гонка, кваліфікації, історія траси."""
    client = client or JolpicaClient()
    now = now or datetime.now(timezone.utc)
    season = season or now.year

    schedule = client.schedule(season)
    if not schedule:  # сезон ще не опублікований — беремо попередній
        season -= 1
        schedule = client.schedule(season)

    completed = [_parse_race(r) for r in client.season_results(season, now.year)]
    completed_rounds = {race.round for race in completed}

    upcoming = None
    for raw in schedule:
        race = _parse_race(raw, with_results=False)
        if race.round in completed_rounds:
            continue
        if race.starts_at >= now:
            upcoming = race
            break

    qualifying: dict[int, dict[str, int]] = {}
    for raw in client.season_qualifying(season, now.year):
        grid = {
            item["Driver"]["driverId"]: int(item["position"])
            for item in raw.get("QualifyingResults", [])
        }
        if grid:
            qualifying[int(raw["round"])] = grid

    circuit_history: dict[int, Race] = {}
    if upcoming:
        for year in range(season - 1, season - 1 - TRACK_HISTORY_YEARS, -1):
            try:
                races = client.circuit_results(year, upcoming.circuit_id, now.year)
            except Exception:  # noqa: BLE001 — історія траси не критична
                continue
            if races:
                circuit_history[year] = _parse_race(races[0])

    return Snapshot(
        season=season,
        as_of=now.isoformat(timespec="seconds"),
        completed=completed,
        upcoming=upcoming,
        qualifying=qualifying,
        circuit_history=circuit_history,
        driver_standings=client.driver_standings(season),
        constructor_standings=client.constructor_standings(season),
    )


def today_utc() -> Date:
    return datetime.now(timezone.utc).date()
