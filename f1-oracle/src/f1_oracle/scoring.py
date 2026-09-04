"""Детермінована модель прогнозу: фічі → сила пілота → ймовірності.

Математику навмисно не віддано LLM. Модель рахує ймовірності сама,
а агент лише інтерпретує їх і додає контекст, якого немає в цифрах.

Схема:
  1. чотири фічі на пілота (форма, кваліфікаційний темп, темп команди, історія траси);
  2. z-нормалізація кожної фічі по полю → зважена сума = "сила";
  3. Монте-Карло Плакетта–Льюса: log(w) = BETA * сила, плюс шум Гумбеля,
     плюс шанс зійти з дистанції → ймовірності перемоги / подіуму / очок.
"""
from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from statistics import fmean, pstdev

from .config import BETA, FORM_WINDOW, RECENCY_DECAY, SEED, SIMULATIONS
from .season import Entry, Race

FIELD_SIZE = 21  # рейтинг = FIELD_SIZE - позиція, тож P1 = 20 балів


@dataclass(frozen=True)
class Weights:
    form: float
    season: float
    quali: float
    team: float
    track: float

    def normalized(self) -> "Weights":
        total = self.form + self.season + self.quali + self.team + self.track
        return Weights(*(getattr(self, f) / total for f in ("form", "season", "quali", "team", "track")))


# Коли сітка вже відома, кваліфікація важить більше за все інше.
WEIGHTS_PRE_QUALI = Weights(form=0.26, season=0.24, quali=0.18, team=0.20, track=0.12)
WEIGHTS_WITH_GRID = Weights(form=0.20, season=0.18, quali=0.32, team=0.18, track=0.12)


@dataclass
class DriverFeatures:
    driver_id: str
    name: str
    code: str
    constructor_id: str
    constructor_name: str
    form: float | None = None
    season: float | None = None
    quali: float | None = None
    team: float | None = None
    track: float | None = None
    reliability: float = 0.9
    grid: int | None = None
    races_counted: int = 0


@dataclass
class DriverProbability:
    driver_id: str
    name: str
    code: str
    team: str
    win_pct: float
    podium_pct: float
    points_pct: float
    strength: float
    grid: int | None
    features: dict[str, float | None] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def _rating(position: int, classified: bool) -> float:
    """Рейтинг за фініш: перемога = 20, схід = 0."""
    if not classified:
        return 0.0
    return float(max(FIELD_SIZE - position, 1))


def _grid_rating(grid: int) -> float:
    """Рейтинг за стартову позицію (0 = старт із піт-лейну)."""
    if grid <= 0:
        return 1.0
    return float(max(FIELD_SIZE - grid, 1))


def _weighted_mean(values: Sequence[tuple[float, float]]) -> float | None:
    """values = [(значення, вага)]."""
    total_weight = sum(w for _, w in values)
    if total_weight == 0:
        return None
    return sum(v * w for v, w in values) / total_weight


def _zscores(values: Mapping[str, float | None]) -> dict[str, float]:
    """z-нормалізація; тим, у кого фічі немає, ставимо середнє поля (z = 0)."""
    present = [v for v in values.values() if v is not None]
    if len(present) < 2:
        return {k: 0.0 for k in values}
    mean = fmean(present)
    spread = pstdev(present) or 1.0
    return {k: 0.0 if v is None else (v - mean) / spread for k, v in values.items()}


def build_features(
    races: Sequence[Race],
    circuit_races: Sequence[Race],
    entries: Sequence[Entry],
    actual_grid: Mapping[str, int] | None = None,
    form_window: int = FORM_WINDOW,
) -> list[DriverFeatures]:
    """Рахує сирі (ще не нормалізовані) фічі по кожному пілоту зі стартового списку."""
    recent = list(races)[-form_window:]
    # ваги: остання гонка найважливіша
    decay = {race.round: RECENCY_DECAY ** i for i, race in enumerate(reversed(recent))}

    team_race_ratings: dict[str, list[tuple[float, float]]] = {}
    for race in recent:
        by_team: dict[str, list[float]] = {}
        for result in race.results:
            by_team.setdefault(result.constructor_id, []).append(
                _rating(result.position, result.classified)
            )
        for constructor_id, ratings in by_team.items():
            team_race_ratings.setdefault(constructor_id, []).append(
                (fmean(ratings), decay[race.round])
            )

    features: list[DriverFeatures] = []
    for entry in entries:
        form_points: list[tuple[float, float]] = []
        quali_points: list[tuple[float, float]] = []
        for race in recent:
            result = next((r for r in race.results if r.driver_id == entry.driver_id), None)
            if result is None:
                continue
            form_points.append((_rating(result.position, result.classified), decay[race.round]))
            quali_points.append((_grid_rating(result.grid), decay[race.round]))

        season_points: list[tuple[float, float]] = []
        starts = classified = 0
        for race in races:
            result = next((r for r in race.results if r.driver_id == entry.driver_id), None)
            if result is None:
                continue
            starts += 1
            classified += int(result.classified)
            season_points.append((_rating(result.position, result.classified), 1.0))
        reliability = (classified + 3) / (starts + 3) if starts else 0.9

        driver_track = [
            _rating(r.position, r.classified)
            for race in circuit_races
            for r in race.results
            if r.driver_id == entry.driver_id
        ]
        team_track = [
            _rating(r.position, r.classified)
            for race in circuit_races
            for r in race.results
            if r.constructor_id == entry.constructor_id
        ]
        # історія пілота на трасі; якщо її немає — історія команди; далі None → середнє поля
        track = fmean(driver_track) if driver_track else (fmean(team_track) if team_track else None)

        features.append(
            DriverFeatures(
                driver_id=entry.driver_id,
                name=entry.name,
                code=entry.code,
                constructor_id=entry.constructor_id,
                constructor_name=entry.constructor_name,
                form=_weighted_mean(form_points),
                season=_weighted_mean(season_points),
                quali=_weighted_mean(quali_points),
                team=_weighted_mean(team_race_ratings.get(entry.constructor_id, [])),
                track=track,
                reliability=min(max(reliability, 0.5), 0.99),
                grid=(actual_grid or {}).get(entry.driver_id),
                races_counted=len(form_points),
            )
        )
    return features


def strengths(
    features: Sequence[DriverFeatures],
    use_grid: bool,
    weights: Weights | None = None,
) -> dict[str, float]:
    """Зважена сума z-нормалізованих фіч."""
    weights = (weights or (WEIGHTS_WITH_GRID if use_grid else WEIGHTS_PRE_QUALI)).normalized()
    z_form = _zscores({f.driver_id: f.form for f in features})
    z_season = _zscores({f.driver_id: f.season for f in features})
    z_team = _zscores({f.driver_id: f.team for f in features})
    z_track = _zscores({f.driver_id: f.track for f in features})
    if use_grid:
        z_quali = _zscores(
            {f.driver_id: (_grid_rating(f.grid) if f.grid else None) for f in features}
        )
    else:
        z_quali = _zscores({f.driver_id: f.quali for f in features})

    return {
        f.driver_id: (
            weights.form * z_form[f.driver_id]
            + weights.season * z_season[f.driver_id]
            + weights.quali * z_quali[f.driver_id]
            + weights.team * z_team[f.driver_id]
            + weights.track * z_track[f.driver_id]
        )
        for f in features
    }


def simulate(
    features: Sequence[DriverFeatures],
    strength_by_driver: Mapping[str, float],
    simulations: int = SIMULATIONS,
    seed: int = SEED,
    beta: float = BETA,
) -> list[DriverProbability]:
    """Монте-Карло: розігруємо порядок фінішу simulations разів."""
    rng = random.Random(seed)
    wins: dict[str, int] = {f.driver_id: 0 for f in features}
    podiums: dict[str, int] = {f.driver_id: 0 for f in features}
    points: dict[str, int] = {f.driver_id: 0 for f in features}
    log_w = {f.driver_id: beta * strength_by_driver[f.driver_id] for f in features}

    for _ in range(simulations):
        keys: list[tuple[float, str]] = []
        for f in features:
            if rng.random() > f.reliability:  # схід — у кінець класифікації
                keys.append((-1e9 + rng.random(), f.driver_id))
                continue
            gumbel = -math.log(-math.log(rng.random()))
            keys.append((log_w[f.driver_id] + gumbel, f.driver_id))
        keys.sort(reverse=True)
        wins[keys[0][1]] += 1
        for _, driver_id in keys[:3]:
            podiums[driver_id] += 1
        for _, driver_id in keys[:10]:
            points[driver_id] += 1

    result = [
        DriverProbability(
            driver_id=f.driver_id,
            name=f.name,
            code=f.code,
            team=f.constructor_name,
            win_pct=round(100 * wins[f.driver_id] / simulations, 2),
            podium_pct=round(100 * podiums[f.driver_id] / simulations, 2),
            points_pct=round(100 * points[f.driver_id] / simulations, 2),
            strength=round(strength_by_driver[f.driver_id], 3),
            grid=f.grid,
            features={
                "form": None if f.form is None else round(f.form, 2),
                "season_pace": None if f.season is None else round(f.season, 2),
                "quali_pace": None if f.quali is None else round(f.quali, 2),
                "team_pace": None if f.team is None else round(f.team, 2),
                "track_record": None if f.track is None else round(f.track, 2),
                "reliability": round(f.reliability, 3),
                "races_counted": f.races_counted,
            },
        )
        for f in features
    ]
    # За рівних відсотків (у межах шуму Монте-Карло) вище стає сильніший за фічами —
    # інакше порядок топ-2 стрибав би від кількості симуляцій.
    result.sort(key=lambda d: (d.win_pct, d.strength), reverse=True)
    return result


def mc_stderr_pp(p_pct: float, simulations: int) -> float:
    """Стандартна похибка оцінки ймовірності, у відсоткових пунктах."""
    p = p_pct / 100
    return round(100 * math.sqrt(max(p * (1 - p), 1e-9) / simulations), 2)


def predict(
    races: Sequence[Race],
    circuit_races: Sequence[Race],
    entries: Sequence[Entry],
    actual_grid: Mapping[str, int] | None = None,
    simulations: int = SIMULATIONS,
    seed: int = SEED,
    beta: float = BETA,
    weights: Weights | None = None,
) -> list[DriverProbability]:
    """Повний прохід: фічі → сили → ймовірності."""
    if not races or not entries:
        return []
    features = build_features(races, circuit_races, entries, actual_grid)
    use_grid = bool(actual_grid) and sum(1 for f in features if f.grid) >= len(features) // 2
    return simulate(features, strengths(features, use_grid, weights), simulations, seed, beta)
