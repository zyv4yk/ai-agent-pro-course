"""Бектест детермінованої моделі на вже проведених гонках сезону (М7: Evaluation).

Для кожного раунду R модель бачить лише раунди < R, а її прогноз
порівнюється з реальним результатом. Це відповідь на питання
"а чому взагалі можна вірити цим відсоткам".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import BETA, SEED
from .data.jolpica import JolpicaClient
from .predictor import predict_past_round
from .scoring import Weights
from .season import Snapshot

EPS = 1e-4


@dataclass
class RoundOutcome:
    round: int
    race: str
    actual_winner: str
    predicted_winner: str
    predicted_win_pct: float
    winner_probability: float
    hit: bool
    podium_overlap: int
    brier: float
    log_loss: float
    leader_pick: str
    leader_hit: bool
    prev_winner_pick: str
    prev_winner_hit: bool
    # повні розподіли — для евалів калібрування й абляції (у summary не потрапляють)
    win_probabilities: dict[str, float] = field(default_factory=dict)
    podium_probabilities: dict[str, float] = field(default_factory=dict)
    winner_id: str = ""
    podium_ids: list[str] = field(default_factory=list)


@dataclass
class BacktestReport:
    season: int
    rounds: list[RoundOutcome] = field(default_factory=list)
    field_size: int = 20

    @property
    def n(self) -> int:
        return len(self.rounds)

    def _mean(self, attr: str) -> float:
        return sum(getattr(r, attr) for r in self.rounds) / self.n if self.n else 0.0

    def summary(self) -> dict:
        if not self.n:
            return {"rounds_evaluated": 0, "note": "недостатньо проведених гонок для бектесту"}
        return {
            "season": self.season,
            "rounds_evaluated": self.n,
            "model": {
                "top1_accuracy": round(sum(r.hit for r in self.rounds) / self.n, 3),
                "podium_overlap_avg": round(self._mean("podium_overlap") / 3, 3),
                "brier": round(self._mean("brier"), 3),
                "log_loss": round(self._mean("log_loss"), 3),
                "avg_prob_on_actual_winner": round(self._mean("winner_probability"), 3),
            },
            "baselines": {
                "championship_leader_top1": round(
                    sum(r.leader_hit for r in self.rounds) / self.n, 3
                ),
                "previous_winner_top1": round(
                    sum(r.prev_winner_hit for r in self.rounds) / self.n, 3
                ),
                "uniform_top1": round(1 / self.field_size, 3),
                "uniform_log_loss": round(math.log(self.field_size), 3),
            },
            "per_round": [
                {
                    "round": r.round,
                    "race": r.race,
                    "predicted": r.predicted_winner,
                    "p": r.predicted_win_pct,
                    "actual": r.actual_winner,
                    "hit": r.hit,
                    "podium_3of3": r.podium_overlap,
                }
                for r in self.rounds
            ],
        }


def _points_before(snapshot: Snapshot, target_round: int) -> dict[str, float]:
    totals: dict[str, float] = {}
    for race in snapshot.completed:
        if race.round >= target_round:
            continue
        for result in race.results:
            totals[result.driver_id] = totals.get(result.driver_id, 0.0) + result.points
    return totals


def backtest(
    snapshot: Snapshot,
    client: JolpicaClient | None = None,
    start_round: int = 4,
    use_qualifying: bool = True,
    simulations: int = 12000,
    seed: int = SEED,
    beta: float = BETA,
    weights: Weights | None = None,
) -> BacktestReport:
    """Проганяє модель по всіх раундах сезону, починаючи з start_round."""
    report = BacktestReport(season=snapshot.season)
    for race in snapshot.completed:
        if race.round < start_round or not race.results:
            continue
        prediction, actual = predict_past_round(
            snapshot,
            race.round,
            client=client,
            use_qualifying=use_qualifying,
            simulations=simulations,
            seed=seed,
            beta=beta,
            weights=weights,
        )
        if prediction is None or actual is None or not prediction.drivers:
            continue
        winner = actual.winner
        if winner is None:
            continue

        probs = {d.driver_id: d.win_pct / 100 for d in prediction.drivers}
        p_winner = max(probs.get(winner.driver_id, 0.0), EPS)
        brier = sum(
            (p - (1.0 if driver_id == winner.driver_id else 0.0)) ** 2
            for driver_id, p in probs.items()
        )
        top = prediction.drivers[0]
        predicted_podium = {d.driver_id for d in prediction.drivers[:3]}

        points = _points_before(snapshot, race.round)
        leader = max(points, key=points.get) if points else ""
        previous = next(
            (r.winner for r in reversed(snapshot.completed) if r.round < race.round and r.winner),
            None,
        )

        report.field_size = max(report.field_size, len(prediction.drivers))
        report.rounds.append(
            RoundOutcome(
                round=race.round,
                race=race.name,
                actual_winner=winner.name,
                predicted_winner=top.name,
                predicted_win_pct=top.win_pct,
                winner_probability=round(p_winner, 4),
                hit=top.driver_id == winner.driver_id,
                podium_overlap=len(predicted_podium & set(actual.podium)),
                brier=brier,
                log_loss=-math.log(p_winner),
                leader_pick=snapshot.name_of(leader),
                leader_hit=leader == winner.driver_id,
                prev_winner_pick=previous.name if previous else "",
                prev_winner_hit=bool(previous and previous.driver_id == winner.driver_id),
                win_probabilities=probs,
                podium_probabilities={
                    d.driver_id: d.podium_pct / 100 for d in prediction.drivers
                },
                winner_id=winner.driver_id,
                podium_ids=list(actual.podium),
            )
        )
    return report
