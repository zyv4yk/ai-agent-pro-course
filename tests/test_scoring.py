"""Тести детермінованої моделі — вона має бути передбачуваною і чесною."""
from __future__ import annotations

import pytest

from f1_oracle.scoring import build_features, mc_stderr_pp, predict
from f1_oracle.season import Entry
from tests.conftest import make_race

SIMS = 4000


def _by_id(probabilities):
    return {d.driver_id: d for d in probabilities}


def test_probabilities_sum_to_one(races, entries):
    result = predict(races, [], entries, simulations=SIMS)
    assert pytest.approx(sum(d.win_pct for d in result), abs=0.5) == 100.0
    assert pytest.approx(sum(d.podium_pct for d in result), abs=1.5) == 300.0


def test_stronger_driver_wins_more_often(races, entries):
    result = _by_id(predict(races, [], entries, simulations=SIMS))
    assert result["fast"].win_pct > result["mid"].win_pct > result["slow"].win_pct


def test_prediction_is_deterministic_for_same_seed(races, entries):
    first = predict(races, [], entries, simulations=SIMS, seed=7)
    second = predict(races, [], entries, simulations=SIMS, seed=7)
    assert [d.win_pct for d in first] == [d.win_pct for d in second]


def test_dnf_history_lowers_win_probability(races, entries):
    """Той самий темп, але з двома сходами — нижча ймовірність перемоги."""
    clean = _by_id(predict(races, [], entries, simulations=SIMS))["fast"].win_pct
    unreliable_races = races[:-2] + [
        make_race(5, ["fast", "fast2", "mid", "mid2", "slow", "slow2"], dnf={"fast"}),
        make_race(6, ["fast", "fast2", "mid", "mid2", "slow", "slow2"], dnf={"fast"}),
    ]
    broken = _by_id(predict(unreliable_races, [], entries, simulations=SIMS))["fast"].win_pct
    assert broken < clean


def test_actual_grid_overrides_qualifying_form(races, entries):
    """Аутсайдер із поулу мусить отримати помітно більше шансів, ніж без поулу."""
    baseline = _by_id(predict(races, [], entries, simulations=SIMS))["slow"].win_pct
    grid = {"slow": 1, "fast": 2, "fast2": 3, "mid": 4, "mid2": 5, "slow2": 6}
    with_pole = _by_id(predict(races, [], entries, grid, simulations=SIMS))["slow"].win_pct
    assert with_pole > baseline * 1.5


def test_debutant_without_history_gets_field_average(races, entries):
    """Новачок без жодної гонки не отримує ні нуля, ні фаворитства."""
    rookie = Entry("rookie", "Rookie Driver", "ROO", "gamma", "Gamma")
    result = _by_id(predict(races, [], [*entries, rookie], simulations=SIMS))
    assert 0.0 < result["rookie"].win_pct < result["fast"].win_pct
    assert result["rookie"].features["form"] is None


def test_track_history_shifts_probabilities(races, entries):
    """Історія траси — окрема фіча, вона мусить впливати на результат."""
    without = _by_id(predict(races, [], entries, simulations=SIMS))["slow"].win_pct
    slow_owns_the_track = [
        make_race(1, ["slow", "slow2", "mid", "mid2", "fast", "fast2"], season=y)
        for y in (2025, 2024)
    ]
    with_history = _by_id(
        predict(races, slow_owns_the_track, entries, simulations=SIMS)
    )["slow"].win_pct
    assert with_history > without


def test_monte_carlo_error_shrinks_with_more_simulations():
    assert mc_stderr_pp(25.0, 40000) < mc_stderr_pp(25.0, 1000)


def test_empty_input_returns_no_prediction(entries):
    assert predict([], [], entries) == []
