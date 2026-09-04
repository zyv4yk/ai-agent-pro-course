"""Тести бектесту: метрики мають рахуватись і бути в очікуваних межах."""
from __future__ import annotations

import math

from f1_oracle.evaluation import backtest


def test_backtest_produces_metrics(snapshot):
    summary = backtest(snapshot, client=None, start_round=3, simulations=2000).summary()
    assert summary["rounds_evaluated"] == 4
    model = summary["model"]
    assert 0.0 <= model["top1_accuracy"] <= 1.0
    assert 0.0 <= model["podium_overlap_avg"] <= 1.0
    assert model["log_loss"] > 0


def test_backtest_beats_random_on_a_predictable_season(snapshot):
    """Сезон, де alpha домінує, модель зобовʼязана вгадувати краще за випадок."""
    summary = backtest(snapshot, client=None, start_round=3, simulations=4000).summary()
    assert summary["model"]["top1_accuracy"] > summary["baselines"]["uniform_top1"]
    assert summary["model"]["log_loss"] < summary["baselines"]["uniform_log_loss"]


def test_backtest_uses_only_past_races(snapshot):
    """Головна пастка бектесту — заглядання в майбутнє. Перевіряємо на першому раунді."""
    from f1_oracle.predictor import predict_past_round

    prediction, actual = predict_past_round(snapshot, 1, simulations=500)
    assert prediction is None  # до першої гонки історії немає — прогнозувати нічим
    assert actual is not None


def test_empty_report_is_safe():
    from f1_oracle.evaluation import BacktestReport

    summary = BacktestReport(season=2026).summary()
    assert summary["rounds_evaluated"] == 0
    assert not math.isnan(float(summary["rounds_evaluated"]))
