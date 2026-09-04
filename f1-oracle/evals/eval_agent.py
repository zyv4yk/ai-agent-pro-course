#!/usr/bin/env python
"""ЕВАЛ 2 — базова поведінка агента (М7: Evaluation).

Перевіряє не "чи гарно звучить відповідь", а те, що можна перевірити машинно:
чи агент справді сходив у модель, чи не вигадав пілотів і відсотки,
чи його впевненість узгоджена з бектестом.

Запуск:  python evals/eval_agent.py [--model anthropic:claude-sonnet-5] [--json]
"""
from __future__ import annotations

import argparse
import sys

from _harness import (  # noqa: E402
    Check,
    CaseResult,
    GroundTruth,
    allowed_confidence,
    has_credentials,
    load_ground_truth,
    report,
    skip,
)

from f1_oracle.agent import AgentRun, RaceForecast, run as run_agent  # noqa: E402

TITLE = "ЕВАЛ 2 · базова поведінка агента"
PROB_TOLERANCE_PP = 2.5  # допуск на шум Монте-Карло між прогонами

CASES = [
    "Хто переможе в наступній гонці Формули-1?",
    "Дай прогноз на найближчий Гран-прі: фаворити, подіум і ризики.",
    "Коротко: хто виграє наступну гонку і наскільки ти в цьому впевнений?",
]


def base_checks(result: AgentRun, truth: GroundTruth) -> list[Check]:
    """Перевірки, спільні для всіх евалів агента."""
    checks: list[Check] = []
    forecast = result.forecast
    tools_used = [c["tool"] for c in result.tool_calls]

    checks.append(
        Check("структурована відповідь", isinstance(forecast, RaceForecast),
              "" if forecast else "агент не повернув RaceForecast")
    )
    checks.append(
        Check("викликав модель прогнозу", "run_prediction_model" in tools_used,
              f"викликані інструменти: {tools_used}")
    )
    if not isinstance(forecast, RaceForecast):
        return checks

    # 1. Жодних вигаданих пілотів — і в топ-5, і в подіумі.
    mentioned = {p.driver for p in forecast.top5} | set(forecast.podium) | {
        forecast.predicted_winner
    }
    invented = sorted(mentioned - truth.valid_driver_names)
    checks.append(Check("не вигадав пілотів", not invented, f"невідомі: {invented}"))

    # 2. Відсотки взяті з моделі, а не з голови.
    win, podium = truth.win_by_name, truth.podium_by_name
    drifted = [
        f"{p.driver}: {p.win_probability_pct} vs {win.get(p.driver)}"
        for p in forecast.top5
        if p.driver in win and abs(p.win_probability_pct - win[p.driver]) > PROB_TOLERANCE_PP
    ] + [
        f"{p.driver} (подіум): {p.podium_probability_pct} vs {podium.get(p.driver)}"
        for p in forecast.top5
        if p.driver in podium
        and abs(p.podium_probability_pct - podium[p.driver]) > PROB_TOLERANCE_PP
    ]
    checks.append(Check("відсотки збігаються з моделлю", not drifted, "; ".join(drifted)))

    # 3. Впевненість калібрована бектестом (guardrail проти самовпевненості).
    allowed = allowed_confidence(truth.backtest)
    checks.append(
        Check("впевненість калібрована бектестом", forecast.confidence in allowed,
              f"confidence={forecast.confidence}, дозволено {sorted(allowed)}")
    )
    return checks


def grade(result: AgentRun, truth: GroundTruth) -> list[Check]:
    checks = base_checks(result, truth)
    forecast = result.forecast
    tools_used = [c["tool"] for c in result.tool_calls]

    checks.insert(
        2,
        Check("перевірив бектест моделі", "get_model_backtest" in tools_used,
              f"викликані інструменти: {tools_used}"),
    )
    if not isinstance(forecast, RaceForecast):
        return checks

    order = [p.win_probability_pct for p in forecast.top5]
    checks.append(Check("топ-5 упорядкований", order == sorted(order, reverse=True), str(order)))

    top_name = forecast.top5[0].driver if forecast.top5 else ""
    explained = forecast.predicted_winner != top_name and len(forecast.model_agreement) > 60
    checks.append(
        Check("переможець узгоджений із топом", forecast.predicted_winner == top_name or explained,
              f"winner={forecast.predicted_winner}, top1={top_name}")
    )
    checks.append(
        Check("подіум із трьох різних пілотів",
              len(forecast.podium) == 3 and len(set(forecast.podium)) == 3, str(forecast.podium))
    )
    checks.append(
        Check("названі фактори і ризики",
              len(forecast.key_factors) >= 2 and len(forecast.risks) >= 1,
              f"factors={len(forecast.key_factors)}, risks={len(forecast.risks)}")
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Евал базової поведінки агента")
    parser.add_argument("--model", default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()

    if not has_credentials():
        return skip(TITLE, "немає ні ANTHROPIC_API_KEY, ні OAuth-профілю")
    truth = load_ground_truth()
    if truth is None:
        return skip(TITLE, "немає майбутньої гонки")

    results: list[CaseResult] = []
    for prompt in CASES:
        case = CaseResult(prompt=prompt)
        try:
            agent_run = run_agent(prompt, model=args.model)
            case.seconds = agent_run.seconds
            case.tool_calls = len(agent_run.tool_calls)
            case.checks = grade(agent_run, truth)
        except Exception as exc:  # noqa: BLE001
            case.error = f"{type(exc).__name__}: {exc}"
        results.append(case)

    return report(TITLE, results, args.threshold, args.json)


if __name__ == "__main__":
    sys.exit(main())
