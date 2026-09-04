#!/usr/bin/env python
"""ЕВАЛ 6 — самоузгодженість і ефективність агента.

Модель прогнозу детермінована: на тих самих даних вона дає ті самі цифри.
Отже, будь-який розкид у відповідях агента на те саме питання — це його власна
нестабільність, а не властивість задачі. Цей евал її вимірює.

Заразом — ефективність: скільки викликів інструментів агент витрачає і чи не
кличе ту саму модель прогнозу по кілька разів за прогін.

Запуск:  python evals/eval_consistency.py [--runs 3] [--model ...] [--json]
"""
from __future__ import annotations

import argparse
import statistics
import sys

from _harness import (  # noqa: E402
    Check,
    CaseResult,
    has_credentials,
    load_ground_truth,
    report,
    skip,
)

from f1_oracle.agent import RaceForecast, run as run_agent  # noqa: E402

TITLE = "ЕВАЛ 6 · самоузгодженість і ефективність"
QUESTION = "Хто переможе в наступній гонці Формули-1?"

PROB_SPREAD_MAX_PP = 2.5   # детермінована модель → розкид має бути в межах шуму Монте-Карло
MAX_TOOL_CALLS = 12        # більше — це вже блукання, а не робота


def main() -> int:
    parser = argparse.ArgumentParser(description="Евал самоузгодженості агента")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--model", default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()

    if not has_credentials():
        return skip(TITLE, "немає ні ANTHROPIC_API_KEY, ні OAuth-профілю")
    if load_ground_truth() is None:
        return skip(TITLE, "немає майбутньої гонки")

    runs = []
    errors = []
    for _ in range(args.runs):
        try:
            runs.append(run_agent(QUESTION, model=args.model))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")

    case = CaseResult(prompt=f"{QUESTION}  (×{args.runs})")
    if errors:
        case.error = "; ".join(errors)
        return report(TITLE, [case], args.threshold, args.json)

    forecasts = [r.forecast for r in runs]
    case.seconds = round(sum(r.seconds for r in runs), 2)
    case.tool_calls = sum(len(r.tool_calls) for r in runs)

    case.checks.append(
        Check("усі прогони дали структуровану відповідь",
              all(isinstance(f, RaceForecast) for f in forecasts),
              f"{sum(isinstance(f, RaceForecast) for f in forecasts)}/{len(forecasts)}")
    )
    if not all(isinstance(f, RaceForecast) for f in forecasts):
        return report(TITLE, [case], args.threshold, args.json)

    winners = {f.predicted_winner for f in forecasts}
    case.checks.append(
        Check("той самий переможець у всіх прогонах", len(winners) == 1, str(sorted(winners)))
    )

    top_probs = [f.top5[0].win_probability_pct for f in forecasts if f.top5]
    spread = max(top_probs) - min(top_probs) if top_probs else 0.0
    case.checks.append(
        Check(f"ймовірність топ-піка стабільна (≤{PROB_SPREAD_MAX_PP} п.п.)",
              spread <= PROB_SPREAD_MAX_PP, f"розкид {spread:.2f} п.п. серед {top_probs}")
    )

    confidences = {f.confidence for f in forecasts}
    case.checks.append(
        Check("однакова впевненість у всіх прогонах", len(confidences) == 1,
              str(sorted(confidences)))
    )

    podiums = {tuple(f.podium) for f in forecasts}
    case.checks.append(
        Check("склад подіуму збігається", len({frozenset(p) for p in podiums}) == 1,
              str(sorted(podiums)))
    )

    model_calls = [
        sum(1 for c in r.tool_calls if c["tool"] == "run_prediction_model") for r in runs
    ]
    case.checks.append(
        Check("модель прогнозу викликана рівно раз за прогін",
              all(n == 1 for n in model_calls), f"виклики по прогонах: {model_calls}")
    )

    per_run_tools = [len(r.tool_calls) for r in runs]
    case.checks.append(
        Check(f"жодного блукання (≤{MAX_TOOL_CALLS} викликів за прогін)",
              all(n <= MAX_TOOL_CALLS for n in per_run_tools), str(per_run_tools))
    )

    broken = [c["tool"] for r in runs for c in r.tool_calls if c["status"] != "ok"]
    case.checks.append(Check("жоден інструмент не впав", not broken, str(broken)))

    if not args.json:
        latencies = [r.seconds for r in runs]
        print(f"\n  латентність прогонів: {latencies} с "
              f"(медіана {statistics.median(latencies):.1f})")
        print(f"  викликів інструментів: {per_run_tools}")

    return report(TITLE, [case], args.threshold, args.json)


if __name__ == "__main__":
    sys.exit(main())
