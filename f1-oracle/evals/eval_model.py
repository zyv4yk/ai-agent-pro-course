#!/usr/bin/env python
"""ЕВАЛ 1 — офлайн-якість детермінованої моделі (без LLM, без витрат на токени).

Проганяє модель по всіх уже проведених гонках сезону: для раунду R модель бачить
лише раунди < R. Порівнює з наївними бейзлайнами і виставляє pass/fail.

Запуск:  python evals/eval_model.py [--start-round 4] [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1_oracle.data.jolpica import JolpicaClient  # noqa: E402
from f1_oracle.evaluation import backtest  # noqa: E402
from f1_oracle.season import load_snapshot  # noqa: E402

# Пороги свідомо скромні: 9-12 гонок — це мала вибірка, і вимагати від моделі
# стабільного побиття "лідера чемпіонату" на такій вибірці було б самообманом.
THRESHOLDS = {
    "log_loss_better_than_uniform": True,
    "podium_overlap_avg_min": 0.40,
    "top1_accuracy_min": 0.25,
    "avg_prob_on_actual_winner_min": 0.15,
}


def run(start_round: int = 4) -> dict:
    client = JolpicaClient()
    snapshot = load_snapshot(client)
    summary = backtest(snapshot, client, start_round=start_round).summary()
    if not summary.get("rounds_evaluated"):
        return {"status": "skipped", "reason": "замало проведених гонок", **summary}

    model, base = summary["model"], summary["baselines"]
    checks = {
        "краще за випадковий вибір (log-loss)": model["log_loss"] < base["uniform_log_loss"],
        "подіум вгадується краще за третину": (
            model["podium_overlap_avg"] >= THRESHOLDS["podium_overlap_avg_min"]
        ),
        "top-1 не гірше за поріг": model["top1_accuracy"] >= THRESHOLDS["top1_accuracy_min"],
        "справжньому переможцю дано розумну ймовірність": (
            model["avg_prob_on_actual_winner"] >= THRESHOLDS["avg_prob_on_actual_winner_min"]
        ),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "summary": summary,
        # Головна чесна метрика: чи б'ємо ми найпростіший бейзлайн узагалі.
        "beats_championship_leader_baseline": (
            model["top1_accuracy"] > base["championship_leader_top1"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Офлайн-евал детермінованої моделі")
    parser.add_argument("--start-round", type=int, default=4)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = run(args.start_round)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] != "failed" else 1

    if result["status"] == "skipped":
        print(f"ЕВАЛ 1 · пропущено: {result['reason']}")
        return 0

    summary = result["summary"]
    model, base = summary["model"], summary["baselines"]
    print(f"\nЕВАЛ 1 · модель прогнозу · сезон {summary['season']}, "
          f"{summary['rounds_evaluated']} гонок\n")
    print(f"{'раунд':>6}  {'гонка':<28} {'прогноз':<24} {'p':>7}  {'факт':<24} ok")
    for row in summary["per_round"]:
        print(f"{row['round']:>6}  {row['race'][:28]:<28} {row['predicted'][:24]:<24} "
              f"{row['p']:>6.1f}%  {row['actual'][:24]:<24} {'✓' if row['hit'] else '✗'}")

    print(f"\n  top-1 accuracy ........ {model['top1_accuracy']:.3f}  "
          f"(лідер чемпіонату {base['championship_leader_top1']:.3f}, "
          f"попередній переможець {base['previous_winner_top1']:.3f}, "
          f"випадково {base['uniform_top1']:.3f})")
    print(f"  подіум (з 3) .......... {model['podium_overlap_avg']:.3f}")
    print(f"  Brier ................. {model['brier']:.3f}")
    print(f"  log-loss .............. {model['log_loss']:.3f}  "
          f"(випадково {base['uniform_log_loss']:.3f})")
    print(f"  p на переможці ........ {model['avg_prob_on_actual_winner']:.3f}\n")

    for name, ok in result["checks"].items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not result["beats_championship_leader_baseline"]:
        print("\n  ⚠ модель НЕ б'є бейзлайн 'завжди перемагає лідер чемпіонату' на цій вибірці —\n"
              "    саме тому агент зобов'язаний ставити впевненість не вище medium")
    print(f"\nРЕЗУЛЬТАТ: {result['status'].upper()}\n")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
