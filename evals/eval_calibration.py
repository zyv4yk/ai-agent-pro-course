#!/usr/bin/env python
"""ЕВАЛ 3 — калібрування ймовірностей (офлайн, без LLM).

Точність — це одне, калібрування — зовсім інше. Питання цього евалу:
коли модель каже "25%", чи справді такі події трапляються приблизно в чверті випадків?
Модель, що вгадує переможця рідко, але з чесними відсотками, корисніша за
самовпевнену: на її ймовірності можна спиратись у рішеннях.

Метрика — ECE (expected calibration error): середнє зважене відхилення
передбаченої частоти від спостереженої по бінах.

Запуск:  python evals/eval_calibration.py [--json]
"""
from __future__ import annotations

import argparse
import json
import sys

from _harness import ROOT, skip  # noqa: F401  (ROOT додає src у sys.path)

from f1_oracle.data.jolpica import JolpicaClient  # noqa: E402
from f1_oracle.evaluation import backtest  # noqa: E402
from f1_oracle.season import load_snapshot  # noqa: E402

TITLE = "ЕВАЛ 3 · калібрування ймовірностей"
BINS = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 35), (35, 50), (50, 101)]

THRESHOLDS = {
    "ece_win_max": 0.06,
    "ece_podium_max": 0.10,
    "worst_bin_max": 0.30,   # для бінів, де є хоч якась статистика
    "min_sharpness": 0.15,   # середній максимум ймовірності: модель не має бути "рівномірною"
}
MIN_BIN_SIZE = 8


def _bucket(pairs: list[tuple[float, int]]) -> list[dict]:
    """pairs = [(ймовірність 0..1, чи сталася подія)]."""
    table = []
    for low, high in BINS:
        chunk = [(p, y) for p, y in pairs if low / 100 <= p < high / 100]
        if not chunk:
            continue
        predicted = sum(p for p, _ in chunk) / len(chunk)
        observed = sum(y for _, y in chunk) / len(chunk)
        table.append(
            {
                "bin": f"{low}-{high}%",
                "n": len(chunk),
                "predicted": round(predicted, 3),
                "observed": round(observed, 3),
                "gap": round(observed - predicted, 3),
            }
        )
    return table


def _ece(table: list[dict]) -> float:
    total = sum(row["n"] for row in table)
    if not total:
        return 0.0
    return round(sum(row["n"] * abs(row["gap"]) for row in table) / total, 4)


def run(start_round: int = 4) -> dict:
    client = JolpicaClient()
    snapshot = load_snapshot(client)
    report = backtest(snapshot, client, start_round=start_round, simulations=20000)
    if not report.rounds:
        return {"status": "skipped", "reason": "замало проведених гонок"}

    win_pairs: list[tuple[float, int]] = []
    podium_pairs: list[tuple[float, int]] = []
    sharpness: list[float] = []
    for outcome in report.rounds:
        sharpness.append(max(outcome.win_probabilities.values(), default=0.0))
        for driver_id, p in outcome.win_probabilities.items():
            win_pairs.append((p, int(driver_id == outcome.winner_id)))
        for driver_id, p in outcome.podium_probabilities.items():
            podium_pairs.append((p, int(driver_id in outcome.podium_ids)))

    win_table = _bucket(win_pairs)
    podium_table = _bucket(podium_pairs)
    ece_win, ece_podium = _ece(win_table), _ece(podium_table)
    worst = max(
        (abs(row["gap"]) for row in win_table + podium_table if row["n"] >= MIN_BIN_SIZE),
        default=0.0,
    )
    mean_sharpness = sum(sharpness) / len(sharpness)

    checks = {
        f"ECE перемоги ≤ {THRESHOLDS['ece_win_max']}": ece_win <= THRESHOLDS["ece_win_max"],
        f"ECE подіуму ≤ {THRESHOLDS['ece_podium_max']}": ece_podium <= THRESHOLDS["ece_podium_max"],
        f"жоден бін (n≥{MIN_BIN_SIZE}) не промахується більш ніж на "
        f"{THRESHOLDS['worst_bin_max']}": worst <= THRESHOLDS["worst_bin_max"],
        "модель не 'розмазує' ймовірності рівномірно": (
            mean_sharpness >= THRESHOLDS["min_sharpness"]
        ),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "rounds": len(report.rounds),
        "observations": {"win": len(win_pairs), "podium": len(podium_pairs)},
        "ece_win": ece_win,
        "ece_podium": ece_podium,
        "worst_bin_gap": round(worst, 3),
        "sharpness": round(mean_sharpness, 3),
        "win_table": win_table,
        "podium_table": podium_table,
        "checks": checks,
    }


def _print_table(title: str, table: list[dict]) -> None:
    print(f"  {title}")
    print(f"    {'бін':>9}  {'n':>5}  {'сказано':>8}  {'сталося':>8}  {'Δ':>7}")
    for row in table:
        flag = " " if abs(row["gap"]) <= 0.1 or row["n"] < MIN_BIN_SIZE else "!"
        print(f"    {row['bin']:>9}  {row['n']:>5}  {row['predicted']:>8.3f}  "
              f"{row['observed']:>8.3f}  {row['gap']:>+7.3f} {flag}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Евал калібрування ймовірностей")
    parser.add_argument("--start-round", type=int, default=4)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = run(args.start_round)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] != "failed" else 1
    if result["status"] == "skipped":
        return skip(TITLE, result["reason"])

    print(f"\n{TITLE} · {result['rounds']} гонок, "
          f"{result['observations']['win']} спостережень\n")
    _print_table("перемога", result["win_table"])
    _print_table("подіум", result["podium_table"])
    print(f"  ECE перемоги ......... {result['ece_win']:.4f}")
    print(f"  ECE подіуму .......... {result['ece_podium']:.4f}")
    print(f"  найгірший бін ........ {result['worst_bin_gap']:.3f}")
    print(f"  гострота (сер. max p)  {result['sharpness']:.3f}\n")
    for name, ok in result["checks"].items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\nРЕЗУЛЬТАТ: {result['status'].upper()}\n")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
