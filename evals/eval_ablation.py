#!/usr/bin/env python
"""ЕВАЛ 4 — абляція фіч із довірчими інтервалами (офлайн, без LLM).

Кожну фічу по черзі вимикаємо (вага = 0) і переганяємо весь бектест.
Питання: чи кожна фіча окупає своє існування — і чи різниця взагалі відрізнима від шуму.

Ключова деталь методики: сезон дає всього ~10 гонок, і на такій вибірці різниця
в log-loss 0.05 нічого не означає. Тому рахується не голе Δ, а **парний бутстреп**
по раундах: 90% довірчий інтервал середньої різниці. Фіча визнається корисною,
тільки якщо весь інтервал лежить вище нуля. Інакше чесна відповідь — "не відрізнити
від шуму", а не "шкідлива".

Запуск:  python evals/eval_ablation.py [--json]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace

from _harness import ROOT, skip  # noqa: F401  (ROOT додає src у sys.path)

from f1_oracle.data.jolpica import JolpicaClient  # noqa: E402
from f1_oracle.evaluation import BacktestReport, backtest  # noqa: E402
from f1_oracle.scoring import WEIGHTS_WITH_GRID  # noqa: E402
from f1_oracle.season import load_snapshot  # noqa: E402

TITLE = "ЕВАЛ 4 · абляція фіч"
FEATURES = {
    "form": "форма за останні гонки",
    "season": "темп за весь сезон",
    "quali": "кваліфікаційний темп / сітка",
    "team": "темп команди",
    "track": "історія траси",
}
SIMULATIONS = 12000
BOOTSTRAP = 5000
CI = 0.90
SEED = 20260904


def _bootstrap_ci(deltas: list[float]) -> tuple[float, float]:
    """Парний бутстреп по раундах: довірчий інтервал середньої різниці."""
    if len(deltas) < 2:
        return (float("-inf"), float("inf"))
    rng = random.Random(SEED)
    n = len(deltas)
    means = []
    for _ in range(BOOTSTRAP):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((1 - CI) / 2 * BOOTSTRAP)]
    hi = means[int((1 + CI) / 2 * BOOTSTRAP) - 1]
    return round(lo, 3), round(hi, 3)


def _log_loss_by_round(report: BacktestReport) -> dict[int, float]:
    return {r.round: r.log_loss for r in report.rounds}


def _verdict(lo: float, hi: float) -> str:
    if lo > 0:
        return "корисна"
    if hi < 0:
        return "шкідлива"
    return "не відрізнити від шуму"


def run(start_round: int = 4) -> dict:
    client = JolpicaClient()
    snapshot = load_snapshot(client)

    # Базовий прогін — на тому самому наборі ваг, що й абляції, інакше Δ змішувала б
    # ефект вимкненої фічі з ефектом перемикання наборів ваг. Береться WEIGHTS_WITH_GRID:
    # у бектесті кваліфікація вже відбулася перед кожною гонкою, тож саме цей набір
    # і працює в продакшні на цих раундах.
    base_report = backtest(
        snapshot, client, start_round=start_round, simulations=SIMULATIONS,
        weights=WEIGHTS_WITH_GRID,
    )
    base_summary = base_report.summary()
    if not base_summary.get("rounds_evaluated"):
        return {"status": "skipped", "reason": "замало проведених гонок"}
    base_by_round = _log_loss_by_round(base_report)

    rows = []
    for feature, label in FEATURES.items():
        ablated = backtest(
            snapshot, client, start_round=start_round, simulations=SIMULATIONS,
            weights=replace(WEIGHTS_WITH_GRID, **{feature: 0.0}),
        )
        summary = ablated.summary()
        by_round = _log_loss_by_round(ablated)
        # Δ > 0 означає "без фічі гірше", тобто фіча корисна.
        deltas = [by_round[rnd] - base_by_round[rnd] for rnd in sorted(base_by_round)]
        lo, hi = _bootstrap_ci(deltas)
        rows.append(
            {
                "feature": feature,
                "label": label,
                "log_loss": summary["model"]["log_loss"],
                "top1": summary["model"]["top1_accuracy"],
                "podium": summary["model"]["podium_overlap_avg"],
                "delta_log_loss": round(sum(deltas) / len(deltas), 3),
                "ci_low": lo,
                "ci_high": hi,
                "verdict": _verdict(lo, hi),
            }
        )
    rows.sort(key=lambda r: r["delta_log_loss"], reverse=True)

    useful = [r for r in rows if r["verdict"] == "корисна"]
    harmful = [r for r in rows if r["verdict"] == "шкідлива"]
    noise = [r for r in rows if r["verdict"] == "не відрізнити від шуму"]
    checks = {
        "жодна фіча не шкодить статистично значуще": not harmful,
        "щонайменше одна фіча значуще корисна": bool(useful),
        "вибірка достатня, щоб хоч щось розрізнити": len(noise) < len(rows),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "rounds": base_summary["rounds_evaluated"],
        "method": f"парний бутстреп по раундах, {BOOTSTRAP} ресемплів, CI {int(CI * 100)}%",
        "baseline": {
            "log_loss": base_summary["model"]["log_loss"],
            "top1": base_summary["model"]["top1_accuracy"],
            "podium": base_summary["model"]["podium_overlap_avg"],
        },
        "ablations": rows,
        "useful": [r["feature"] for r in useful],
        "harmful": [r["feature"] for r in harmful],
        "indistinguishable": [r["feature"] for r in noise],
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Евал абляції фіч")
    parser.add_argument("--start-round", type=int, default=4)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = run(args.start_round)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] != "failed" else 1
    if result["status"] == "skipped":
        return skip(TITLE, result["reason"])

    base = result["baseline"]
    print(f"\n{TITLE} · {result['rounds']} гонок · {result['method']}\n")
    print(f"  повна модель: log-loss {base['log_loss']:.3f} · top-1 {base['top1']:.3f} · "
          f"подіум {base['podium']:.3f}\n")
    print(f"  {'вимкнена фіча':<28} {'Δ log-loss':>11} {'90% CI':>18}   вердикт")
    for row in result["ablations"]:
        ci = f"[{row['ci_low']:+.2f}; {row['ci_high']:+.2f}]"
        print(f"  {row['label']:<28} {row['delta_log_loss']:>+11.3f} {ci:>18}   {row['verdict']}")
    print("\n  (Δ > 0 = без фічі гірше = фіча корисна)\n")
    for name, ok in result["checks"].items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if result["indistinguishable"]:
        print(f"\n  ⓘ на цій вибірці не відрізнити від шуму: "
              f"{', '.join(result['indistinguishable'])} — не привід ні викидати, ні залишати")
    print(f"\nРЕЗУЛЬТАТ: {result['status'].upper()}\n")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
