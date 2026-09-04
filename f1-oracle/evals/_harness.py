"""Спільна обвʼязка евалів: перевірки, друк звіту, доступ до "правди" моделі.

Кожен евал — це набір `Check`ів із машинним критерієм. Жодного LLM-судді:
евал має бути відтворюваним і не коштувати ще одного виклику моделі на кейс.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


@dataclass
class Check:
    """Одна машинна перевірка."""

    name: str
    passed: bool
    detail: str = ""


@dataclass
class CaseResult:
    """Результат одного кейса евалу."""

    prompt: str
    checks: list[Check] = field(default_factory=list)
    seconds: float = 0.0
    tool_calls: int = 0
    error: str = ""

    @property
    def score(self) -> float:
        return sum(c.passed for c in self.checks) / len(self.checks) if self.checks else 0.0


@dataclass
class GroundTruth:
    """Те, з чим звіряємо відповідь агента: вихід детермінованої моделі й бектест."""

    prediction: object
    valid_driver_names: set[str]
    backtest: dict

    @property
    def win_by_name(self) -> dict[str, float]:
        return {d.name: d.win_pct for d in self.prediction.drivers}

    @property
    def podium_by_name(self) -> dict[str, float]:
        return {d.name: d.podium_pct for d in self.prediction.drivers}


def has_credentials() -> bool:
    """Ключ API або OAuth-токен підписки (`ant auth print-credentials --env`)."""
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


def credential_source() -> str:
    if os.getenv("ANTHROPIC_API_KEY"):
        return "ANTHROPIC_API_KEY"
    if os.getenv("ANTHROPIC_AUTH_TOKEN"):
        return "OAuth-профіль (підписка)"
    return "немає"


def load_ground_truth() -> GroundTruth | None:
    """Тягне знімок сезону один раз і рахує еталонні цифри для звірки."""
    from f1_oracle.evaluation import backtest
    from f1_oracle.predictor import predict_upcoming
    from f1_oracle.tools import get_client, get_snapshot

    snapshot = get_snapshot()
    prediction = predict_upcoming(snapshot)
    if prediction is None:
        return None
    return GroundTruth(
        prediction=prediction,
        valid_driver_names={e.name for e in snapshot.entries},
        backtest=backtest(snapshot, get_client()).summary(),
    )


def allowed_confidence(backtest_summary: dict) -> set[str]:
    """Стеля впевненості, яку заслужив бектест (guardrail проти самовпевненості)."""
    top1 = backtest_summary.get("model", {}).get("top1_accuracy", 0.0)
    leader = backtest_summary.get("baselines", {}).get("championship_leader_top1", 0.0)
    if top1 < 0.3:
        return {"low"}
    if top1 <= leader:
        return {"low", "medium"}
    return {"low", "medium", "high"}


def report(title: str, results: list[CaseResult], threshold: float, as_json: bool) -> int:
    """Друкує звіт і повертає код виходу (0 — пройдено)."""
    scored = [r for r in results if not r.error]
    total = sum(r.score for r in scored) / len(scored) if scored else 0.0
    status = "passed" if scored and total >= threshold and not any(r.error for r in results) else "failed"

    if as_json:
        print(
            json.dumps(
                {
                    "eval": title,
                    "status": status,
                    "score": round(total, 3),
                    "threshold": threshold,
                    "cases": [
                        {
                            "prompt": r.prompt,
                            "score": round(r.score, 3),
                            "seconds": r.seconds,
                            "tool_calls": r.tool_calls,
                            "error": r.error,
                            "checks": [
                                {"name": c.name, "passed": c.passed, "detail": c.detail}
                                for c in r.checks
                            ],
                        }
                        for r in results
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if status == "passed" else 1

    print(f"\n{title} · авторизація: {credential_source()}\n")
    for r in results:
        print(f"— {r.prompt}")
        if r.error:
            print(f"    [ERROR] {r.error}\n")
            continue
        for c in r.checks:
            mark = "PASS" if c.passed else "FAIL"
            detail = f"  ({c.detail})" if (c.detail and not c.passed) else ""
            print(f"    [{mark}] {c.name}{detail}")
        print(f"    score {r.score:.2f} · {r.tool_calls} викликів · {r.seconds}s\n")
    print(f"РЕЗУЛЬТАТ: {status.upper()} · {total:.3f} (поріг {threshold})\n")
    return 0 if status == "passed" else 1


def skip(title: str, reason: str) -> int:
    print(f"{title} · пропущено: {reason}")
    return 0
