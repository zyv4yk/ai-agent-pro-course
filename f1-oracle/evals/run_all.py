#!/usr/bin/env python
"""Прогін усіх евалів однією командою.

  python evals/run_all.py             # усі (LLM-евали пропустяться без авторизації)
  python evals/run_all.py --offline   # лише ті, що не викликають LLM
  python evals/run_all.py --model anthropic:claude-sonnet-5
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent

OFFLINE = [
    ("ЕВАЛ 1 · точність моделі", "eval_model.py"),
    ("ЕВАЛ 3 · калібрування", "eval_calibration.py"),
    ("ЕВАЛ 4 · абляція фіч", "eval_ablation.py"),
]
LLM = [
    ("ЕВАЛ 2 · поведінка агента", "eval_agent.py"),
    ("ЕВАЛ 5 · guardrails", "eval_guardrails.py"),
    ("ЕВАЛ 6 · самоузгодженість", "eval_consistency.py"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Прогін усіх евалів")
    parser.add_argument("--offline", action="store_true", help="лише евали без LLM")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    plan = OFFLINE + ([] if args.offline else LLM)
    outcomes: list[tuple[str, int]] = []
    for title, script in plan:
        cmd = [sys.executable, str(EVALS_DIR / script)]
        if args.model and script in {name for _, name in LLM}:
            cmd += ["--model", args.model]
        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
        outcomes.append((title, subprocess.run(cmd, check=False).returncode))

    print(f"\n{'=' * 78}\nПІДСУМОК\n{'=' * 78}")
    for title, code in outcomes:
        print(f"  [{'PASS' if code == 0 else 'FAIL'}] {title}")
    failed = [t for t, c in outcomes if c != 0]
    print(f"\n{len(outcomes) - len(failed)}/{len(outcomes)} евалів пройдено\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
