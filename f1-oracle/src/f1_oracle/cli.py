"""CLI: агент, чиста модель, бектест."""
from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import MODEL, SIMULATIONS
from .data.jolpica import JolpicaClient
from .evaluation import backtest
from .predictor import predict_upcoming
from .season import load_snapshot

console = Console()


def _snapshot_and_client():
    client = JolpicaClient()
    return load_snapshot(client), client


def cmd_next(_: argparse.Namespace) -> int:
    snapshot, _client = _snapshot_and_client()
    if snapshot.upcoming is None:
        console.print("[yellow]У календарі більше немає майбутніх гонок.[/]")
        return 0
    race = snapshot.upcoming
    console.print(
        Panel(
            f"[bold]{race.name}[/] — раунд {race.round}, сезон {snapshot.season}\n"
            f"{race.circuit_name}, {race.locality} ({race.country})\n"
            f"Дата: {race.date} {race.time or ''}\n"
            f"Проведено гонок: {len(snapshot.completed)} · "
            f"кваліфікація: {'є' if snapshot.upcoming_qualifying() else 'ще не було'}",
            title="Наступна гонка",
        )
    )
    return 0


def cmd_model(args: argparse.Namespace) -> int:
    snapshot, _client = _snapshot_and_client()
    prediction = predict_upcoming(snapshot, simulations=args.simulations)
    if prediction is None:
        console.print("[red]Недостатньо даних для прогнозу.[/]")
        return 1
    if args.json:
        print(json.dumps(prediction.as_dict(top=args.top), ensure_ascii=False, indent=2))
        return 0

    table = Table(title=f"{prediction.race} · {prediction.circuit} · {prediction.date}")
    table.add_column("#", justify="right")
    table.add_column("Пілот")
    table.add_column("Команда")
    table.add_column("Перемога", justify="right")
    table.add_column("Подіум", justify="right")
    table.add_column("Очки", justify="right")
    table.add_column("Сила", justify="right")
    for i, d in enumerate(prediction.drivers[: args.top], 1):
        table.add_row(
            str(i), d.name, d.team,
            f"{d.win_pct:.1f}%", f"{d.podium_pct:.1f}%", f"{d.points_pct:.1f}%",
            f"{d.strength:+.2f}",
        )
    console.print(table)
    console.print(
        f"[dim]Кваліфікація врахована: {'так' if prediction.used_qualifying else 'ні'} · "
        f"похибка Монте-Карло ±{prediction.as_dict()['monte_carlo_stderr_pp']} п.п. · "
        f"дані станом на {prediction.data_as_of}[/]"
    )
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    snapshot, client = _snapshot_and_client()
    summary = backtest(snapshot, client, start_round=args.start_round).summary()
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if not summary.get("rounds_evaluated"):
        console.print("[yellow]Недостатньо проведених гонок для бектесту.[/]")
        return 0

    table = Table(title=f"Бектест сезону {summary['season']} ({summary['rounds_evaluated']} гонок)")
    table.add_column("Раунд", justify="right")
    table.add_column("Гонка")
    table.add_column("Прогноз")
    table.add_column("p", justify="right")
    table.add_column("Факт")
    table.add_column("✓", justify="center")
    for row in summary["per_round"]:
        table.add_row(
            str(row["round"]), row["race"], row["predicted"], f"{row['p']:.1f}%",
            row["actual"], "[green]так[/]" if row["hit"] else "[red]ні[/]",
        )
    console.print(table)
    model, base = summary["model"], summary["baselines"]
    console.print(
        Panel(
            f"top-1 accuracy: [bold]{model['top1_accuracy']}[/] "
            f"(лідер чемпіонату: {base['championship_leader_top1']}, "
            f"переможець попередньої: {base['previous_winner_top1']}, "
            f"випадково: {base['uniform_top1']})\n"
            f"подіум (частка вгаданих із 3): [bold]{model['podium_overlap_avg']}[/]\n"
            f"Brier: [bold]{model['brier']}[/] · log-loss: [bold]{model['log_loss']}[/] "
            f"(випадково: {base['uniform_log_loss']})\n"
            f"середня ймовірність, дана справжньому переможцю: "
            f"[bold]{model['avg_prob_on_actual_winner']}[/]",
            title="Метрики",
        )
    )
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    import os

    from .agent import DEFAULT_QUESTION, run

    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        console.print(
            "[red]Немає облікових даних Anthropic.[/] Два способи:\n"
            "  • ключ:    скопіюйте .env.example у .env і впишіть ANTHROPIC_API_KEY\n"
            "  • профіль: [bold]ant auth login[/], далі\n"
            "             [dim]set -a; eval \"$(ant auth print-credentials --env)\"; set +a[/]\n"
            "[dim]Команди `model`, `backtest` і `next` працюють без цього — вони не кличуть LLM.[/]"
        )
        return 1

    question = args.question or DEFAULT_QUESTION
    console.print(f"[dim]Модель: {args.model or MODEL}[/]")
    try:
        result = run(question, model=args.model, verbose=not args.quiet)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Агент упав:[/] {exc}")
        return 1

    if args.json and result.forecast:
        print(result.forecast.model_dump_json(indent=2))
        return 0

    forecast = result.forecast
    if forecast is None:
        console.print(Panel(result.text or "порожня відповідь", title="Відповідь агента"))
        return 1

    console.print(
        Panel(
            f"[bold green]{forecast.predicted_winner}[/]\n"
            f"{forecast.race} · {forecast.circuit} · {forecast.date}\n"
            f"впевненість: [bold]{forecast.confidence}[/]",
            title="Прогноз",
        )
    )
    table = Table(title="Фаворити")
    table.add_column("Пілот")
    table.add_column("Команда")
    table.add_column("Перемога", justify="right")
    table.add_column("Подіум", justify="right")
    table.add_column("Чому")
    for pick in forecast.top5:
        table.add_row(
            pick.driver, pick.team,
            f"{pick.win_probability_pct:.1f}%", f"{pick.podium_probability_pct:.1f}%",
            pick.reason,
        )
    console.print(table)
    console.print(Panel("\n".join(f"• {f}" for f in forecast.key_factors), title="Ключові фактори"))
    console.print(Panel("\n".join(f"• {r}" for r in forecast.risks), title="Ризики"))
    console.print(Panel(forecast.model_agreement, title="Агент vs модель"))
    console.print(
        f"[dim]Подіум: {' → '.join(forecast.podium)} · дані: {forecast.data_as_of} · "
        f"{len(result.tool_calls)} викликів інструментів за {result.seconds}s[/]"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="f1-oracle", description="Агент-прогнозист Формули-1")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("next", help="показати наступну гонку")
    p.set_defaults(func=cmd_next)

    p = sub.add_parser("model", help="прогноз лише детермінованої моделі, без LLM")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--simulations", type=int, default=SIMULATIONS)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_model)

    p = sub.add_parser("backtest", help="перевірка моделі на вже проведених гонках")
    p.add_argument("--start-round", type=int, default=4)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("predict", help="повний прогноз агента (потрібен ANTHROPIC_API_KEY)")
    p.add_argument("question", nargs="?", default=None)
    p.add_argument("--model", default=None, help="напр. anthropic:claude-sonnet-5")
    p.add_argument("--json", action="store_true")
    p.add_argument("--quiet", action="store_true", help="не показувати виклики інструментів")
    p.set_defaults(func=cmd_predict)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
