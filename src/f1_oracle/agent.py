"""Агент F1 Oracle: LangChain 1.x create_agent + інструменти + межі циклу.

Розподіл ролей навмисний:
  • інструменти дають факти й ймовірності (детерміновано, з API);
  • LLM відповідає за інтерпретацію, контекст і структуровану відповідь.
LLM не рахує відсотки — інакше він їх просто вигадає.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)
from langchain.agents.structured_output import ToolStrategy
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from .config import MODEL, MODEL_CALL_LIMIT, TOOL_CALL_LIMIT
from .tools import ALL_TOOLS

SYSTEM_PROMPT = """\
Ти — F1 Oracle, аналітик, який прогнозує переможця наступної гонки Формули-1.

Порядок роботи:
1. get_next_race — дізнайся, яка гонка попереду.
2. run_prediction_model — це головне джерело ймовірностей. Виклич обовʼязково.
3. get_model_backtest — перевір, наскільки модель була точною цього сезону.
4. Далі бери контекст, якого немає в цифрах: get_qualifying_grid, get_race_weather,
   get_circuit_history, get_recent_form, get_championship_standings.

Жорсткі правила:
- Жодного відсотка з голови. Усі ймовірності — рівно ті, що повернув run_prediction_model.
- Не вигадуй пілотів і команди: у відповіді можуть бути лише ті, кого повернули інструменти.
- Якщо різниця між двома лідерами менша за monte_carlo_stderr_pp — так і скажи, що шанси рівні.
- Впевненість став за бектестом, а не за відчуттям: top-1 accuracy нижче 0.5 означає
  щонайбільше "medium", а якщо модель гірша за бейзлайн championship_leader — "low".
- Можеш не погодитись із моделлю (наприклад, через дощ або відому дискваліфікацію),
  але тоді прямо напиши це в model_agreement і поясни причину.
- Кваліфікація ще не відбулася — це головне джерело невизначеності, згадай про це в risks.
- Відповідай українською, стисло і по суті, без води.
"""


class DriverPick(BaseModel):
    """Один пілот у прогнозі."""

    driver: str = Field(description="Повне імʼя пілота")
    team: str = Field(description="Команда")
    win_probability_pct: float = Field(description="Ймовірність перемоги у відсотках, з моделі")
    podium_probability_pct: float = Field(description="Ймовірність подіуму у відсотках, з моделі")
    reason: str = Field(description="Одне речення: чому саме така оцінка")


class RaceForecast(BaseModel):
    """Структурована відповідь агента."""

    race: str = Field(description="Назва гонки")
    circuit: str = Field(description="Траса")
    date: str = Field(description="Дата гонки, YYYY-MM-DD")
    predicted_winner: str = Field(description="Найімовірніший переможець")
    confidence: Literal["low", "medium", "high"] = Field(
        description="Впевненість, обґрунтована бектестом моделі"
    )
    top5: list[DriverPick] = Field(description="Пʼятірка фаворитів за спаданням ймовірності")
    podium: list[str] = Field(description="Трійка найімовірнішого подіуму, по порядку")
    key_factors: list[str] = Field(description="Фактори, що визначають результат")
    risks: list[str] = Field(description="Що може зламати цей прогноз")
    model_agreement: str = Field(
        description="Чи згоден агент із математичною моделлю і чому саме"
    )
    data_as_of: str = Field(description="Станом на який момент узяті дані")


MAX_TOKENS = int(os.getenv("F1_MAX_TOKENS", "8000"))


def build_chat_model(model: str | Any | None = None) -> Any:
    """Модель для агента. Підтримує два способи авторизації.

    • `ANTHROPIC_API_KEY` — звичайний ключ API;
    • `ANTHROPIC_AUTH_TOKEN` — короткоживучий OAuth-токен профілю `ant auth login`
      (запуск «через підписку»). Для /v1/messages такий токен вимагає ще й
      beta-заголовка `oauth-2025-04-20`, інакше запит відхиляється.

    max_tokens задається явно: у структурованій відповіді пʼять пілотів з
    поясненнями, і дефолтної межі langchain на це не вистачає.
    """
    spec = model or MODEL
    if not isinstance(spec, str):  # уже готовий BaseChatModel (напр. фейковий у тестах)
        return spec

    provider, _, name = spec.partition(":")
    if provider != "anthropic" or not name:
        return spec

    token = os.getenv("ANTHROPIC_AUTH_TOKEN")
    if token and not os.getenv("ANTHROPIC_API_KEY"):
        return ChatAnthropic(
            model=name,
            max_tokens=MAX_TOKENS,
            default_headers={
                "Authorization": f"Bearer {token}",
                "anthropic-beta": "oauth-2025-04-20",
            },
        )
    return ChatAnthropic(model=name, max_tokens=MAX_TOKENS)


class TraceMiddleware(AgentMiddleware):
    """Мінімальна спостережуваність (М7): що викликав агент і скільки це тривало."""

    def __init__(self, verbose: bool = False) -> None:
        super().__init__()
        self.verbose = verbose
        self.calls: list[dict[str, Any]] = []

    def wrap_tool_call(self, request, handler):  # type: ignore[override]
        name = request.tool_call["name"]
        started = time.perf_counter()
        try:
            result = handler(request)
            status = "ok"
            return result
        except Exception:
            status = "error"
            raise
        finally:
            elapsed = time.perf_counter() - started
            self.calls.append({"tool": name, "status": status, "seconds": round(elapsed, 2)})
            if self.verbose:
                print(f"  → {name} ({status}, {elapsed:.2f}s)")


@dataclass
class AgentRun:
    forecast: RaceForecast | None
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0


def build_agent(model: str | None = None, verbose: bool = False, tracer: TraceMiddleware | None = None):
    """Збирає агента. Межі циклу — не косметика, а захист від нескінченних викликів."""
    tracer = tracer or TraceMiddleware(verbose=verbose)
    return create_agent(
        model=build_chat_model(model),
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        response_format=ToolStrategy(RaceForecast),
        middleware=[
            tracer,
            ModelCallLimitMiddleware(run_limit=MODEL_CALL_LIMIT, exit_behavior="end"),
            ToolCallLimitMiddleware(run_limit=TOOL_CALL_LIMIT, exit_behavior="continue"),
            ToolRetryMiddleware(max_retries=2, initial_delay=0.5),
        ],
    )


DEFAULT_QUESTION = "Хто переможе в наступній гонці Формули-1? Дай повний прогноз."


def run(question: str = DEFAULT_QUESTION, model: str | None = None, verbose: bool = False) -> AgentRun:
    """Один прохід агента: питання → структурований прогноз."""
    tracer = TraceMiddleware(verbose=verbose)
    agent = build_agent(model=model, verbose=verbose, tracer=tracer)
    started = time.perf_counter()
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    elapsed = time.perf_counter() - started

    forecast = result.get("structured_response")
    messages = result.get("messages", [])
    text = ""
    for message in reversed(messages):
        content = getattr(message, "content", "")
        if isinstance(content, str) and content.strip():
            text = content
            break
        if isinstance(content, list):
            parts = [b.get("text", "") for b in content if isinstance(b, dict)]
            if any(parts):
                text = "\n".join(p for p in parts if p)
                break
    return AgentRun(
        forecast=forecast if isinstance(forecast, RaceForecast) else None,
        text=text,
        tool_calls=tracer.calls,
        seconds=round(elapsed, 2),
    )
