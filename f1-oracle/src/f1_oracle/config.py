"""Конфігурація агента: модель, кеш, ліміти."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# .env шукаємо поруч із проєктом, а не лише в поточній теці — інакше евал,
# запущений з кореня репозиторію, його не побачить.
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()
CACHE_DIR = Path(os.getenv("F1_CACHE_DIR", str(PROJECT_ROOT / ".cache")))

# Модель для агента. Формат рядка — як у langchain: "<провайдер>:<модель>".
MODEL = os.getenv("F1_MODEL", "anthropic:claude-opus-5")

# Межі агентного циклу (М1: MAX_TURNS, М6: guardrails).
MODEL_CALL_LIMIT = int(os.getenv("F1_MODEL_CALL_LIMIT", "12"))
TOOL_CALL_LIMIT = int(os.getenv("F1_TOOL_CALL_LIMIT", "20"))

# Скільки сезонів історії траси враховувати.
TRACK_HISTORY_YEARS = int(os.getenv("F1_TRACK_HISTORY_YEARS", "3"))

# Скільки останніх гонок формує "форму".
FORM_WINDOW = int(os.getenv("F1_FORM_WINDOW", "5"))

# Симуляцій Монте-Карло на один прогноз.
SIMULATIONS = int(os.getenv("F1_SIMULATIONS", "20000"))
SEED = int(os.getenv("F1_SEED", "42"))

HTTP_TIMEOUT = float(os.getenv("F1_HTTP_TIMEOUT", "30"))

# Крутість софтмакса: більше — модель впевненіша у фавориті.
BETA = float(os.getenv("F1_BETA", "3.0"))

# Затухання ваги старіших гонок у "формі".
RECENCY_DECAY = float(os.getenv("F1_RECENCY_DECAY", "0.85"))
