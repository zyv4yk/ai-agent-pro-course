"""Клієнт Jolpica-F1 (наступник Ergast API) з дисковим кешем.

Jolpica тримає ліміт ~4 запити/сек і 500/год для анонімних клієнтів,
тому кожна відповідь лягає на диск: минулі сезони — назавжди,
поточний — на кілька годин.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import httpx

from ..config import CACHE_DIR, HTTP_TIMEOUT

BASE_URL = "https://api.jolpi.ca/ergast/f1"
PAGE_SIZE = 100
CURRENT_TTL = 6 * 3600  # поточний сезон живе 6 годин
FINISHED_TTL = 365 * 24 * 3600  # завершені сезони не змінюються


class JolpicaError(RuntimeError):
    """Помилка отримання даних з Jolpica-F1."""


class JolpicaClient:
    def __init__(self, cache_dir: Path | None = None, offline: bool = False) -> None:
        self.cache_dir = Path(cache_dir or CACHE_DIR) / "jolpica"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self._client = httpx.Client(
            timeout=HTTP_TIMEOUT,
            headers={"User-Agent": "f1-oracle/0.1 (AI agents PRO course project)"},
        )

    # ── низький рівень ───────────────────────────────────────────────
    def _cache_path(self, url: str) -> Path:
        return self.cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()[:24]}.json"

    def _get(self, path: str, ttl: int = CURRENT_TTL, **params: Any) -> dict:
        query = "&".join(f"{k}={v}" for k, v in sorted(params.items()) if v is not None)
        url = f"{BASE_URL}/{path.strip('/')}.json" + (f"?{query}" if query else "")

        cached = self._cache_path(url)
        if cached.exists() and (time.time() - cached.stat().st_mtime) < ttl:
            return json.loads(cached.read_text())
        if self.offline:
            raise JolpicaError(f"offline-режим: немає кешу для {url}")

        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = self._client.get(url)
                if response.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                response.raise_for_status()
                payload = response.json()
                cached.write_text(json.dumps(payload))
                return payload
            except Exception as exc:  # noqa: BLE001 — ретраїмо будь-яку мережеву помилку
                last_error = exc
                time.sleep(0.5 * (attempt + 1))
        if cached.exists():  # протухлий кеш кращий за падіння агента
            return json.loads(cached.read_text())
        raise JolpicaError(f"не вдалося отримати {url}: {last_error}")

    def _paginate_races(self, path: str, ttl: int) -> list[dict]:
        """Ergast ріже результати по 100 рядків, тому гонки склеюємо по раунду."""
        merged: dict[str, dict] = {}
        offset = 0
        while True:
            data = self._get(path, ttl=ttl, limit=PAGE_SIZE, offset=offset)["MRData"]
            for race in data["RaceTable"]["Races"]:
                key = race["round"]
                if key not in merged:
                    merged[key] = {k: v for k, v in race.items()}
                else:
                    for field in ("Results", "QualifyingResults", "SprintResults"):
                        if field in race:
                            merged[key].setdefault(field, []).extend(race[field])
            offset += PAGE_SIZE
            if offset >= int(data["total"]):
                break
        return [merged[k] for k in sorted(merged, key=int)]

    @staticmethod
    def _ttl_for(season: int, current_season: int) -> int:
        return CURRENT_TTL if season >= current_season else FINISHED_TTL

    # ── публічні методи ──────────────────────────────────────────────
    def schedule(self, season: int) -> list[dict]:
        data = self._get(f"{season}", ttl=CURRENT_TTL, limit=30)
        return data["MRData"]["RaceTable"]["Races"]

    def season_results(self, season: int, current_season: int) -> list[dict]:
        return self._paginate_races(f"{season}/results", self._ttl_for(season, current_season))

    def season_qualifying(self, season: int, current_season: int) -> list[dict]:
        return self._paginate_races(f"{season}/qualifying", self._ttl_for(season, current_season))

    def circuit_results(self, season: int, circuit_id: str, current_season: int) -> list[dict]:
        return self._paginate_races(
            f"{season}/circuits/{circuit_id}/results", self._ttl_for(season, current_season)
        )

    def driver_standings(self, season: int) -> list[dict]:
        data = self._get(f"{season}/driverstandings", ttl=CURRENT_TTL, limit=PAGE_SIZE)
        lists = data["MRData"]["StandingsTable"]["StandingsLists"]
        return lists[0]["DriverStandings"] if lists else []

    def constructor_standings(self, season: int) -> list[dict]:
        data = self._get(f"{season}/constructorstandings", ttl=CURRENT_TTL, limit=PAGE_SIZE)
        lists = data["MRData"]["StandingsTable"]["StandingsLists"]
        return lists[0]["ConstructorStandings"] if lists else []
