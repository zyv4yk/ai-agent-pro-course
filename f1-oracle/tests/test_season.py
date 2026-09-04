"""Тести доменного шару: парсинг результатів і збірка знімка сезону."""
from __future__ import annotations

from f1_oracle.season import _is_classified, _parse_race

RAW_RACE = {
    "season": "2026",
    "round": "13",
    "raceName": "Italian Grand Prix",
    "date": "2026-09-06",
    "time": "13:00:00Z",
    "Circuit": {
        "circuitId": "monza",
        "circuitName": "Autodromo Nazionale di Monza",
        "Location": {"lat": "45.6156", "long": "9.28111", "locality": "Monza", "country": "Italy"},
    },
    "Results": [
        {
            "position": "2",
            "positionText": "2",
            "grid": "3",
            "points": "18",
            "status": "+5.123",
            "Driver": {"driverId": "b", "givenName": "B", "familyName": "Two", "code": "BBB"},
            "Constructor": {"constructorId": "t2", "name": "Team Two"},
        },
        {
            "position": "1",
            "positionText": "1",
            "grid": "1",
            "points": "25",
            "status": "Finished",
            "Driver": {"driverId": "a", "givenName": "A", "familyName": "One", "code": "AAA"},
            "Constructor": {"constructorId": "t1", "name": "Team One"},
        },
        {
            "position": "3",
            "positionText": "R",
            "grid": "2",
            "points": "0",
            "status": "Engine",
            "Driver": {"driverId": "c", "givenName": "C", "familyName": "Three", "code": "CCC"},
            "Constructor": {"constructorId": "t1", "name": "Team One"},
        },
    ],
}


def test_results_are_sorted_and_typed():
    race = _parse_race(RAW_RACE)
    assert [r.driver_id for r in race.results] == ["a", "b", "c"]
    assert race.winner is not None and race.winner.driver_id == "a"
    assert race.podium == ["a", "b", "c"]
    assert race.lat == 45.6156


def test_lapped_driver_counts_as_classified_but_retired_does_not():
    race = _parse_race(RAW_RACE)
    by_id = {r.driver_id: r for r in race.results}
    assert by_id["b"].classified is True  # фініш із відставанням — це фініш
    assert by_id["c"].classified is False  # зійшов з дистанції


def test_is_classified_edge_cases():
    assert _is_classified("Finished", "1")
    assert _is_classified("+1 Lap", "12")
    assert not _is_classified("Accident", "R")
    assert not _is_classified("Disqualified", "D")


def test_snapshot_entries_come_from_last_race(snapshot):
    ids = {e.driver_id for e in snapshot.entries}
    assert ids == {"fast", "fast2", "mid", "mid2", "slow", "slow2"}
    assert snapshot.name_of("fast") == "Fast Driver"


def test_race_start_time_is_timezone_aware(snapshot):
    assert snapshot.upcoming is not None
    assert snapshot.upcoming.starts_at.tzinfo is not None
