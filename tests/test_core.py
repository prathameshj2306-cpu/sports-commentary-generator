
"""Tests for the Sports Commentary Generator milestones."""

import json

import pytest

from app.core import DATA_DIR, milestone_1, validate_records


FOOTBALL_EVENT_TYPES = {
    "goal",
    "penalty",
    "save",
    "yellow_card",
    "red_card",
    "corner",
    "free_kick",
    "substitution",
    "missed_chance",
    "offside",
}

CRICKET_EVENT_TYPES = {
    "six",
    "four",
    "wicket",
    "dot_ball",
    "single",
    "double",
    "wide",
    "no_ball",
    "fifty",
    "century",
}

REQUIRED_FIELDS = {
    "sport",
    "event_type",
    "description",
    "team",
    "player",
    "match_context",
    "commentary",
}


def test_milestone_1():
    result = milestone_1()

    # Check dataset counts.
    assert result == {
        "football": 100,
        "cricket": 100,
        "total": 200,
    }

    # Check that all three dataset files exist.
    football_path = DATA_DIR / "football.jsonl"
    cricket_path = DATA_DIR / "cricket.jsonl"
    combined_path = DATA_DIR / "combined_dataset.jsonl"

    assert football_path.exists()
    assert cricket_path.exists()
    assert combined_path.exists()

    # Load all datasets.
    def read_jsonl(path):
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    football = read_jsonl(football_path)
    cricket = read_jsonl(cricket_path)
    combined = read_jsonl(combined_path)

    assert len(football) == 100
    assert len(cricket) == 100
    assert len(combined) == 200

    # Check sport-specific event types.
    assert {r["event_type"] for r in football} == FOOTBALL_EVENT_TYPES
    assert {r["event_type"] for r in cricket} == CRICKET_EVENT_TYPES

    # Check that records belong to the correct sport.
    assert all(r["sport"] == "football" for r in football)
    assert all(r["sport"] == "cricket" for r in cricket)

    # Check required fields and non-empty text.
    for record in combined:
        assert REQUIRED_FIELDS.issubset(record.keys())

        for field in REQUIRED_FIELDS:
            assert isinstance(record[field], str)
            assert record[field].strip()

    # Check for duplicate records.
    unique_records = {
        (r["sport"], r["description"], r["commentary"])
        for r in combined
    }
    assert len(unique_records) == len(combined)


def test_validate_records_rejects_missing_fields():
    invalid_record = {
        "sport": "football",
        "event_type": "goal",
        "description": "A goal is scored.",
    }

    with pytest.raises(ValueError):
        validate_records([invalid_record])


def test_validate_records_rejects_invalid_sport():
    invalid_record = {
        "sport": "basketball",
        "event_type": "score",
        "description": "A point is scored.",
        "team": "Team A",
        "player": "Player A",
        "match_context": "Final minute",
        "commentary": "What a moment!",
    }

    with pytest.raises(ValueError):
        validate_records([invalid_record])


def test_validate_records_rejects_duplicates():
    record = {
        "sport": "football",
        "event_type": "goal",
        "description": "A goal is scored.",
        "team": "Team A",
        "player": "Player A",
        "match_context": "Final minute",
        "commentary": "What a fantastic goal!",
    }

    with pytest.raises(ValueError):
        validate_records([record, record])


# Milestones 2–5 will be tested when we start implementing them.
@pytest.mark.skip(reason="Milestone 2 not started")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


@pytest.mark.skip(reason="Milestone 3 not started")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


@pytest.mark.skip(reason="Milestone 4 not started")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


@pytest.mark.skip(reason="Milestone 5 not started")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")
