
"""Tests for the Sports Commentary Generator milestones."""

import json
import pytest
from transformers import GPT2Config, AutoModelForCausalLM
from app.core import preprocess_record

from app.core import (
    DATA_DIR,
    milestone_1,
    milestone_3,
    milestone_4,
    milestone_5,
    validate_records,
    inject_lora_adapter,
)


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
def test_milestone_2():
    # Create a tiny GPT-2 model without downloading anything.
    config = GPT2Config(
        vocab_size=100,
        n_positions=64,
        n_embd=32,
        n_layer=1,
        n_head=2,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )

    base_model = AutoModelForCausalLM.from_config(config)

    # Inject LoRA adapters.
    model = inject_lora_adapter(base_model)

    # Collect trainable and total parameters.
    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    # LoRA parameters must exist and be trainable.
    assert trainable_params > 0

    # Only a portion of the model should be trainable.
    assert trainable_params < total_params

    # Verify GPT-2 attention projection was adapted.
    assert any(
        "c_attn" in module_name
        for module_name in model.targeted_module_names
    )

    # Verify original base model parameters are frozen.
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if "lora_" not in name
    )


class DummyTokenizer:
    """Small tokenizer substitute for preprocessing tests."""

    eos_token_id = 99

    def __call__(self, text, add_special_tokens=False):
        tokens = text.split()
        return {"input_ids": list(range(1, len(tokens) + 1))}


def test_preprocess_record_masks_prompt():
    record = {
        "sport": "football",
        "event_type": "goal",
        "description": "Player scores a goal",
        "team": "Blue FC",
        "player": "Leo",
        "match_context": "Final minute",
        "commentary": "What a fantastic goal",
    }

    result = preprocess_record(record, DummyTokenizer())

    first_target = result["labels"].index(
        next(label for label in result["labels"] if label != -100)
    )

    # Prompt tokens must be masked.
    assert all(label == -100 for label in result["labels"][:first_target])

    # Commentary tokens must remain trainable.
    assert any(label != -100 for label in result["labels"])

    # Input and labels must have matching lengths.
    assert len(result["input_ids"]) == len(result["labels"])


def test_preprocess_record_respects_max_length():
    record = {
        "sport": "football",
        "event_type": "goal",
        "description": "Player scores a goal",
        "team": "Blue FC",
        "player": "Leo",
        "match_context": "Final minute",
        "commentary": "What a fantastic goal",
    }

    result = preprocess_record(
        record,
        DummyTokenizer(),
        max_length=10,
    )

    assert len(result["input_ids"]) <= 10
    assert len(result["labels"]) <= 10
    assert result["input_ids"][-1] == 99
    assert result["labels"][-1] == 99


def test_preprocess_record_rejects_invalid_max_length():
    record = {
        "sport": "football",
        "event_type": "goal",
        "description": "Player scores a goal",
        "team": "Blue FC",
        "player": "Leo",
        "match_context": "Final minute",
        "commentary": "What a fantastic goal",
    }

    with pytest.raises(ValueError):
        preprocess_record(record, DummyTokenizer(), max_length=1)




def test_milestone_3():
    """Verify the Streamlit application file exists and is valid Python."""

    app_path = DATA_DIR.parent / "app" / "main.py"

    assert app_path.exists()

    # Compile without executing the Streamlit application.
    compile(
        app_path.read_text(encoding="utf-8"),
        str(app_path),
        "exec",
    )


def test_milestone_4():
    """Verify evaluation generates commentary for representative events."""

    results = milestone_4()

    assert len(results) == 4

    for result in results:
        assert result["sport"] in {"football", "cricket"}
        assert result["event_type"]
        assert result["expected"].strip()
        assert isinstance(result["generated"], str)
        assert result["generated"].strip()


def test_milestone_5():
    """Verify LoRA merging creates a standalone model."""

    result = milestone_5()

    merged_path = DATA_DIR.parent / "models" / "sports-commentary-merged"

    assert result["base_model"] == "gpt2"
    assert merged_path.exists()
    assert (merged_path / "config.json").exists()
    assert (merged_path / "model.safetensors").exists()
    assert (merged_path / "tokenizer.json").exists()

