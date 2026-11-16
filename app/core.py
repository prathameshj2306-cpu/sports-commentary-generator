
"""Core functionality for the Sports Commentary Generator."""

import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

REQUIRED_FIELDS = {
    "sport",
    "event_type",
    "description",
    "team",
    "player",
    "match_context",
    "commentary",
}


def load_jsonl(file_path):
    """Load records from a JSON Lines file."""
    records = []

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {file_path}, line {line_number}"
                ) from error

            records.append(record)

    return records


def validate_records(records):
    """Validate dataset structure, content, and duplicate records."""

    seen = set()

    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Every dataset record must be a JSON object.")

        if not REQUIRED_FIELDS.issubset(record.keys()):
            raise ValueError("A dataset record is missing required fields.")

        if record["sport"] not in {"football", "cricket"}:
            raise ValueError("Unsupported sport found in dataset.")

        for field in REQUIRED_FIELDS:
            if not isinstance(record[field], str) or not record[field].strip():
                raise ValueError(f"Invalid or empty field: {field}")

        unique_key = (
            record["sport"],
            record["description"],
            record["commentary"],
        )

        if unique_key in seen:
            raise ValueError("Duplicate event-commentary record found.")

        seen.add(unique_key)


def write_jsonl(file_path, records):
    """Write records into a JSON Lines file."""

    with file_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def milestone_1():
    """Load, validate, and combine football and cricket datasets."""

    football_path = DATA_DIR / "football.jsonl"
    cricket_path = DATA_DIR / "cricket.jsonl"

    if not football_path.exists():
        raise FileNotFoundError(f"Dataset not found: {football_path}")

    if not cricket_path.exists():
        raise FileNotFoundError(f"Dataset not found: {cricket_path}")

    football_records = load_jsonl(football_path)
    cricket_records = load_jsonl(cricket_path)

    # Check that each source file contains the correct sport.
    if any(record.get("sport") != "football" for record in football_records):
        raise ValueError("Football dataset contains non-football records.")

    if any(record.get("sport") != "cricket" for record in cricket_records):
        raise ValueError("Cricket dataset contains non-cricket records.")

    combined_records = football_records + cricket_records

    validate_records(combined_records)

    write_jsonl(DATA_DIR / "combined_dataset.jsonl", combined_records)

    return {
        "football": len(football_records),
        "cricket": len(cricket_records),
        "total": len(combined_records),
    }

def inject_lora_adapter(model):
    """
    Inject LoRA adapters into a GPT-2 causal language model.

    Args:
        model: Pretrained or locally initialized GPT-2 model.

    Returns:
        Model with LoRA adapters injected.
    """

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["c_attn"],
        fan_in_fan_out=True,
    )

    lora_model = get_peft_model(model, lora_config)

    return lora_model

def load_model_and_tokenizer(model_name="gpt2"):
    """Load a pretrained causal language model and its tokenizer."""

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def load_trained_model(
    model_name="gpt2",
    adapter_dir=None,
):
    """Load GPT-2 with the trained LoRA adapter."""

    if adapter_dir is None:
        adapter_dir = PROJECT_ROOT / "models" / "sports-commentary-lora"

    adapter_dir = Path(adapter_dir)

    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError("LoRA adapter configuration not found.")

    if not (adapter_dir / "adapter_model.safetensors").exists():
        raise FileNotFoundError("LoRA adapter weights not found.")

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base_model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model.config.pad_token_id = tokenizer.pad_token_id

    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    return model, tokenizer


def generate_commentary(
    record,
    model=None,
    tokenizer=None,
    max_new_tokens=50,
):
    """Generate sports commentary for a given event."""

    # Load the trained model only if it was not supplied.
    if model is None or tokenizer is None:
        model, tokenizer = load_trained_model()

    # Convert the event details into the training prompt format.
    prompt = format_training_prompt(record)

    # Tokenize the prompt.
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    # Move input tensors to the model's device.
    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    # Generate commentary.
    model.eval()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=3,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Extract only the newly generated tokens.
    prompt_length = inputs["input_ids"].shape[1]
    generated_tokens = output[0][prompt_length:]

    commentary = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return commentary.strip()



def format_training_prompt(record):
    """Convert a sports event into a structured prompt."""

    return (
        "### Sports Event\n"
        f"Sport: {record['sport']}\n"
        f"Event: {record['event_type']}\n"
        f"Description: {record['description']}\n"
        f"Team: {record['team']}\n"
        f"Player: {record['player']}\n"
        f"Match Context: {record['match_context']}\n"
        "\n"
        "### Commentary\n"
    )


def preprocess_record(record, tokenizer, max_length=128):
    """Tokenize one record and mask prompt tokens from the loss."""

    if max_length < 2:
        raise ValueError("max_length must be at least 2.")

    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must have an EOS token.")

    prompt = format_training_prompt(record)
    commentary = record["commentary"]

    prompt_tokens = tokenizer(
        prompt,
        add_special_tokens=False,
    )["input_ids"]

    target_tokens = tokenizer(
        commentary,
        add_special_tokens=False,
    )["input_ids"]

    eos_token_id = tokenizer.eos_token_id

    # Reserve one position for the EOS token.
    available_tokens = max_length - 1

    # Preserve the commentary whenever possible.
    if len(target_tokens) > available_tokens:
        target_tokens = target_tokens[:available_tokens]
        prompt_tokens = []
    else:
        prompt_budget = available_tokens - len(target_tokens)
        prompt_tokens = prompt_tokens[:prompt_budget]

    input_ids = prompt_tokens + target_tokens + [eos_token_id]

    labels = (
        [-100] * len(prompt_tokens)
        + target_tokens
        + [eos_token_id]
    )

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def prepare_training_dataset(tokenizer, max_length=128):
    """Load and preprocess the combined sports commentary dataset."""

    dataset_path = DATA_DIR / "combined_dataset.jsonl"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    records = load_jsonl(dataset_path)
    validate_records(records)

    processed_records = [
        preprocess_record(record, tokenizer, max_length)
        for record in records
    ]

    return processed_records

def train_lora_model(
    model_name="gpt2",
    output_dir=None,
    epochs=1,
    learning_rate=2e-4,
    max_length=128,
    max_samples=10,
):
    """Train LoRA adapters on the sports commentary dataset."""

    if epochs < 1:
        raise ValueError("epochs must be at least 1.")

    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be at least 1 or None.")

    if output_dir is None:
        output_dir = PROJECT_ROOT / "models" / "sports-commentary-lora"

    output_dir = Path(output_dir)

    # 1. Load pretrained model and tokenizer.
    model, tokenizer = load_model_and_tokenizer(model_name)

    # 2. Inject LoRA adapters.
    model = inject_lora_adapter(model)

    # 3. Prepare training examples.
    training_records = prepare_training_dataset(
        tokenizer,
        max_length=max_length,
    )

    if max_samples is not None:
        training_records = training_records[:max_samples]

    # 4. Create a DataLoader.
    def collate_batch(batch):
        """Convert a batch of tokenized records into PyTorch tensors."""

        return {
            "input_ids": torch.tensor(
                [item["input_ids"] for item in batch],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [item["attention_mask"] for item in batch],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                [item["labels"] for item in batch],
                dtype=torch.long,
            ),
        }


    train_loader = DataLoader(
        training_records,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_batch,
    )

    # 5. Optimize only LoRA parameters.
    optimizer = torch.optim.AdamW(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters(),
        ),
        lr=learning_rate,
    )

    model.train()
    epoch_losses = []

    # 6. Train the model.
    for epoch in range(epochs):
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        epoch_losses.append(average_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Average Loss: {average_loss:.4f}"
        )

    # 7. Save the trained LoRA adapter and tokenizer.
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\nLoRA training completed!")
    print("Adapter saved at:", output_dir)

    return {
        "model_name": model_name,
        "samples_trained": len(training_records),
        "epochs": epochs,
        "average_loss": epoch_losses[-1],
        "adapter_path": str(output_dir),
    }

# --- Milestone 2: Implement LoRA adapter injection and training.
def milestone_2():
    """Execute the LoRA training milestone."""

    return train_lora_model(
        model_name="gpt2",
        epochs=5,
        learning_rate=1e-4,
        max_length=128,
        max_samples=None,
    )


# --- Milestone 3: Build the Streamlit interface.
def milestone_3():
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Test commentary quality.

def evaluate_commentary():
    """Generate commentary for representative football and cricket events."""

    dataset_path = DATA_DIR / "combined_dataset.jsonl"
    records = load_jsonl(dataset_path)

    # Select four representative event types.
    selected_events = [
        ("football", "goal"),
        ("football", "penalty"),
        ("cricket", "six"),
        ("cricket", "wicket"),
    ]

    model, tokenizer = load_trained_model()

    results = []

    for sport, event_type in selected_events:
        record = next(
            (
                item for item in records
                if item["sport"] == sport
                and item["event_type"] == event_type
            ),
            None,
        )

        if record is None:
            print(f"Missing example: {sport} - {event_type}")
            continue

        generated = generate_commentary(
            record,
            model=model,
            tokenizer=tokenizer,
        )

        result = {
            "sport": sport,
            "event_type": event_type,
            "expected": record["commentary"],
            "generated": generated,
        }

        results.append(result)

        print("\n" + "=" * 60)
        print(f"Sport: {sport}")
        print(f"Event: {event_type}")
        print("Expected:", record["commentary"])
        print("Generated:", generated)

    return results


def milestone_4():
    """Execute the qualitative model evaluation milestone."""
    return evaluate_commentary()




def merge_lora_adapter(
    model_name="gpt2",
    adapter_dir=None,
    output_dir=None,
):
    """Merge the trained LoRA adapter into GPT-2 and save it."""

    if adapter_dir is None:
        adapter_dir = (
            PROJECT_ROOT / "models" / "sports-commentary-lora"
        )

    if output_dir is None:
        output_dir = (
            PROJECT_ROOT / "models" / "sports-commentary-merged"
        )

    adapter_dir = Path(adapter_dir)
    output_dir = Path(output_dir)

    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError("LoRA adapter configuration not found.")

    if not (adapter_dir / "adapter_model.safetensors").exists():
        raise FileNotFoundError("LoRA adapter weights not found.")

    # Load the tokenizer and pretrained GPT-2.
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base_model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model.config.pad_token_id = tokenizer.pad_token_id

    # Load the trained LoRA adapter.
    peft_model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
    )

    # Merge adapter weights into the base model.
    merged_model = peft_model.merge_and_unload()
    merged_model.eval()

    # Save the standalone model and tokenizer.
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_model.save_pretrained(
        output_dir,
        safe_serialization=True,
    )
    tokenizer.save_pretrained(output_dir)

    print("\nLoRA adapter merged successfully!")
    print("Merged model saved at:", output_dir)

    return {
        "base_model": model_name,
        "adapter_path": str(adapter_dir),
        "merged_model_path": str(output_dir),
    }


def milestone_5():
    """Execute the adapter merging milestone."""
    return merge_lora_adapter()
