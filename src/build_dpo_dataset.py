"""Utilities for loading local DPO jsonl files into Hugging Face datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

DEFAULT_TRAIN_PATH = Path("data/dpo_train.jsonl")
DEFAULT_EVAL_PATH = Path("data/dpo_eval.jsonl")
REQUIRED_FIELDS = ("prompt", "chosen", "rejected")


def _normalize_text(value: Any, field_name: str, line_number: int) -> str:
    """Convert raw JSON values into stripped strings and validate empties."""
    if not isinstance(value, str):
        raise ValueError(
            f"Line {line_number}: field '{field_name}' must be a string, "
            f"got {type(value).__name__}."
        )

    normalized = value.strip()
    if not normalized:
        raise ValueError(f"Line {line_number}: field '{field_name}' is empty.")
    return normalized


def _load_dpo_records(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate one DPO split from a jsonl file."""
    records: list[dict[str, Any]] = []
    file_path = Path(path)

    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Line {line_number}: invalid JSON in {file_path}."
                ) from exc

            missing_fields = [field for field in REQUIRED_FIELDS if field not in payload]
            if missing_fields:
                missing = ", ".join(missing_fields)
                raise ValueError(
                    f"Line {line_number}: missing required field(s): {missing}."
                )

            prompt = _normalize_text(payload["prompt"], "prompt", line_number)
            chosen = _normalize_text(payload["chosen"], "chosen", line_number)
            rejected = _normalize_text(payload["rejected"], "rejected", line_number)

            if chosen == rejected:
                raise ValueError(
                    f"Line {line_number}: 'chosen' and 'rejected' must differ."
                )

            normalized_record = dict(payload)
            normalized_record["prompt"] = prompt
            normalized_record["chosen"] = chosen
            normalized_record["rejected"] = rejected
            records.append(normalized_record)

    if not records:
        raise ValueError(f"No usable records were found in {file_path}.")

    return records


def build_dpo_split(path: str | Path) -> Dataset:
    """Build a single DPO dataset split from a jsonl file."""
    return Dataset.from_list(_load_dpo_records(path))


def build_dpo_dataset(
    train_path: str | Path = DEFAULT_TRAIN_PATH,
    eval_path: str | Path = DEFAULT_EVAL_PATH,
) -> DatasetDict:
    """Build a DatasetDict compatible with TRL's DPOTrainer."""
    return DatasetDict(
        {
            "train": build_dpo_split(train_path),
            "eval": build_dpo_split(eval_path),
        }
    )


def main() -> None:
    """CLI for quickly validating and inspecting the built DPO dataset."""
    parser = argparse.ArgumentParser(
        description="Build and validate DPO datasets from local jsonl files."
    )
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--eval-path", default=str(DEFAULT_EVAL_PATH))
    args = parser.parse_args()

    dataset_dict = build_dpo_dataset(args.train_path, args.eval_path)
    print(dataset_dict)
    print("Train columns:", dataset_dict["train"].column_names)
    print("Eval columns:", dataset_dict["eval"].column_names)
    dataset_dict.save_to_disk("data/dpo_dataset")


if __name__ == "__main__":
    main()
