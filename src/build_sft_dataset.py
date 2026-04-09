"""Utilities for loading local SFT jsonl files into Hugging Face datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

DEFAULT_TRAIN_PATH = Path("data/sft_train.jsonl")
DEFAULT_EVAL_PATH = Path("data/sft_eval.jsonl")
REQUIRED_FIELDS = ("prompt", "response")


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


def _load_sft_records(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate one SFT split from a jsonl file."""
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
            response = _normalize_text(payload["response"], "response", line_number)

            normalized_record = dict(payload)
            normalized_record["prompt"] = prompt
            normalized_record["response"] = response
            normalized_record["completion"] = response
            records.append(normalized_record)

    if not records:
        raise ValueError(f"No usable records were found in {file_path}.")

    return records


def build_sft_split(path: str | Path) -> Dataset:
    """Build a single SFT dataset split from a jsonl file."""
    return Dataset.from_list(_load_sft_records(path))


def build_sft_dataset(
    train_path: str | Path = DEFAULT_TRAIN_PATH,
    eval_path: str | Path = DEFAULT_EVAL_PATH,
) -> DatasetDict:
    """Build a DatasetDict compatible with TRL's SFTTrainer."""
    return DatasetDict(
        {
            "train": build_sft_split(train_path),
            "eval": build_sft_split(eval_path),
        }
    )


def main() -> None:
    """CLI for quickly validating and inspecting the built SFT dataset."""
    parser = argparse.ArgumentParser(
        description="Build and validate SFT datasets from local jsonl files."
    )
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--eval-path", default=str(DEFAULT_EVAL_PATH))
    args = parser.parse_args()

    dataset_dict = build_sft_dataset(args.train_path, args.eval_path)
    print(dataset_dict)
    print("Train columns:", dataset_dict["train"].column_names)
    print("Eval columns:", dataset_dict["eval"].column_names)
    dataset_dict.save_to_disk("data/sft_dataset")


if __name__ == "__main__":
    main()
