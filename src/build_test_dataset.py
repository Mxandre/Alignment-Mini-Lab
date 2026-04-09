from __future__ import annotations
from pathlib import Path
from datasets import DatasetDict, Dataset
from typing import Any
import json
import argparse


DEFAULT_TEST_PATH = "data/test.jsonl"
def _normalilze_text(value: Any, field_name: str, line_number: int) -> str:
    if not isinstance(value, str) : 
        raise ValueError(
            f"Line{line_number}, field{field_name}: content must be string",
            f"get {type(value).__name__}"
        )
    normalize_value = value.strip()
    if not normalize_value :
        raise ValueError(
            f"Line {line_number}, field {field_name}: content cannot be empty"
        )
    return normalize_value
    

def _load_test_records(
        test_path : Path | str = DEFAULT_TEST_PATH
):
    records : list[dict[str, Any]] = []
    file_path = Path(test_path)
    with file_path.open("r", encoding = "utf-8") as file:
        for test_line, content in enumerate(file):
            line = content.strip()
            if not line:
                continue
            try: 
                payload = json.loads(content)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Line {test_line}: invalid JSON in {file_path}."
                ) from exc
            normalize_prompt = _normalilze_text(payload["prompt"], "prompt", test_line)
            normalized_record = dict(payload)
            normalized_record["prompt"] = normalize_prompt
            records.append(normalized_record)
    return records
    
def build_test_split(
    test_path : Path | str = DEFAULT_TEST_PATH
):
    return Dataset.from_list(_load_test_records(test_path))

def build_test_dataset(
    test_path: Path | str  = DEFAULT_TEST_PATH
):
    return DatasetDict(
        {"test" : build_test_split(test_path)}
    )


def main():
    parser = argparse.ArgumentParser(
        description="build the test files"
    )
    parser.add_argument("--test_path", default=DEFAULT_TEST_PATH)
    args  = parser.parse_args()
    test_dataset = build_test_dataset(args.test_path)
    test_dataset.save_to_disk("data/test_dataset")

if __name__ == "__main__" :
    main()