"""Publish the validated dataset to the training_data/ folder (format.jsonl shape).

Writes, into the separate top-level training_data/ directory:
  - dataset.jsonl  : all validated records (same structure as format.jsonl)
  - train.jsonl    : deterministic 90% split
  - val.jsonl      : deterministic 10% split

Usage: python split_dataset.py [--val-frac 0.1]
"""
from __future__ import annotations

import argparse

from common import OUTPUT_DIR, TRAINING_DIR, read_jsonl, write_jsonl

VALID_PATH = OUTPUT_DIR / "dataset.valid.jsonl"
DATASET_OUT = TRAINING_DIR / "dataset.jsonl"
TRAIN_PATH = TRAINING_DIR / "train.jsonl"
VAL_PATH = TRAINING_DIR / "val.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    rows = read_jsonl(VALID_PATH)
    # Full validated dataset, in the same format as format.jsonl.
    write_jsonl(DATASET_OUT, rows)

    # Deterministic interleaved split: every Nth example -> val.
    step = max(2, round(1 / args.val_frac))
    val = [r for i, r in enumerate(rows) if i % step == 0]
    train = [r for i, r in enumerate(rows) if i % step != 0]

    write_jsonl(TRAIN_PATH, train)
    write_jsonl(VAL_PATH, val)
    print(f"[split] validated {len(rows)} -> {DATASET_OUT}")
    print(f"[split] train {len(train)} ({TRAIN_PATH}), val {len(val)} ({VAL_PATH})")


if __name__ == "__main__":
    main()
