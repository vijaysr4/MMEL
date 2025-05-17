#!/usr/bin/env python
"""
Exploratory analysis of the raw LLaVA results stored in
<project_root>/output/llava_name_desc_1k.arrow.

The script is location‑independent: it resolves *project_root* as the parent
folder of this file's directory, so you can run it from anywhere.

Outputs
-------
* Prints basic dataset info, null counts, and a random sample.
* Displays the name‑length (token) distribution.
* Writes nothing back to disk.
"""
from __future__ import annotations

import pathlib
import random
from collections import Counter
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_from_disk

# ──────────────────────────────────────────────────────────────────────────────
# Paths – robust to where you call `python analyse_llava_output.py` from.
# ──────────────────────────────────────────────────────────────────────────────
# Path to the Arrow file relative to the current working directory
INFER_ARROW = pathlib.Path("output/llava_name_desc_1k.arrow")

if not INFER_ARROW.exists():
    raise FileNotFoundError(f"Expected file not found: {INFER_ARROW}")

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def token_count(text: str | None) -> int:
    return len(text.split()) if text else 0

# ──────────────────────────────────────────────────────────────────────────────
# Main analysis
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not INFER_ARROW.exists():
        raise FileNotFoundError(f"Expected file not found: {INFER_ARROW}")

    df = load_from_disk(str(INFER_ARROW)).to_pandas()

    print("\n── Dataset overview ─────────────────────────────────────")
    print(df.info(max_cols=0))

    null_name = df["entity_name"].isna().sum()
    null_desc = df["entity_description"].isna().sum()
    print(f"\nRows                : {len(df)}")
    print(f"Missing name fields : {null_name}")
    print(f"Missing descr fields: {null_desc}")

    lengths = [token_count(t) for t in df["entity_name"].fillna("")]
    length_counts = Counter(lengths)
    print("\nName length distribution (tokens):")
    for length, count in sorted(length_counts.items()):
        print(f"  {length:2d} tokens : {count}")

    # Histogram plot
    plt.figure(figsize=(6, 4))
    plt.hist(lengths, bins=range(0, max(lengths) + 2), edgecolor="black")
    plt.title("Name Length Distribution (tokens)")
    plt.xlabel("Tokens in entity_name")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    print("\n── Random sample of 10 entries ───────────────────────────")
    for _, row in df.sample(n=min(10, len(df)), random_state=42).iterrows():
        print(f"QID {row['true_qid']}: {row['entity_name'] or '[NO NAME]'}")
        print(f"   → {row['entity_description'] or '[NO DESCRIPTION]'}\n")


if __name__ == "__main__":
    main()
