#!/usr/bin/env python
"""
Visualise ground‑truth vs. predicted QIDs produced by `wikidata_linking.py`.

Inputs
------
/output/llava_linked_1k.arrow   # generated predictions with columns
                               #   true_qid, pred_qid, entity_name, entity_description

Output (side‑effects)
---------------------
* Displays a bar chart of correct vs. incorrect links.
* Prints a concise classification report.
* Saves a CSV of mis‑linked records to `/output/linked_errors.csv` for
  manual inspection.

Run
---
python visualize_groundtruth.py

Dependencies: matplotlib, pandas, datasets
"""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_from_disk
from sklearn.metrics import classification_report  # Only for text summary

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PRED_ARROW = pathlib.Path("../output/llava_linked_1k.arrow")
ERROR_CSV  = pathlib.Path("../output/linked_errors.csv")

# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main() -> None:
    # Load predictions
    ds = load_from_disk(str(PRED_ARROW))
    df = ds.to_pandas()

    # Compute correctness column
    df["is_correct"] = df["true_qid"] == df["pred_qid"]

    # Textual summary
    total   = len(df)
    correct = int(df["is_correct"].sum())
    accuracy = correct / total if total else 0.0
    print(f"Total samples     : {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy           : {accuracy:.2%}\n")

    # Classification report (sklearn treats each unique QID as a label)
    # We only show overall precision/recall/F1 via micro‑average.
    print(classification_report(
        y_true=df["true_qid"],
        y_pred=df["pred_qid"],
        output_dict=False,
        zero_division=0,
    ))

    # Plot correct vs. incorrect counts
    counts = df["is_correct"].value_counts().reindex([True, False], fill_value=0)
    plt.figure(figsize=(5, 4))
    counts.plot(kind="bar")
    plt.title("Entity Linking Results (Top‑1)")
    plt.ylabel("Number of Images")
    plt.xticks([0, 1], ["Correct", "Incorrect"], rotation=0)
    plt.tight_layout()
    plt.show()

    # Save mis‑linked records for manual review
    errors = df[~df["is_correct"]]
    errors.to_csv(ERROR_CSV, index=False)
    print(f"Mis‑linked records saved to {ERROR_CSV}")


if __name__ == "__main__":
    main()
