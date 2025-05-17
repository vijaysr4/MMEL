#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Link LLaVA-generated entity names and descriptions back to Wikidata QIDs,
report top-1 accuracy, and display up to five candidate QIDs per example.

Input : output/llava_name_desc_1k.arrow
Output: output/llava_linked_1k.arrow
"""
from __future__ import annotations

import time
import pathlib
from typing import Dict, List, Optional

import requests
from datasets import load_from_disk, Dataset

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
CODE_DIR    = pathlib.Path(__file__).resolve().parent
INPUT_ARROW = CODE_DIR / "output" / "llava_name_desc_1k.arrow"
OUTPUT_ARROW= CODE_DIR / "output" / "llava_linked_1k.arrow"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
USER_AGENT   = "VL-EntityLinker/0.1 (vijay.murugan@ip-paris.fr)"
API_SLEEP    = 0.1      # seconds between requests
TOP_N        = 1        # Top-1 for primary accuracy
CANDIDATE_N  = 5        # Number of candidates to display per example

# ──────────────────────────────────────────────────────────────────────────────
# Wikidata search helpers
# ──────────────────────────────────────────────────────────────────────────────
def wikidata_search(query: str, limit: int) -> List[Dict]:
    """
    Query the Wikidata wbsearchentities API.
    Returns list of hits, each with keys 'id', 'label', 'description'.
    """
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": "en",
        "search": query,
        "limit": str(limit),
    }
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(WIKIDATA_API, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json().get("search", [])
    except requests.RequestException:
        return []

def link_entity(name: Optional[str], description: Optional[str]) -> Optional[str]:
    """
    Attempt to predict a QID given name and description.
    1) Search by name
    2) Search by description
    3) Search by "name description"
    """
    if name:
        hits = wikidata_search(name, limit=TOP_N)
        if hits:
            return hits[0]["id"]
    if description:
        hits = wikidata_search(description, limit=TOP_N)
        if hits:
            return hits[0]["id"]
    if name and description:
        hits = wikidata_search(f"{name} {description}", limit=TOP_N)
        if hits:
            return hits[0]["id"]
    return None

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    # load LLaVA outputs
    ds = load_from_disk(INPUT_ARROW)

    predictions: List[Dict] = []
    correct = 0

    # link each example
    for row in ds:
        true_qid = row["true_qid"]
        name      = row["entity_name"]
        desc      = row["entity_description"]

        pred_qid = link_entity(name, desc)
        predictions.append({
            "true_qid": true_qid,
            "pred_qid": pred_qid,
            "entity_name": name,
            "entity_description": desc,
        })
        if pred_qid == true_qid:
            correct += 1
        time.sleep(API_SLEEP)

    # report top-1 accuracy
    total = len(predictions)
    accuracy = correct / total if total else 0.0
    print(f"Linked {total} entities — Top-1 accuracy: {accuracy:.2%}")

    # display up to CANDIDATE_N candidates per example
    print("\nCandidate QIDs per example (up to 5):")
    for rec in predictions:
        name = rec["entity_name"] or ""
        true = rec["true_qid"]
        hits = wikidata_search(name, limit=CANDIDATE_N)
        qids = [h["id"] for h in hits]
        print(f"True: {true} | Name: {name}")
        print("  →", qids or ["(no candidates)"])

    # persist predictions
    Dataset.from_list(predictions).save_to_disk(OUTPUT_ARROW)
    print(f"\nPredictions saved to {OUTPUT_ARROW}")

if __name__ == "__main__":
    main()
