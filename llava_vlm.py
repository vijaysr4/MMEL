#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Infer (entity name, one-sentence description) for 1 000 Wikidata images with
LLaVA-1.5-7B and save the results to /output/llava_name_desc_1k.arrow.
"""

from __future__ import annotations

import io
import os
import pathlib
import shutil
from typing import Dict, List, Optional

import requests
import torch
from datasets import Dataset, Image as HFImage, load_dataset
from PIL import Image, UnidentifiedImageError
from transformers import AutoProcessor, LlavaForConditionalGeneration

# ───────────────────────────────
# config
# ───────────────────────────────
HF_ROOT = pathlib.Path("/Data/MMEL")
OUTPUT_DIR = pathlib.Path("output/llava_name_desc_1k.arrow")
DATASET_NAME = "aiintelligentsystems/vel_commons_wikidata"
DATASET_CONFIG = "all_wikidata_items"
SAMPLE_SIZE = 1_000
LLAVA_MODEL = "llava-hf/llava-1.5-7b-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHAT_PROMPT = (
    "USER: <image>\n"
    "Identify the main entity in this image. Respond strictly in this format:\n"
    "Name: <entity name>\n"
    "Description: <one-sentence description>\n"
    "ASSISTANT:"
)


# ───────────────────────────────
# helpers
# ───────────────────────────────
def set_hf_env(root: pathlib.Path) -> None:
    os.environ["HF_HOME"] = str(root / "huggingface")
    os.environ["HF_DATASETS_CACHE"] = str(root / "huggingface" / "datasets")
    os.environ["TRANSFORMERS_CACHE"] = str(root / "huggingface" / "transformers")
    root.mkdir(parents=True, exist_ok=True)


def to_pil(obj) -> Optional[Image.Image]:
    try:
        if obj is None:
            return None
        if isinstance(obj, Image.Image):
            return obj.convert("RGB")
        if isinstance(obj, (bytes, bytearray)):
            return Image.open(io.BytesIO(obj)).convert("RGB")
        if isinstance(obj, dict):
            if (b := obj.get("bytes")) is not None:
                return Image.open(io.BytesIO(b)).convert("RGB")
            if (p := obj.get("path")) is not None:
                return Image.open(p).convert("RGB")
    except UnidentifiedImageError:
        pass
    return None


def choose_image(row: Dict) -> Optional[Image.Image]:
    img = to_pil(row.get("png")) or to_pil(row.get("jpg"))
    if img is None:
        try:
            r = requests.get(row["__url__"], timeout=5)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
        except (requests.RequestException, UnidentifiedImageError):
            return None
    return None if img.size == (1, 1) else img


def infer_name_desc(
    img: Image.Image,
    processor: AutoProcessor,
    model: LlavaForConditionalGeneration,
) -> Dict[str, Optional[str]]:
    """
    Run LLaVA and return {'entity_name', 'entity_description'}.
    Uses explicit kwargs to avoid duplicate 'images' errors.
    """
    inputs = processor(
        text=CHAT_PROMPT,
        images=[img],              # wrap single image in a list
        return_tensors="pt",
    ).to(model.device)

    out_ids = model.generate(**inputs, max_new_tokens=80)
    text = processor.decode(out_ids[0], skip_special_tokens=True)

    name = desc = None
    for line in (l.strip() for l in text.splitlines() if l.strip()):
        if line.lower().startswith("name:"):
            name = line.split(":", 1)[1].strip()
        elif line.lower().startswith("description:"):
            desc = line.split(":", 1)[1].strip()
    return {"entity_name": name, "entity_description": desc}


def save_arrow(records: List[Dict], out_dir: pathlib.Path) -> None:
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).save_to_disk(out_dir)


# ───────────────────────────────
# main
# ───────────────────────────────
def main() -> None:
    set_hf_env(HF_ROOT)

    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, cache_dir=str(HF_ROOT))
    ds = ds.cast_column("png", HFImage(decode=True))
    ds = ds.cast_column("jpg", HFImage(decode=True))
    sample = ds["train"].select(range(SAMPLE_SIZE))

    print("Loading LLaVA-1.5-7B …")
    processor = AutoProcessor.from_pretrained(LLAVA_MODEL, use_fast=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_MODEL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    outputs: List[Dict] = []
    skipped = 0
    for idx, row in enumerate(sample, 1):
        img = choose_image(row)
        if img is None:
            skipped += 1
            continue

        outputs.append(
            {"true_qid": row["__key__"], **infer_name_desc(img, processor, model)}
        )

        if idx % 100 == 0:
            print(f"Scanned {idx}/{SAMPLE_SIZE}")

    print(f"Generated entries : {len(outputs):,}")
    print(f"Skipped entries   : {skipped:,}")
    save_arrow(outputs, OUTPUT_DIR)
    print(f"Saved dataset     : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

