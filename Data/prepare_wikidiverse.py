#!/usr/bin/env python3
"""
prepare_wikidiverse.py

— Reads all three splits (train.json, valid.json, test.json) directly from annotated_data_V2.zip.
— Maps every mention‐level example to its downloaded image in wikinewsImgs/ (MD5‑prefix naming).
— Loads both the filtered entity descriptions and the original Wikipedia plain descriptions,
  then merges them so that whenever the filtered description is missing, we use the plain one.
— Emits a single merged train.jsonl containing every example with the best available description.
"""
import os
import json
import argparse
import re
import hashlib
from zipfile import ZipFile
from typing import Dict, List

from PIL import Image

def load_entity_descriptions(desc_path: str) -> Dict[str, str]:
    """
    Load filtered entity descriptions from entity2desc_filtered.txt.
    Each line: "<wiki_url>@@@@<description>"
    """
    descriptions: Dict[str, str] = {}
    with open(desc_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            url, desc = line.split('@@@@', 1)
            descriptions[url] = desc
    return descriptions

def load_full_wikipedia_info(tsv_path: str) -> Dict[str, str]:
    """
    Load the original Wikipedia info TSV (split with '@@@@').
    Fields:
      0: annotated description
      1: plain description  ← we want this
      4: entity URL         ← use this as key
    """
    mapping: Dict[str, str] = {}
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split('@@@@')
            if len(parts) < 5:
                continue
            plain_desc = parts[1]
            entity_url = parts[4]
            mapping[entity_url] = plain_desc
    return mapping

def map_wikinews_url_to_local(img_url: str, imgs_dir: str) -> str:
    """
    Map a Wikinews image URL to its local MD5‑hashed filename under imgs_dir.
    """
    m_img = img_url.split("/")[-1]
    prefix = hashlib.md5(m_img.encode("utf-8")).hexdigest()
    suffix = re.sub(
        r'(\S+(?=\.(jpg|JPG|png|PNG|svg|SVG)))|(\S+(?=\.(jpeg|JPEG)))',
        "",
        m_img
    )
    local_name = (prefix + suffix).replace(".svg", ".png").replace(".SVG", ".png")
    return os.path.join(imgs_dir, local_name)

def collect_all_examples(
    annotated_zip: str,
    imgs_dir: str,
    entity2desc: Dict[str, str]
) -> List[Dict[str, str]]:
    """
    Read all three splits inside annotated_zip and return a combined list of examples.
    """
    all_samples: List[Dict[str, str]] = []
    with ZipFile(annotated_zip, "r") as z:
        for member in z.namelist():
            if not member.endswith(("train.json", "valid.json", "test.json")):
                continue
            raw = json.loads(z.read(member))
            for caption, img_url, topic, mentions in raw:
                img_path = map_wikinews_url_to_local(img_url, imgs_dir)
                if not os.path.exists(img_path):
                    continue
                try:
                    Image.open(img_path).verify()
                except Exception:
                    continue

                for m_str, m_type, start, end, wiki_url in mentions:
                    desc = entity2desc.get(wiki_url, "No description available.")
                    prompt = (
                        "USER: <image>\n"
                        "Identify the entity and describe it.\n"
                        "ASSISTANT:"
                    )
                    target = f"Entity: {m_str}\nDescription: {desc}"
                    all_samples.append({
                        "image_path": img_path,
                        "prompt": prompt,
                        "target": target
                    })
    return all_samples

def write_jsonl(examples: List[Dict[str, str]], out_path: str) -> None:
    """Write list of dicts to a JSONL file."""
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", required=True,
        help="WikiDiverse root (annotated_data_V2.zip, entity2desc_filtered.txt, original_wikipedia_info.tsv, wikinewsImgs/)"
    )
    args = parser.parse_args()

    data_dir      = args.data_dir
    annotated_zip = os.path.join(data_dir, "annotated_data_V2.zip")
    filtered_txt  = os.path.join(data_dir, "entity2desc_filtered.txt")
    wikiinfo_tsv  = os.path.join(data_dir, "original_wikipedia_info.tsv")
    imgs_dir      = os.path.join(data_dir, "wikinewsImgs")
    out_file      = os.path.join(data_dir, "train.jsonl")

    # 1) load filtered descriptions
    filtered = load_entity_descriptions(filtered_txt)
    # 2) load full plain descriptions
    full_info = load_full_wikipedia_info(wikiinfo_tsv)
    # 3) merge: whenever filtered is missing or default, use full_info
    for url, plain in full_info.items():
        if filtered.get(url, "No description available.") == "No description available.":
            filtered[url] = plain
    entity2desc = filtered

    # 4) collect and write
    examples = collect_all_examples(annotated_zip, imgs_dir, entity2desc)
    write_jsonl(examples, out_file)
    print(f"Wrote {len(examples)} examples to {out_file}")

if __name__ == "__main__":
    main()
