#!/usr/bin/env python3
"""
fill_wikidesc_with_progress.py

For each record in train.jsonl:
  - recover its wiki_url by matching back into annotated_data_V2.zip
  - if description == “No description available.”, fetch summary via Wikipedia REST API
  - write out train_filled.jsonl

Shows a tqdm progress bar over all input examples.
"""
import os
import json
import time
import hashlib
import re
import requests
from zipfile import ZipFile
from urllib.parse import urlparse, unquote
from tqdm import tqdm

# Paths
DATA_DIR = "/Data/MMEL/wiki_D"
IN_FILE  = os.path.join(DATA_DIR, "train.jsonl")
OUT_FILE = os.path.join(DATA_DIR, "train_filled.jsonl")
ANNOT    = os.path.join(DATA_DIR, "annotated_data_V2.zip")

# Build mention→wiki_url map
mention2url = {}
with ZipFile(ANNOT) as z:
    members = [m for m in z.namelist() if m.endswith(".json")]
    for member in members:
        raw = json.loads(z.read(member))
        for caption, img_url, topic, mentions in raw:
            for m_str, m_type, start, end, wurl in mentions:
                # key: (entity_str, image_url)
                mention2url[(m_str, img_url)] = wurl

# Wikipedia summary endpoint
API = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
session = requests.Session()
session.headers.update({"User-Agent": "WikiDiverseBot/1.0 (you@domain)"})

# Process input with a progress bar
with open(IN_FILE, 'r', encoding='utf-8') as fin, \
     open(OUT_FILE, 'w', encoding='utf-8') as fout:

    total = sum(1 for _ in open(IN_FILE, 'r', encoding='utf-8'))
    fin.seek(0)

    for line in tqdm(fin, total=total, desc="Filling descriptions"):
        ex = json.loads(line)
        # parse entity and image_url
        entity = ex["target"].split("\n",1)[0].split(":",1)[1].strip()
        img_url = None
        # recover original img_url by hashing local filename back to URL key
        local_fn = os.path.basename(ex["image_path"])
        # find the matching mention key
        for (m_str, url), wurl in mention2url.items():
            # hash that url → local name
            m_img  = url.split("/")[-1]
            prefix = hashlib.md5(m_img.encode("utf-8")).hexdigest()
            suffix = re.sub(r'(\S+(?=\.(jpg|jpeg|png|svg)))|(\S+(?=\.(JPG|JPEG|PNG|SVG)))','',m_img)
            if prefix+suffix == local_fn:
                img_url = url
                wiki_url = wurl
                break

        # only fill if no description available
        if "No description available." in ex["target"] and img_url and wiki_url:
            title = unquote(urlparse(wiki_url).path.split("/")[-1])
            try:
                r = session.get(API.format(title), timeout=10)
                r.raise_for_status()
                summary = r.json().get("extract","").split(".")[0] + "."
                ex["target"] = f"Entity: {entity}\nDescription: {summary}"
            except Exception:
                pass
            time.sleep(0.1)

        fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
