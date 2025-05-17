'''
@inproceedings{wang2022wikidiverse,
title={WikiDiverse: A Multimodal Entity Linking Dataset with Diversified Contextual Topics and Entity Types},
author={Wang, Xuwu and Tian, Junfeng and Gui, Min and Li, Zhixu and Wang, Rui and Yan, Ming and Chen, Lihan and Xiao, Yanghua},
booktitle={ACL},
year={2022}
}
'''

#!/usr/bin/env python3
"""
dataset_download.py: Download WikiDiverse dataset components into /Data/MMEL/wiki_D

This script uses gdown for Google Drive shared files and requests for generic URLs.
"""
import os
import requests

try:
    import gdown
except ImportError:
    raise ImportError("Please install gdown: pip install gdown")

# Mapping of output filenames to their download URLs
DOWNLOADS = {
    # Annotated data (passage-level)
    "annotated_data_V2.zip":
        "https://drive.google.com/uc?id=1jsoa994_8tW9X19pb1cISKrMG8hTwItv&export=download",

    # Data with retrieved 10 candidates
    "cands10_V2.zip":
        "https://drive.google.com/uc?id=1ATTF_AzYAnUlM1N84S_dtFu-y867CELY&export=download",

    # Filtered entity descriptions
    "entity2desc_filtered.txt":
        "https://drive.google.com/uc?id=1LKjcWrU6YdFfLX6iKi0cFKtyhf4t2bbe&export=download",

    # Original Wikipedia information (TSV)
    "wiki_original_info.tsv":
        "https://pan.quark.cn/s/d6a7b66efe21",

    # P(e|m) data
    "pem_data.txt":
        "https://drive.google.com/uc?id=1Ss9cGb5c3nZtfzJvbFV_0-lEk1USBAAb&export=download",

    # Wikipedia entity-image alignments
    "entity2imgURLs.txt":
        "https://drive.google.com/uc?id=1ukoThqll410GG3P0I7-29kg299OzYgOT&export=download",
}

# Base directory for downloads
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def download_gdrive(url, output_path):
    """
    Download a Google Drive file given its shareable link.
    """
    print(f"Downloading (gdrive): {url}\n  -> {output_path}")
    gdown.download(url, output_path, quiet=False)


def download_generic(url, output_path):
    """
    Download a file from a generic URL with streaming.
    """
    print(f"Downloading: {url}\n  -> {output_path}")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(output_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def main():
    base_dir = '/Data/MMEL/wiki_D'
    ensure_dir(base_dir)

    for fname, url in DOWNLOADS.items():
        out_path = os.path.join(base_dir, fname)
        # Skip if already downloaded
        if os.path.exists(out_path):
            print(f"Already exists, skipping: {out_path}")
            continue

        # Choose method based on domain
        if 'drive.google.com' in url:
            download_gdrive(url, out_path)
        else:
            download_generic(url, out_path)

    print("All downloads completed.")


if __name__ == '__main__':
    main()
