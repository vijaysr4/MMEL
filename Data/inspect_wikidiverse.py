#!/usr/bin/env python3
"""
analyze_original_wikidiverse.py

Inspect the raw WikiDiverse dataset: report how many image references in annotated_data_V2.zip
actually exist under wikinewsImgs/, list missing filenames, and show a sample of entries.
"""
import os
import json
import argparse
from zipfile import ZipFile
from collections import Counter

from urllib.parse import unquote


def analyze_split(annotated_zip: str, split: str, imgs_dir: str):
    """
    For a given split, count total entries, missing image files, and sample raw records.

    Returns dict with counts and lists.
    """
    missing_files = []
    total = 0
    samples = []
    with ZipFile(annotated_zip, 'r') as z:
        # find JSON inside zip
        name = next(n for n in z.namelist() if n.endswith(f"{split}.json"))
        raw = json.loads(z.read(name))

    for caption, img_url, topic, mentions in raw:
        total += 1
        # decode URL-encoded name
        img_name = unquote(os.path.basename(img_url).split('?',1)[0])
        img_path = os.path.join(imgs_dir, img_name)
        if not os.path.exists(img_path):
            missing_files.append(img_name)
        if len(samples) < 5:
            samples.append({
                'caption': caption,
                'img_name': img_name,
                'topic': topic,
                'mentions': mentions
            })
    return {
        'total_images': total,
        'missing_count': len(missing_files),
        'missing_files': missing_files,
        'samples': samples
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True,
                        help='Path to WikiDiverse root (contains annotated_data_V2.zip and wikinewsImgs/)')
    args = parser.parse_args()

    data_dir = args.data_dir
    annotated_zip = os.path.join(data_dir, 'annotated_data_V2.zip')
    imgs_dir = os.path.join(data_dir, 'wikinewsImgs')

    results = {}
    for split in ('train','valid','test'):
        res = analyze_split(annotated_zip, split, imgs_dir)
        results[split] = res

    # Print summary
    for split, res in results.items():
        print(f"=== {split.upper()} ===")
        print(f"Total examples: {res['total_images']}")
        print(f"Missing images: {res['missing_count']}")
        if res['missing_count'] > 0:
            # show 10 most common missing names
            freq = Counter(res['missing_files'])
            common = freq.most_common(10)
            print("Most common missing filenames:")
            for name, cnt in common:
                print(f"  {name}: {cnt}")
        print("Sample entries:")
        for s in res['samples']:
            print(json.dumps(s, ensure_ascii=False, indent=2))
        print()

if __name__ == '__main__':
    main()
