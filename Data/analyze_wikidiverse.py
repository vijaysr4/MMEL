#!/usr/bin/env python3
"""
validate_wikidiverse.py

Validate the processed WikiDiverse JSONL splits under a data directory.

For each of train.jsonl, valid.jsonl, test.jsonl it checks:
  - JSON parse success
  - presence of required keys: image_path, prompt, target
  - image_path file exists
  - PIL can open the image
  - prompt and target are non-empty strings

Usage:
    python validate_wikidiverse.py --data_dir /Data/MMEL/wiki_D
"""
import os
import json
import argparse
from PIL import Image, UnidentifiedImageError

def validate_split(jsonl_path: str):
    total = 0
    parse_errors = 0
    missing_keys = 0
    missing_images = 0
    open_errors = 0
    empty_prompt = 0
    empty_target = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            total += 1
            line = line.rstrip('\n')
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue

            # Check keys
            if not all(k in ex for k in ('image_path','prompt','target')):
                missing_keys += 1
                continue

            # Check prompt/target non-empty
            if not ex['prompt'].strip():
                empty_prompt += 1
            if not ex['target'].strip():
                empty_target += 1

            # Check image exists
            img_path = ex['image_path']
            if not os.path.exists(img_path):
                missing_images += 1
            else:
                # check PIL can open
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                except (UnidentifiedImageError, OSError):
                    open_errors += 1

    return {
        'total': total,
        'parse_errors': parse_errors,
        'missing_keys': missing_keys,
        'empty_prompt': empty_prompt,
        'empty_target': empty_target,
        'missing_images': missing_images,
        'open_errors': open_errors
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True,
                        help='Path to WikiDiverse root containing train.jsonl, valid.jsonl, test.jsonl')
    args = parser.parse_args()

    for split in ('train','valid','test'):
        path = os.path.join(args.data_dir, f'{split}.jsonl')
        if not os.path.exists(path):
            print(f"{split}: file not found at {path}")
            continue

        stats = validate_split(path)
        print(f"=== {split} ===")
        print(f"Total lines          : {stats['total']}")
        print(f"JSON parse errors    : {stats['parse_errors']}")
        print(f"Examples missing keys: {stats['missing_keys']}")
        print(f"Empty prompts        : {stats['empty_prompt']}")
        print(f"Empty targets        : {stats['empty_target']}")
        print(f"Missing images       : {stats['missing_images']}")
        print(f"Unreadable images    : {stats['open_errors']}")
        print()

if __name__ == '__main__':
    main()
