#!/usr/bin/env python3
"""
inspect_train.py

Load train.jsonl and report:
  - total examples
  - how many images exist & are valid
  - prompt/target length statistics
  - a few sample records
"""
import os
import json
import argparse
from statistics import mean, median
from PIL import Image, UnidentifiedImageError

def load_examples(path, max_load=None):
    exs = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_load and i >= max_load:
                break
            exs.append(json.loads(line))
    return exs

def check_images(examples):
    missing = []
    unreadable = []
    for ex in examples:
        p = ex['image_path']
        if not os.path.exists(p):
            missing.append(p)
        else:
            try:
                Image.open(p).verify()
            except (UnidentifiedImageError, OSError):
                unreadable.append(p)
    return missing, unreadable

def prompt_target_stats(examples):
    p_lens = [len(ex['prompt']) for ex in examples]
    t_lens = [len(ex['target']) for ex in examples]
    return {
        'prompt': (min(p_lens), mean(p_lens), median(p_lens), max(p_lens)),
        'target': (min(t_lens), mean(t_lens), median(t_lens), max(t_lens)),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True,
                        help='Path to WikiDiverse root containing train.jsonl')
    parser.add_argument('--max_load', type=int, default=None,
                        help='Maximum number of examples to sample for stats')
    args = parser.parse_args()

    train_path = os.path.join(args.data_dir, 'train_filled.jsonl')
    examples = load_examples(train_path, max_load=args.max_load)
    total = len(examples)
    missing, unreadable = check_images(examples)
    stats = prompt_target_stats(examples)

    print(f"Total examples loaded: {total}")
    print(f"Images missing       : {len(missing)}")
    if missing:
        print("  First missing:", missing[0])
    print(f"Images unreadable    : {len(unreadable)}")
    if unreadable:
        print("  First unreadable:", unreadable[0])

    print("\nPrompt lengths (chars):")
    print(f"  min {stats['prompt'][0]}, mean {stats['prompt'][1]:.1f}, "
          f"median {stats['prompt'][2]}, max {stats['prompt'][3]}")
    print("Target lengths (chars):")
    print(f"  min {stats['target'][0]}, mean {stats['target'][1]:.1f}, "
          f"median {stats['target'][2]}, max {stats['target'][3]}")

    print("\nA few sample entries:")
    for ex in examples[:5]:
        print("IMAGE:", ex['image_path'])
        print("PROMPT:", ex['prompt'])
        print("TARGET:", ex['target'])
        print("-"*40)

if __name__ == '__main__':
    main()
