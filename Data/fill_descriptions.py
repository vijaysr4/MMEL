#!/usr/bin/env python3
"""
count_no_desc.py  – quick sanity check
"""
import os, json, argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True, help="JSONL file to inspect")
    args = p.parse_args()

    total = 0
    no_desc = 0
    with open(args.file, encoding="utf-8") as f:
        for line in f:
            total += 1
            if "No description available." in line:
                no_desc += 1
    print(f"{no_desc}/{total} records still have the fallback string.")

if __name__ == "__main__":
    main()
