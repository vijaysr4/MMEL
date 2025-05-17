from datasets import load_from_disk
from collections import Counter
import textwrap

# path to the dataset written by the LLaVA pipeline
ds = load_from_disk("/Data/MMEL/llava_predictions")

print(ds)                                     # features + size

N = 5                                         # how many rows to preview
print(f"\n--- First {N} predictions ---")
for i in range(N):
    r = ds[i]
    print(f"[{i}] QID: {r['qid']}\n"
          f"     Name       : {r['predicted_name']}\n"
          f"     Description: {textwrap.fill(r['predicted_description'] or '', 72)}\n")

# simple quality stats
total     = len(ds)
named_cnt = sum(bool(n) for n in ds["predicted_name"])
print(f"Non‑empty names: {named_cnt}/{total} ({named_cnt/total:.1%})")

desc_len = Counter(len((d or "").split()) for d in ds["predicted_description"])
print("\nMost common description lengths (top 5):")
for length, freq in desc_len.most_common(5):
    print(f"{length:>2} words → {freq}")
