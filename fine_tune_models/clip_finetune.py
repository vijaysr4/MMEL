import os
from datasets import load_dataset, Dataset
from typing import Dict, List
from PIL import Image
import numpy as np

# Set environment variables so that all Hugging Face caches go to /Data/MMEL
os.environ["HF_HOME"] = "/Data/MMEL/huggingface"  # changes location for hub-related files
os.environ["HF_DATASETS_CACHE"] = "/Data/MMEL/huggingface/datasets"  # for datasets specifically
os.environ["TRANSFORMERS_CACHE"] = "/Data/MMEL/huggingface/transformers"  # if you're using Transformers

# Specify cache directory
cache_path = "/Data/MMEL"
os.makedirs(cache_path, exist_ok=True)

# Load the dataset with the custom cache_dir
ds = load_dataset("aiintelligentsystems/vel_commons_wikidata",
                  "all_wikidata_items",
                  cache_dir=cache_path)
print(ds)
train = ds["train"]
print(train[0]["json"])