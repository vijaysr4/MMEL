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

# Select a sample of 1000 records from the 'train' split (adjust if necessary)
sample_size = 1000
sample = ds["train"].select(range(sample_size))

print("Sample of the dataset:")
print(sample)


def check_jpg_presence(dataset: Dataset) -> Dict[int, List[str]]:
    """
    Check each record in a Hugging Face dataset for missing image data in the 'jpg' field.

    Returns a dictionary mapping record indices to missing fields.
    """
    missing_fields_dict: Dict[int, List[str]] = {}
    for idx, record in enumerate(dataset):
        missing_fields: List[str] = []
        if record.get("jpg") in (None, "", []):
            missing_fields.append("jpg")
        if missing_fields:
            missing_fields_dict[idx] = missing_fields
    return missing_fields_dict


# Check the sample for missing JPG images
missing_jpg = check_jpg_presence(sample)
if missing_jpg:
    print("\nRecords with missing JPG images:")
    for idx, fields in missing_jpg.items():
        print(f"Record {idx} is missing: {fields}")
else:
    print("\nAll records have JPG images available.")


# Define a filter function that removes image-text pairs where the image has ambiguous shape.
def filter_out_ambiguous(example):
    """
    Returns True for records whose JPG image does NOT have the ambiguous shape (1,1,3);
    returns False if the image is missing or has shape (1,1,3).
    """
    image_input = example.get("jpg", None)
    if image_input is None:
        # Consider missing image as not valid.
        return False
    # If image_input is a string, assume it's a file path and try to open the image.
    if isinstance(image_input, str):
        try:
            image = Image.open(image_input)
        except Exception as e:
            print(f"Error opening image for record {example.get('__key__')}: {e}")
            return False
    else:
        image = image_input

    # Ensure the image is in RGB mode.
    try:
        image = image.convert("RGB")
    except Exception as e:
        print(f"Error converting image for record {example.get('__key__')}: {e}")
        return False

    # Convert image to a NumPy array and check its shape.
    image_array = np.array(image)
    print(f"Record {example.get('__key__')} - Image array shape: {image_array.shape}")
    # Filter out images with shape exactly (1, 1, 3)
    if image_array.shape == (1, 1, 3):
        print(f"Record {example.get('__key__')} - Ignored due to ambiguous image shape (1,1,3).")
        return False
    return True


# Filter out records with ambiguous JPG image shape.
filtered_sample = sample.filter(filter_out_ambiguous)

# Further filter to keep only the 'jpg', '__key__', and 'json' columns.
# Original columns: ['jpg', 'json', 'npy', 'png', '__key__', '__url__']
filtered_sample = filtered_sample.remove_columns(['npy', 'png', '__url__'])

print("\nFiltered sample with only 'jpg', '__key__', and 'json' fields (and without ambiguous images):")
print(filtered_sample)

# Save the filtered dataset to disk.
output_dir = os.path.join(cache_path, "1000_filtered_vel_commons_wikidata")
filtered_sample.save_to_disk(output_dir)
print(f"\nFiltered sample saved to: {output_dir}")
