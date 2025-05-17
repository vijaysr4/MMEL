import os
from datasets import load_from_disk, Dataset
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import numpy as np

os.environ["HF_HOME"] = "/Data/MMEL/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/Data/MMEL/huggingface/datasets"
os.environ["TRANSFORMERS_CACHE"] = "/Data/MMEL/huggingface/transformers"

cache_path = "/Data/MMEL"
os.makedirs(cache_path, exist_ok=True)

dataset = load_from_disk("/Data/MMEL/1000_filtered_vel_commons_wikidata")
print("Loaded dataset:")
print(dataset)

model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def compute_clip_embeddings(example: dict) -> dict:
    json_data = example.get("json", {})
    entity_name = json_data.get("name", "")
    entity_description = json_data.get("description", "")
    text_input = f"{entity_name}. {entity_description}"

    image_input = example.get("jpg", None)
    if image_input is None:
        example["clip_image_embedding"] = None
    else:
        if isinstance(image_input, str):
            try:
                image_input = Image.open(image_input)
            except Exception as e:
                print(f"Error opening image for record {example.get('__key__')}: {e}")
                example["clip_image_embedding"] = None
                return example
        image_input = image_input.convert("RGB")

    inputs = processor(text=[text_input],
                       images=image_input,
                       return_tensors="pt",
                       padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    text_embedding = outputs.text_embeds[0].cpu().numpy().tolist()
    image_embedding = outputs.image_embeds[0].cpu().numpy().tolist() if image_input is not None else None
    example["clip_text_embedding"] = text_embedding
    example["clip_image_embedding"] = image_embedding
    return example


dataset_with_embeddings = dataset.map(compute_clip_embeddings, load_from_cache_file=False)


def select_required_fields(example: dict) -> dict:
    json_data = example.get("json", {})
    qid = json_data.get("qid", example.get("__key__"))
    return {
        "clip_text_embedding": example.get("clip_text_embedding"),
        "clip_image_embedding": example.get("clip_image_embedding"),
        "qid": qid
    }


final_dataset = dataset_with_embeddings.map(select_required_fields, remove_columns=dataset_with_embeddings.column_names)

output_path = "/Data/MMEL/fast_access_embeddings"
final_dataset.save_to_disk(output_path)
print(f"Fast-access dataset with embeddings saved to: {output_path}")
