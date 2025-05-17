import os, torch, numpy as np
from io import BytesIO
from PIL import Image
from datasets import load_from_disk
from transformers import AutoProcessor, LlavaForConditionalGeneration

# ------------------------------------------------------------------
# 1. Environment & dataset
# ------------------------------------------------------------------
os.environ["HF_HOME"] = "/Data/MMEL/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/Data/MMEL/huggingface/datasets"
os.environ["TRANSFORMERS_CACHE"] = "/Data/MMEL/huggingface/transformers"

dataset = load_from_disk("/Data/MMEL/1000_filtered_vel_commons_wikidata")
print(dataset)

# 2. Model & processor

model_name = "llava-hf/llava-1.5-7b-hf"
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model      = LlavaForConditionalGeneration.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
).to(device)
processor  = AutoProcessor.from_pretrained(model_name)
processor.tokenizer.padding_side = "left"        # small quality boost :contentReference[oaicite:0]{index=0}

# ------------------------------------------------------------------
# 3. Utility: make any “jpg” field a RGB PIL.Image

def to_pil(img_field):
    if isinstance(img_field, Image.Image):
        return img_field.convert("RGB")
    if isinstance(img_field, str):
        return Image.open(img_field).convert("RGB")
    if isinstance(img_field, (bytes, bytearray)):
        return Image.open(BytesIO(img_field)).convert("RGB")
    if isinstance(img_field, np.ndarray):
        return Image.fromarray(img_field).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(img_field)}")


# 4. Per‑example prediction
def predict_entity(example):
    try:
        img = to_pil(example["jpg"])
    except Exception as e:
        print("Bad image:", e)
        example["predicted_name"]        = None
        example["predicted_description"] = None
        return example

    # ---- STEP 1: build chat prompt string (NO images here) ----
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},  # placeholder only
                {
                    "type": "text",
                    "text": (
                        "Identify the main entity in the picture and give description "
                        "description.\n"
                        "Format strictly as:\n"
                        "Name: <entity name>\n"
                        "Description: <one‑sentence description>"
                    ),
                },
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True
    )                                        # returns plain string

    # ---- STEP 2: tokenise + encode with real image ----
    inputs = processor(
        text=prompt,
        images=img,
        return_tensors="pt",
        padding=True
    ).to(device, torch.float16)

    # ---- STEP 3: generate & decode ----
    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=64, num_beams=4)
    answer = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

    # ---- STEP 4: simple parsing ----
    name, desc = None, None
    if "Name:" in answer and "Description:" in answer:
        before, after = answer.split("Description:", 1)
        name = before.replace("Name:", "").strip()
        desc = after.strip()
    else:                                   # fallback
        desc = answer.strip()

    example["predicted_name"]        = name
    example["predicted_description"] = desc
    return example


# 5. Map over the whole dataset
dataset_with_preds = dataset.map(predict_entity, load_from_cache_file=False)

# keep only what you need
def strip_fields(ex):
    qid = ex.get("json", {}).get("qid", ex.get("__key__"))
    return {
        "qid": qid,
        "predicted_name": ex["predicted_name"],
        "predicted_description": ex["predicted_description"],
    }

out_ds = dataset_with_preds.map(strip_fields,
                                remove_columns=dataset_with_preds.column_names)

out_path = "lava_predictions"
out_ds.save_to_disk(out_path)
print("Saved predictions to:", out_path)
