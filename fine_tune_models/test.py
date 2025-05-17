#!/usr/bin/env python3
# finetune_llava_jsonl.py
#
# Fine-tune LLaVA-1.5 (HF format) on a JSONL file that contains:
#   { "image_path": "...", "prompt": "...", "target": "..." }
#
# The script
#   • loads the JSONL, splits into train / valid
#   • handles images with PIL
#   • fixes `num_image_tokens` so patch-token count matches the vision tower
#   • supports Q-LoRA (4-bit) or full-precision training
#   • trains with Lightning + gradient accumulation
#   • saves adapters / processor to --output_dir

import os
import json
import random
import argparse
import warnings
from typing import List, Dict, Any

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from nltk import edit_distance

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)

# ------------- dataset -------------------------------------------------
class JsonlDataset(Dataset):
    def __init__(self, records: List[Dict[str, str]]) -> None:
        self.recs = records

    def __len__(self) -> int:                        # type: ignore[override]
        return len(self.recs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:   # type: ignore[override]
        r = self.recs[idx]
        return {
            "image":  Image.open(r["image_path"]).convert("RGB"),
            "prompt": r["prompt"],
            "target": r["target"],
        }

# ------------- collate -------------------------------------------------
def _format(batch: List[Dict[str, Any]], training: bool) -> List[str]:
    if training:
        return [
            f'USER: <image>\n{b["prompt"]}\nASSISTANT: {b["target"]}'
            for b in batch
        ]
    return [f'USER: <image>\n{b["prompt"]}\nASSISTANT:' for b in batch]


def collate_fn(batch: List[Dict[str, Any]],
               proc,
               max_len: int,
               training: bool):
    images = [b["image"] for b in batch]
    texts  = _format(batch, training)
    enc = proc(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )
    if training:
        labels = enc.input_ids.clone()
        labels[labels == proc.tokenizer.pad_token_id] = -100
        enc["labels"] = labels
        return enc
    gold = [b["target"] for b in batch]
    return enc, gold

# ------------- lightning module ---------------------------------------
class LitLLaVA(L.LightningModule):
    def __init__(self,
                 args,
                 processor,
                 model,
                 train_records: List[Dict[str, str]],
                 val_records:   List[Dict[str, str]]):
        super().__init__()
        self.save_hyperparameters(vars(args))
        self.p  = processor
        self.m  = model
        self.ml = args.max_len
        self.train_records = train_records
        self.val_records   = val_records

    # ---- train step ----
    def training_step(self, batch, _):
        loss = self.m(**batch).loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    # ---- val step ----
    def validation_step(self, batch_pair, _):
        enc, gold = batch_pair
        enc = {k: v.to(self.device) for k, v in enc.items()}

        out = self.m.generate(
            **enc,
            max_new_tokens=self.ml,
            do_sample=False,
        )

        preds = self.p.batch_decode(
            out[:, enc["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        score = np.mean(
            edit_distance(p, t) / max(len(p), len(t))
            for p, t in zip(preds, gold)
        )
        self.log("val_edit_distance", float(score), prog_bar=True)

    # ---- optimiser ----
    def configure_optimizers(self):
        return torch.optim.AdamW(self.m.parameters(), lr=self.hparams["lr"])

    # ---- dataloaders ----
    def train_dataloader(self):
        ds = JsonlDataset(self.train_records)
        return DataLoader(
            ds,
            batch_size=self.hparams["batch_size"],
            shuffle=True,
            num_workers=4,
            collate_fn=lambda b: collate_fn(b, self.p, self.ml, True),
        )

    def val_dataloader(self):
        ds = JsonlDataset(self.val_records)
        return DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            collate_fn=lambda b: collate_fn(b, self.p, self.ml, False),
        )

# ------------- helpers -------------------------------------------------
def read_jsonl(path: str) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]

def load_model(model_id: str, qlora: bool):
    if qlora:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        base = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            quantization_config=bnb_cfg,
            trust_remote_code=True,
        )
    else:
        base = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    # ---- FIX: align num_image_tokens with vision tower ----
    vt_cfg     = base.get_vision_tower().config
    num_tokens = (vt_cfg.image_size // vt_cfg.patch_size) ** 2    # 576 for 224×224
    base.config.num_image_tokens = num_tokens        # new name
    base.config.image_token_len  = num_tokens        # legacy
    # --------------------------------------------------------

    base.gradient_checkpointing_enable()
    base = prepare_model_for_kbit_training(base)
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    return get_peft_model(base, lora_cfg)

# ------------- main ----------------------------------------------------
def main():
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--model_id", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--output_dir", default="./outputs_llava")
    args = parser.parse_args()

    records = read_jsonl(args.jsonl)
    random.shuffle(records)
    cut   = max(1, int(len(records) * args.val_frac))
    val_r = records[:cut]
    tr_r  = records[cut:]

    processor = AutoProcessor.from_pretrained(args.model_id)
    processor.tokenizer.padding_side = "right"

    model = load_model(args.model_id, args.use_qlora)

    lit_mod = LitLLaVA(args, processor, model, tr_r, val_r)
    wandb_logger = WandbLogger(project="llava_jsonl_ft", name="run")
    stopper = EarlyStopping(monitor="val_edit_distance", patience=2, mode="min")

    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        accumulate_grad_batches=args.accum,
        max_epochs=args.epochs,
        logger=wandb_logger,
        callbacks=[stopper],
        num_sanity_val_steps=2,
    )
    trainer.fit(lit_mod)

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
