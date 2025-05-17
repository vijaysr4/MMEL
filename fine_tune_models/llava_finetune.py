#!/usr/bin/env python3
# finetune_llava_jsonl_final.py
"""Minimal, memory-friendly fine-tuning of LLaVA-1.5 on a JSONL file."""

from __future__ import annotations
import os, json, random, argparse, warnings, re
from typing import List, Dict, Any

import torch, lightning as L
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from nltk import edit_distance
import numpy as np
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

# ---------- prompt helpers ----------
_IMG = re.compile(r"<image>", re.I)
_USER= re.compile(r"^USER:\s*", re.I)
def norm_prompt(p:str)->str:
    txt = _USER.sub("", p)
    txt = _IMG.sub("", txt).strip()
    return f"<image> {txt}"

# ---------- dataset ----------
class JsonlDS(Dataset):
    def __init__(self, recs:List[Dict[str,str]], train:bool): self.recs, self.train = recs, train
    def __len__(self): return len(self.recs)
    def __getitem__(self, idx):
        r   = self.recs[idx]
        img = Image.open(r["image_path"]).convert("RGB")
        prompt = norm_prompt(r["prompt"])
        if self.train: prompt = f"{prompt} {r['target']}"
        return {"image":img, "prompt":prompt, "target":r["target"]}

# ---------- collate ----------
def collate_train(b, proc, maxlen):
    enc = proc(text=[x["prompt"] for x in b],
               images=[x["image"]  for x in b],
               padding=True,truncation=True,max_length=maxlen,return_tensors="pt")
    labels = enc.input_ids.clone(); labels[labels==proc.tokenizer.pad_token_id]=-100
    enc["labels"]=labels; return enc
def collate_eval(b, proc, maxlen):
    enc = proc(text=[x["prompt"] for x in b],
               images=[x["image"]  for x in b],
               padding=True,truncation=True,max_length=maxlen,return_tensors="pt")
    return enc, [x["target"] for x in b]

# ---------- lightning ----------
class PL(L.LightningModule):
    def __init__(self,args,proc,model,tr,val): super().__init__(); self.save_hyperparameters(vars(args))
    # stash
        self.p, self.m, self.ml = proc, model, args.max_length
        self.tr, self.val = tr, val
    def training_step(self,b,_): out=self.m(**b); self.log("loss",out.loss,prog_bar=True); return out.loss
    def validation_step(self,pair,_):
        enc,gold=pair; enc={k:v.to(self.device) for k,v in enc.items()}
        gen=self.m.generate(**enc,max_new_tokens=self.ml,do_sample=False)
        pr=self.p.batch_decode(gen[:,enc["input_ids"].shape[1]:],skip_special_tokens=True)
        self.log("val_ed",float(np.mean([edit_distance(p,t)/max(len(p),len(t))for p,t in zip(pr,gold)])),prog_bar=True)
    def configure_optimizers(self):return torch.optim.AdamW(self.m.parameters(),lr=self.hparams["lr"])
    def train_dataloader(self):
        return DataLoader(JsonlDS(self.tr,True),batch_size=self.hparams["batch_size"],shuffle=True,
                          num_workers=4,collate_fn=lambda b:collate_train(b,self.p,self.ml))
    def val_dataloader(self):
        return DataLoader(JsonlDS(self.val,False),batch_size=1,shuffle=False,
                          num_workers=2,collate_fn=lambda b:collate_eval(b,self.p,self.ml))

# ---------- util ----------
def read(fp): return [json.loads(l) for l in open(fp,encoding="utf-8")]
def build(model_id,use4):
    if use4:
        cfg=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16)
        base=LlavaForConditionalGeneration.from_pretrained(model_id,torch_dtype=torch.float16,quantization_config=cfg,trust_remote_code=True)
    else:
        base=LlavaForConditionalGeneration.from_pretrained(model_id,torch_dtype=torch.float16,trust_remote_code=True)
    base=prepare_model_for_kbit_training(base)
    return get_peft_model(base,LoraConfig(r=8,lora_alpha=8,lora_dropout=0.1,
                                          target_modules=["q_proj","k_proj","v_proj","o_proj"]))

# ---------- main ----------
def main():
    warnings.filterwarnings("ignore",category=UserWarning)
    ap=argparse.ArgumentParser()
    ap.add_argument("--jsonl",required=True); ap.add_argument("--model_id",required=True)
    ap.add_argument("--output_dir",default="./outputs")
    ap.add_argument("--batch_size",type=int,default=1)               # micro-batch
    ap.add_argument("--accum",type=int,default=8)                    # gradient accumulation
    ap.add_argument("--epochs",type=int,default=3); ap.add_argument("--lr",type=float,default=2e-4)
    ap.add_argument("--max_length",type=int,default=384)
    ap.add_argument("--val_frac",type=float,default=0.05)
    ap.add_argument("--use_qlora",action="store_true")
    args=ap.parse_args()

    recs=read(args.jsonl); random.shuffle(recs)
    cut=max(1,int(len(recs)*args.val_frac)); val,tr=recs[:cut],recs[cut:]

    proc=AutoProcessor.from_pretrained(args.model_id)
    proc.tokenizer.padding_side="right"
    proc.patch_size=14
    proc.vision_feature_select_strategy="default"
    proc.image_processor.size={"height":224,"width":224}   # <── shrink vision tokens

    model=build(args.model_id,args.use_qlora)
    module=PL(args,proc,model,tr,val)
    wandb=   WandbLogger(project="llava-wikidiverse",name="run",offline=True)
    stopper= EarlyStopping(monitor="val_ed",mode="min",patience=2)

    trainer=L.Trainer(accelerator="gpu",devices=1,precision="16-mixed",
                      max_epochs=args.epochs,logger=wandb,callbacks=[stopper],
                      accumulate_grad_batches=args.accum,
                      limit_val_batches=2,num_sanity_val_steps=0)
    trainer.fit(module)

    os.makedirs(args.output_dir,exist_ok=True)
    model.save_pretrained(args.output_dir); proc.save_pretrained(args.output_dir)

if __name__=="__main__": main()
