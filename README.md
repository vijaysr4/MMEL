# MMEL: Multimodal Entity Linking with VLMs on WikiData

> **Research Project**  
> Multimodal Entity Linking with Vision and Text on WikiData  
> Supervisor: Dr. Mehwish Alam  
> December 2024 – Ongoing

---

## Project Overview

This repository contains code and data for **MMEL**, a research project exploring how to link visual and textual content to WikiData entities (QIDs). We:

1. **Fine-tune** two vision-language models (CLIP and LLAVA NEXT) on paired images & text.  
2. **Map** learned embeddings to WikiData Knowledge Graph by computing QID similarities.  
3. **Extend** the pipeline to multi-entity images and videos.  
4. **Analyze** “dark spots” in VLM predictions—masking behaviors on sensitive or under-represented data.

---

## Repository Structure

```text
MMEL/
├── Data/                        # Raw & preprocessed datasets
│   ├── images/                  # Image assets
│   └── text/                    # Text captions / labels
│
├── fine_tune_models/            # Scripts & configs to fine-tune CLIP, LLAVA NEXT
│   ├── train_clip.py
│   └── train_llava.py
│
├── lava_predictions/            # Stored inference outputs from LLAVA NEXT
│
├── test_and_vis/                # Evaluation scripts & visualization notebooks
│
├── .gitignore
├── dataset_analysis.py          # Exploratory data analysis
├── embedding_similarity.py      # QID similarity mapping
├── llava_vlm.py                 # LLAVA NEXT wrapper & helper functions
├── model.py                     # Unified model definitions
├── read_predictions.py          # Loader for prediction files
├── test.py                      # End-to-end pipeline runner
├── utils.py                     # Common utilities
├── vlm_models.py                # CLIP / other VLM wrappers
└── wikidata_linking.py          # Core linking logic to WikiData KG
```
