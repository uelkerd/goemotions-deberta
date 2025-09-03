#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SAMO — GoEmotions Multi‑Label Trainer (2×3090‑ready)
"""
import os, json, math, random, argparse, warnings
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

# ------------------------------
# Utilities
# ------------------------------

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def enable_tf32():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# ------------------------------
# Dataset loader (JSONL with {text, labels})
# ------------------------------
class JsonlMultiLabelDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int):
        self.examples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                text = obj.get("text", None)
                labels = obj.get("labels", None)
                if text is None or labels is None:
                    continue
                self.examples.append({"text": text, "labels": labels})
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        encoding = self.tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(example["labels"], dtype=torch.float)
        }

if __name__ == "__main__":
    print("SAMO training script loaded successfully!")
