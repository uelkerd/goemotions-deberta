#!/usr/bin/env python3
"""
Quick training test to check if model is learning
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import f1_score

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]

class QuickDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_samples=100):
        self.tokenizer = tokenizer
        self.data = []

        import json
        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                item = json.loads(line.strip())
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        labels = item['labels']

        if isinstance(labels, int):
            labels = [labels]

        label_vector = [0.0] * len(EMOTION_LABELS)
        for label_idx in labels:
            if 0 <= label_idx < len(EMOTION_LABELS):
                label_vector[label_idx] = 1.0

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_vector, dtype=torch.float)
        }

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    probs = 1.0 / (1.0 + np.exp(-predictions))
    preds = (probs > 0.5).astype(int)

    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    f1_micro = f1_score(labels, preds, average='micro', zero_division=0)

    return {
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'avg_preds': preds.sum(axis=1).mean()
    }

def test_training():
    print("🚀 QUICK TRAINING TEST")
    print("=" * 50)

    # Load components
    tokenizer = AutoTokenizer.from_pretrained("models/deberta-v3-large", use_fast=False, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        "models/deberta-v3-large",
        num_labels=len(EMOTION_LABELS),
        problem_type="multi_label_classification"
    )

    # Create small datasets
    train_dataset = QuickDataset("data/goemotions/train.jsonl", tokenizer, max_samples=200)
    val_dataset = QuickDataset("data/goemotions/val.jsonl", tokenizer, max_samples=50)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Training arguments - very simple
    training_args = TrainingArguments(
        output_dir="./quick_test_output",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        learning_rate=1e-5,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=20,
        save_steps=100,
        save_total_limit=1,
        report_to="none",
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("📊 BEFORE TRAINING - Baseline metrics:")
    eval_results = trainer.evaluate()
    print(f"   F1 Macro: {eval_results.get('eval_f1_macro', 0):.4f}")
    print(f"   F1 Micro: {eval_results.get('eval_f1_micro', 0):.4f}")
    print(f"   Avg Preds: {eval_results.get('eval_avg_preds', 0):.2f}")

    print("\n🏋️ STARTING TRAINING...")
    train_result = trainer.train()

    print("📊 AFTER TRAINING - Final metrics:")
    eval_results = trainer.evaluate()
    print(f"   F1 Macro: {eval_results.get('eval_f1_macro', 0):.4f}")
    print(f"   F1 Micro: {eval_results.get('eval_f1_micro', 0):.4f}")
    print(f"   Avg Preds: {eval_results.get('eval_avg_preds', 0):.2f}")
    print(f"   Training Loss: {train_result.training_loss:.4f}")

    # Check if model learned anything
    if eval_results.get('eval_f1_macro', 0) > 0.1:
        print("✅ Model shows signs of learning")
    else:
        print("❌ Model does not appear to be learning")

if __name__ == "__main__":
    test_training()