#!/usr/bin/env python3
"""
DeBERTa-v3-large training script with local caching
Uses locally cached models and datasets for fast, offline training
"""

import os
import json
import math
import random
import argparse
import warnings
from pathlib import Path

# Set environment variables for DeBERTa-v3-large compatibility
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_TOKEN"] = "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"
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # Enable offline mode

from typing import List, Dict, Any
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoConfig,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from sklearn.metrics import f1_score, precision_recall_fscore_support
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GoEmotions labels
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]

class JsonlMultiLabelDataset(Dataset):
    """Dataset for multi-label classification from JSONL files"""
    
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        with open(jsonl_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        labels = item['labels']
        
        # Ensure labels is a list of integers (emotion indices)
        if isinstance(labels, int):
            labels = [labels]
        elif not isinstance(labels, list):
            labels = list(labels)
        
        # Convert to multi-label format (28-dimensional binary vector)
        label_vector = [0.0] * len(EMOTION_LABELS)
        for label_idx in labels:
            if 0 <= label_idx < len(EMOTION_LABELS):
                label_vector[label_idx] = 1.0
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_vector, dtype=torch.float)
        }

def compute_metrics_with_thresholds(eval_pred):
    """Compute metrics with multiple thresholds"""
    predictions, labels = eval_pred
    
    # Convert logits to probabilities
    probs = 1.0 / (1.0 + np.exp(-predictions))
    
    metrics = {}
    
    # Evaluate at multiple thresholds
    for threshold in [0.3, 0.5, 0.7]:
        preds = (probs >= threshold).astype(int)
        suffix = f"_t{int(threshold*10)}"
        
        # Compute metrics
        f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
        f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
        
        metrics[f"f1_micro{suffix}"] = f1_micro
        metrics[f"f1_macro{suffix}"] = f1_macro
        metrics[f"avg_preds{suffix}"] = preds.sum(axis=1).mean()
    
    # Primary metrics (using 0.3 threshold for better performance)
    metrics["f1_micro"] = metrics["f1_micro_t3"]
    metrics["f1_macro"] = metrics["f1_macro_t3"]
    
    return metrics

def load_model_and_tokenizer_local(model_type="deberta-v3-large"):
    """Load model and tokenizer from local cache or download if needed"""
    print(f"🤖 Loading {model_type}...")
    
    # Determine model path and name
    if model_type == "deberta-v3-large":
        model_path = "models/deberta-v3-large"
        model_name = "microsoft/deberta-v3-large"
    elif model_type == "roberta-large":
        model_path = "models/roberta-large"
        model_name = "roberta-large"
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Check if local cache exists
    if os.path.exists(model_path) and os.path.exists(f"{model_path}/config.json"):
        print(f"📁 Found local cache at {model_path}")
        try:
            # Load tokenizer from local cache
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=False,
                local_files_only=True
            )
            print(f"✅ {model_type} tokenizer loaded from local cache")
            
            # Load model from local cache
            config = AutoConfig.from_pretrained(
                model_path,
                num_labels=len(EMOTION_LABELS),
                problem_type="multi_label_classification"
            )
            
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                config=config
            )
            print(f"✅ {model_type} model loaded from local cache")
            
            return model, tokenizer
            
        except Exception as e:
            print(f"⚠️  Failed to load from local cache: {e}")
            print("🔄 Will download fresh copy...")
    
    # Download fresh copy with offline mode strategy
    print(f"🔄 Downloading {model_type} with offline mode strategy...")
    
    try:
        # Create directory
        os.makedirs(model_path, exist_ok=True)
        
        # Load tokenizer with offline mode
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            local_files_only=False  # Allow download
        )
        tokenizer.save_pretrained(model_path)
        print(f"✅ {model_type} tokenizer downloaded and cached")
        
        # Load model
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=len(EMOTION_LABELS),
            problem_type="multi_label_classification"
        )
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config
        )
        model.save_pretrained(model_path)
        print(f"✅ {model_type} model downloaded and cached")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Failed to download {model_type}: {e}")
        return None, None

def load_dataset_local():
    """Load GoEmotions dataset from local cache"""
    print("📊 Loading GoEmotions dataset from local cache...")
    
    train_path = "data/goemotions/train.jsonl"
    val_path = "data/goemotions/val.jsonl"
    
    # Check if local cache exists
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("❌ Local dataset cache not found")
        print("💡 Run 'python scripts/setup_local_cache.py' first")
        return None, None
    
    try:
        # Load metadata
        with open("data/goemotions/metadata.json", "r") as f:
            metadata = json.load(f)
        
        print(f"✅ GoEmotions dataset loaded from local cache")
        print(f"   Training examples: {metadata['train_size']}")
        print(f"   Validation examples: {metadata['val_size']}")
        print(f"   Total emotions: {len(metadata['emotions'])}")
        
        return train_path, val_path
        
    except Exception as e:
        print(f"❌ Failed to load dataset from local cache: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./outputs/deberta")
    parser.add_argument("--model_type", type=str, default="deberta-v3-large", 
                       choices=["deberta-v3-large", "roberta-large"])
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--max_length", type=int, default=512)
    
    args = parser.parse_args()
    
    print("🚀 GoEmotions DeBERTa Training (LOCAL CACHE VERSION)")
    print("="*60)
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🤖 Model: {args.model_type} (from local cache)")
    print(f"📊 Dataset: GoEmotions (from local cache)")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer from local cache
    model, tokenizer = load_model_and_tokenizer_local(args.model_type)
    if model is None or tokenizer is None:
        print("❌ Failed to load model. Exiting.")
        return
    
    # Load datasets from local cache
    train_path, val_path = load_dataset_local()
    if train_path is None or val_path is None:
        print("❌ Failed to load dataset. Exiting.")
        return
    
    # Create datasets
    print("🔄 Creating datasets...")
    train_dataset = JsonlMultiLabelDataset(train_path, tokenizer, args.max_length)
    val_dataset = JsonlMultiLabelDataset(val_path, tokenizer, args.max_length)
    
    print(f"✅ Created {len(train_dataset)} training examples")
    print(f"✅ Created {len(val_dataset)} validation examples")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        tf32=args.tf32,
        gradient_checkpointing=False,  # Disable to avoid distributed training issues
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        dataloader_drop_last=False,
        remove_unused_columns=False,
        report_to="none",  # Disable TensorBoard to avoid dependency issues
        ddp_find_unused_parameters=False,  # Optimize DDP performance
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_with_thresholds,
    )
    
    # Train
    print("🚀 Starting training...")
    trainer.train()
    
    # Save final model
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Evaluate
    print("📊 Final evaluation...")
    eval_results = trainer.evaluate()
    
    # Save evaluation results
    eval_report = {
        "model": args.model_type,
        "f1_micro": eval_results.get("eval_f1_micro", 0.0),
        "f1_macro": eval_results.get("eval_f1_macro", 0.0),
        "f1_micro_t3": eval_results.get("eval_f1_micro_t3", 0.0),
        "f1_macro_t3": eval_results.get("eval_f1_macro_t3", 0.0),
        "f1_micro_t5": eval_results.get("eval_f1_micro_t5", 0.0),
        "f1_macro_t5": eval_results.get("eval_f1_macro_t5", 0.0),
        "eval_loss": eval_results.get("eval_loss", 0.0),
        "training_args": vars(args)
    }
    
    with open(os.path.join(args.output_dir, "eval_report.json"), "w") as f:
        json.dump(eval_report, f, indent=2)
    
    print("✅ Training completed!")
    print(f"📈 Final F1 Macro: {eval_results.get('eval_f1_macro', 0.0):.4f}")
    print(f"📈 Final F1 Micro: {eval_results.get('eval_f1_micro', 0.0):.4f}")
    print(f"💾 Model saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
