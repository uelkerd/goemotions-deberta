#!/usr/bin/env python3
"""
Fixed training script for DeBERTa-v3-large using offline mode
Based on successful Strategy 2 from our testing
"""

import os
import json
import math
import random
import argparse
import warnings

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
from peft import LoraConfig, get_peft_model, TaskType
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
            'labels': torch.tensor(labels, dtype=torch.float)
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

def load_deberta_v3_model_and_tokenizer():
    """Load DeBERTa-v3-large using the working offline approach"""
    print("🤖 Loading DeBERTa-v3-large with offline mode...")
    
    # Local cache path
    local_path = "/workspace/.hf_home/hub/models--microsoft--deberta-v3-large/snapshots/64a8c8eab3e352a784c658aef62be1662607476f"
    
    try:
        if os.path.exists(local_path):
            print(f"✅ Loading from local cache: {local_path}")
            tokenizer = AutoTokenizer.from_pretrained(
                local_path,
                use_fast=False,
                local_files_only=True
            )
            print("✅ Tokenizer loaded from local cache")
        else:
            print("❌ Local cache not found")
            return None, None
        
        # Load model
        config = AutoConfig.from_pretrained(
            local_path,
            num_labels=len(EMOTION_LABELS),
            problem_type="multi_label_classification"
        )
        
        model = AutoModelForSequenceClassification.from_pretrained(
            local_path,
            config=config
        )
        print("✅ DeBERTa-v3-large model loaded successfully!")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Failed to load DeBERTa-v3-large: {e}")
        return None, None

def setup_lora(model):
    """Setup LoRA for efficient fine-tuning"""
    print("🔧 Setting up LoRA configuration...")
    
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=["query_proj", "key_proj", "value_proj", "dense"],
        modules_to_save=["classifier", "score"]
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./deberta_out")
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
    parser.add_argument("--ddp_backend", type=str, default="nccl")
    parser.add_argument("--max_length", type=int, default=512)
    
    args = parser.parse_args()
    
    print("🚀 DeBERTa-v3-large GoEmotions Training (FIXED VERSION)")
    print("="*60)
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🤖 Model: DeBERTa-v3-large (using offline mode)")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = load_deberta_v3_model_and_tokenizer()
    if model is None or tokenizer is None:
        print("❌ Failed to load DeBERTa-v3-large. Exiting.")
        return
    
    # Setup LoRA
    model = setup_lora(model)
    
    # Load datasets
    print("📊 Loading datasets...")
    train_dataset = JsonlMultiLabelDataset("./samo_out/train.jsonl", tokenizer, args.max_length)
    val_dataset = JsonlMultiLabelDataset("./samo_out/val.jsonl", tokenizer, args.max_length)
    
    print(f"✅ Loaded {len(train_dataset)} training examples")
    print(f"✅ Loaded {len(val_dataset)} validation examples")
    
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
        gradient_checkpointing=args.gradient_checkpointing,
        ddp_backend=args.ddp_backend,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        dataloader_drop_last=False,
        remove_unused_columns=False,
        report_to="tensorboard",
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
        "model": "microsoft/deberta-v3-large",
        "f1_micro": eval_results.get("eval_f1_micro", 0.0),
        "f1_macro": eval_results.get("eval_f1_macro", 0.0),
        "f1_micro_t3": eval_results.get("eval_f1_micro_t3", 0.0),
        "f1_macro_t3": eval_results.get("eval_f1_macro_t3", 0.0),
        "f1_micro_t5": eval_results.get("eval_f1_micro_t5", 0.0),
        "f1_macro_t5": eval_results.get("eval_f1_macro_t5", 0.0),
        "eval_loss": eval_results.get("eval_loss", 0.0),
    }
    
    with open(os.path.join(args.output_dir, "eval_report.json"), "w") as f:
        json.dump(eval_report, f, indent=2)
    
    print("✅ Training completed!")
    print(f"📈 Final F1 Macro: {eval_results.get('eval_f1_macro', 0.0):.4f}")
    print(f"📈 Final F1 Micro: {eval_results.get('eval_f1_micro', 0.0):.4f}")
    print(f"💾 Model saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
