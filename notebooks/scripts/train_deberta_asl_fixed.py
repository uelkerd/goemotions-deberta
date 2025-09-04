#!/usr/bin/env python3
"""
Fixed training script for DeBERTa-v3-large using Asymmetric Loss (ASL)
Combines fixed DeBERTa loading with ASL implementation for GoEmotions
"""

import os
import json
import math
import random
import argparse
import warnings
import datetime
from typing import List, Dict, Any
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoConfig,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import f1_score, precision_recall_fscore_support
import logging

# Set environment variables for DeBERTa-v3-large compatibility
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_TOKEN"] = "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"
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # Enable offline mode
os.environ["ACCELERATE_USE_CPU"] = "false"
os.environ["LOCAL_RANK"] = "0"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

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

# ------------------------------
# Asymmetric Loss Implementation
# ------------------------------

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification
    Optimized for imbalanced datasets like GoEmotions
    """
    def __init__(self, gamma_neg=2.0, gamma_pos=1.0, clip=0.05, eps=1e-8, pos_alpha=1.0):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.pos_alpha = pos_alpha

    def forward(self, x, y):
        """
        Args:
            x: logits from model (batch_size, num_labels)
            y: target labels (batch_size, num_labels)
        """
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Asymmetric focusing
        los_pos = self.pos_alpha * y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        loss = los_pos + los_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = xs_pos * y + xs_neg * (1 - y)
            gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            loss = loss * torch.pow(1 - pt, gamma)

        return -loss.mean()

# ------------------------------
# Custom Trainer with Asymmetric Loss
# ------------------------------

class ASLTrainer(Trainer):
    """
    Custom Trainer that uses Asymmetric Loss for multi-label classification
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize Asymmetric Loss with optimized parameters for GoEmotions
        self.loss_fct = AsymmetricLoss(
            gamma_neg=2.0,    # Focus more on hard negatives
            gamma_pos=1.0,    # Moderate focus on positives
            clip=0.05,        # Clip negative predictions
            pos_alpha=1.0     # Balanced positive weight
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override compute_loss to use Asymmetric Loss
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.loss_fct(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

# ------------------------------
# Dataset and Metrics
# ------------------------------

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

        # Convert GoEmotions format to multi-label vector
        if isinstance(labels, list) and len(labels) > 0:
            # GoEmotions has single-label format, convert to multi-label
            label_vector = [0.0] * len(EMOTION_LABELS)
            for label_idx in labels:
                if 0 <= label_idx < len(EMOTION_LABELS):
                    label_vector[label_idx] = 1.0
        else:
            # Fallback to neutral if no labels
            label_vector = [0.0] * len(EMOTION_LABELS)
            label_vector[-1] = 1.0  # neutral

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

def load_deberta_v3_model_and_tokenizer():
    """Load DeBERTa-v3-large using existing checkpoint to avoid torch vulnerability"""
    print("🤖 Loading DeBERTa-v3-large from existing checkpoint...")

    # Use the existing phase1_asymmetric checkpoint which has safetensors
    checkpoint_path = "./phase1_asymmetric"

    try:
        if os.path.exists(checkpoint_path):
            print(f"✅ Loading from checkpoint: {checkpoint_path}")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_path,
                use_fast=False,
                local_files_only=True
            )
            print("✅ Tokenizer loaded from checkpoint")

            # Load model from safetensors checkpoint
            model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint_path,
                local_files_only=True
            )
            print("✅ Model loaded from safetensors checkpoint!")

        else:
            print("❌ Checkpoint not found, trying original model...")
            # Fallback to original loading (will likely fail due to torch vulnerability)
            local_path = "/workspace/.hf_home/hub/models--microsoft--deberta-v3-large/snapshots/64a8c8eab3e352a784c658aef62be1662607476f"

            if os.path.exists(local_path):
                print(f"✅ Loading from local cache: {local_path}")
                tokenizer = AutoTokenizer.from_pretrained(
                    local_path,
                    use_fast=False,
                    local_files_only=True
                )
                print("✅ Tokenizer loaded from local cache")

                config = AutoConfig.from_pretrained(
                    local_path,
                    num_labels=len(EMOTION_LABELS),
                    problem_type="multi_label_classification"
                )

                model = AutoModelForSequenceClassification.from_pretrained(
                    local_path,
                    config=config
                )
                print("✅ DeBERTa-v3-large model loaded from cache!")
            else:
                print("❌ No model sources available")
                return None, None

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
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--ddp_backend", type=str, default="nccl")
    parser.add_argument("--max_length", type=int, default=256)

    args = parser.parse_args()

    print("🚀 DeBERTa-v3-large ASL Training (FIXED VERSION)")
    print("="*60)
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🤖 Model: DeBERTa-v3-large (using offline mode)")
    print(f"🎯 Loss: Asymmetric Loss (ASL)")
    print(f"📊 Batch size: {args.per_device_train_batch_size}")
    print(f"📈 Learning rate: {args.learning_rate}")
    print(f"🎯 Epochs: {args.num_train_epochs}")

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
    train_dataset = JsonlMultiLabelDataset("./data/goemotions/train.jsonl", tokenizer, args.max_length)
    val_dataset = JsonlMultiLabelDataset("./data/goemotions/val.jsonl", tokenizer, args.max_length)

    print(f"✅ Loaded {len(train_dataset)} training examples")
    print(f"✅ Loaded {len(val_dataset)} validation examples")

    # Training arguments - disable distributed training
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
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        save_steps=100,
        eval_steps=100,
        dataloader_drop_last=False,
        remove_unused_columns=False,
        report_to="tensorboard",
        dataloader_num_workers=0,
        local_rank=-1,  # Disable distributed training
    )

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ASL Trainer
    trainer = ASLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_with_thresholds,
    )

    # Train
    print("🚀 Starting ASL training...")
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
        "loss_function": "Asymmetric Loss (ASL)",
        "f1_micro": eval_results.get("eval_f1_micro", 0.0),
        "f1_macro": eval_results.get("eval_f1_macro", 0.0),
        "f1_micro_t3": eval_results.get("eval_f1_micro_t3", 0.0),
        "f1_macro_t3": eval_results.get("eval_f1_macro_t3", 0.0),
        "f1_micro_t5": eval_results.get("eval_f1_micro_t5", 0.0),
        "f1_macro_t5": eval_results.get("eval_f1_macro_t5", 0.0),
        "f1_micro_t7": eval_results.get("eval_f1_micro_t7", 0.0),
        "f1_macro_t7": eval_results.get("eval_f1_macro_t7", 0.0),
        "avg_preds_t3": eval_results.get("eval_avg_preds_t3", 0.0),
        "avg_preds_t5": eval_results.get("eval_avg_preds_t5", 0.0),
        "avg_preds_t7": eval_results.get("eval_avg_preds_t7", 0.0),
        "eval_loss": eval_results.get("eval_loss", 0.0),
    }

    with open(os.path.join(args.output_dir, "eval_report.json"), "w") as f:
        json.dump(eval_report, f, indent=2)

    print("✅ ASL Training completed!")
    print(f"📈 Final F1 Macro: {eval_results.get('eval_f1_macro', 0.0):.4f}")
    print(f"📈 Final F1 Micro: {eval_results.get('eval_f1_micro', 0.0):.4f}")
    print(f"📊 Average predictions (t=0.3): {eval_results.get('eval_avg_preds_t3', 0.0):.2f}")
    print(f"💾 Model saved to: {args.output_dir}")

if __name__ == "__main__":
    main()