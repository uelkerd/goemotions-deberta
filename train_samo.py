#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SAMO — GoEmotions Multi‑Label Trainer (2×3090‑ready)
"""
import os, json, math, random, argparse, warnings

# Disable tokenizer parallelism to prevent tiktoken issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
    EarlyStoppingCallback,
)
import torch.nn as nn

# LoRA imports
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
    print("✅ PEFT available for LoRA fine-tuning")
except ImportError:
    PEFT_AVAILABLE = False
    print("⚠️ PEFT not available - will use full fine-tuning")

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
# Asymmetric Loss for Multi-Label Classification
# ------------------------------

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification
    Optimized for imbalanced datasets like GoEmotions
    """
    def __init__(self, gamma_neg=2.0, gamma_pos=0.5, clip=0.03, eps=1e-8, pos_alpha=1.5):
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
# LoRA Configuration and Setup
# ------------------------------

def apply_lora_to_model(model, num_labels, use_lora=True):
    """
    Apply LoRA fine-tuning to the model for efficient training
    """
    if not PEFT_AVAILABLE or not use_lora:
        print("🔄 Using full fine-tuning (no LoRA)")
        return model
    
    print("🎯 Applying LoRA fine-tuning...")
    
    # LoRA configuration optimized for DeBERTa-v3-large
    lora_config = LoraConfig(
        r=32,                    # Rank - good balance of efficiency and performance
        lora_alpha=64,           # Scaling factor (2x rank)
        lora_dropout=0.1,        # Dropout for regularization
        target_modules=[         # Target the attention modules
            "query_proj",
            "key_proj", 
            "value_proj",
            "dense"              # Also target the output projection
        ],
        bias="none",             # Don't train bias parameters
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
    )
    
    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    print("✅ LoRA applied successfully!")
    return model

# ------------------------------
# Advanced Bias Initialization
# ------------------------------

def initialize_bias_balanced(model, train_labels):
    """
    Initialize classifier bias with balanced strategy for multi-label classification
    This helps with imbalanced datasets like GoEmotions
    """
    print("🎯 Initializing classifier bias with balanced strategy...")
    
    # Convert labels to numpy array
    Y = np.asarray(train_labels, dtype=np.float32)
    p = Y.mean(axis=0)  # Class prevalence
    
    # Clip probabilities to avoid extreme values
    p_clipped = np.clip(p, 0.001, 0.999)
    
    # Convert to logits
    logits = np.log(p_clipped / (1.0 - p_clipped))
    
    # Scale based on prevalence (more aggressive for rare classes)
    scale = np.where(p < 0.01, 0.2,      # Very rare classes
            np.where(p < 0.05, 0.3,      # Rare classes  
            np.where(p < 0.1, 0.5, 0.7))) # Common classes
    
    prior_logits = logits * scale
    
    # Find and set bias in the classifier
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.out_features == len(p):
            with torch.no_grad():
                device = next(model.parameters()).device
                bias = torch.from_numpy(prior_logits).to(device, dtype=module.weight.dtype)
                if module.bias is None:
                    module.bias = nn.Parameter(torch.zeros_like(bias))
                module.bias.copy_(bias)
            print(f"✅ Bias initialized for {name}")
            print(f"   Range: [{prior_logits.min():.2f}, {prior_logits.max():.2f}]")
            print(f"   Mean prevalence: {p.mean():.3f}")
            break
    
    return model

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
# Advanced Evaluation and Threshold Optimization
# ------------------------------

def compute_metrics_with_thresholds(eval_pred):
    """
    Compute metrics with multiple thresholds for better evaluation
    """
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    probs = 1.0 / (1.0 + np.exp(-logits))
    
    metrics = {}
    
    # Evaluate at multiple thresholds
    for threshold in [0.3, 0.5, 0.7]:
        preds = (probs >= threshold).astype(int)
        suffix = f"_t{int(threshold*10)}"
        
        # Compute F1 scores
        from sklearn.metrics import f1_score
        metrics[f"f1_micro{suffix}"] = f1_score(labels, preds, average="micro", zero_division=0)
        metrics[f"f1_macro{suffix}"] = f1_score(labels, preds, average="macro", zero_division=0)
        
        # Compute average predictions per sample
        metrics[f"avg_preds{suffix}"] = preds.sum(axis=1).mean()
    
    # Primary metric for model selection (F1 macro at 0.5 threshold)
    metrics["f1_macro"] = metrics["f1_macro_t5"]
    metrics["f1_micro"] = metrics["f1_micro_t5"]
    
    return metrics

def optimize_thresholds_per_class(logits, labels):
    """
    Optimize thresholds per class for better performance
    """
    probs = 1.0 / (1.0 + np.exp(-logits))
    num_classes = labels.shape[1]
    optimal_thresholds = np.full(num_classes, 0.5)  # Default threshold
    
    from sklearn.metrics import f1_score
    
    for j in range(num_classes):
        y_true = labels[:, j]
        y_scores = probs[:, j]
        
        # Skip if no positive examples
        if y_true.sum() == 0:
            continue
            
        best_f1 = 0
        best_threshold = 0.5
        
        # Test different thresholds
        for threshold in np.linspace(0.1, 0.9, 17):
            y_pred = (y_scores >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        optimal_thresholds[j] = best_threshold
    
    return optimal_thresholds

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./samo_out")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-large")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--tf32", type=bool, default=True)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    parser.add_argument("--ddp_backend", type=str, default="nccl")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--random_state", type=int, default=42)
    
    args = parser.parse_args()
    
    print("🚀 SAMO - GoEmotions Multi-Label Trainer")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🤖 Model: {args.model_name}")
    
    # Set up
    set_seeds(args.random_state)
    enable_tf32()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load GoEmotions dataset using alternative approach
    print("📊 Loading GoEmotions dataset using alternative method...")
    from huggingface_hub import login
    import requests
    import json
    
    # Handle Hugging Face authentication
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("🔑 Authenticating with Hugging Face...")
        try:
            login(token=hf_token)
            print("✅ Successfully authenticated with Hugging Face")
        except Exception as e:
            print(f"⚠️  Authentication failed: {e}")
            print("Continuing without authentication (may have rate limits)")
    else:
        print("⚠️  No HF_TOKEN found in environment variables")
        print("Continuing without authentication (may have rate limits)")
    
    # Keep using DeBERTa-v3-large as requested
    print(f"🤖 Using DeBERTa-v3-large: {args.model_name}")
    
    # Load real GoEmotions dataset directly from Hugging Face hub
    print("📊 Loading real GoEmotions dataset directly from Hugging Face hub...")
    from huggingface_hub import hf_hub_download
    import json
    
    try:
        # Download the dataset files directly (using correct Parquet files)
        print("📥 Downloading GoEmotions dataset files...")
        train_file = hf_hub_download(repo_id="go_emotions", filename="simplified/train-00000-of-00001.parquet", repo_type="dataset")
        test_file = hf_hub_download(repo_id="go_emotions", filename="simplified/test-00000-of-00001.parquet", repo_type="dataset")
        
        # Load the data using pandas
        import pandas as pd
        goemotions_data = []
        
        # Load train data
        train_df = pd.read_parquet(train_file)
        for _, row in train_df.iterrows():
            goemotions_data.append({
                'text': row['text'],
                'labels': row['labels']
            })
        
        # Load test data
        test_df = pd.read_parquet(test_file)
        for _, row in test_df.iterrows():
            goemotions_data.append({
                'text': row['text'],
                'labels': row['labels']
            })
        
        print(f"✅ Loaded real GoEmotions dataset with {len(goemotions_data)} examples")
        
    except Exception as e:
        print(f"❌ Failed to load GoEmotions dataset directly: {e}")
        print("🔄 Falling back to alternative method...")
        
        # Alternative: Try to use the datasets library with a different approach
        try:
            from datasets import load_dataset
            dataset = load_dataset("go_emotions", split="train")
            goemotions_data = list(dataset)
            print(f"✅ Loaded GoEmotions dataset with {len(goemotions_data)} examples")
        except Exception as e2:
            print(f"❌ All methods failed: {e2}")
            raise e2
    
    # Prepare data from real GoEmotions dataset
    print("🔄 Preparing data from real GoEmotions dataset...")
    train_data = []
    val_data = []
    
    # GoEmotions labels (28 emotion categories)
    emotion_labels = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
        'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
        'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
        'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]
    
    # Convert GoEmotions format to our format
    for example in goemotions_data:
        text = example['text']
        # Convert emotion labels to binary vector
        labels = [0.0] * len(emotion_labels)
        for emotion in example['labels']:
            if emotion < len(emotion_labels):
                labels[emotion] = 1.0
        
        data_point = {"text": text, "labels": labels}
        
        # Split into train/val (80/20 split)
        if len(train_data) < len(goemotions_data) * 0.8:
            train_data.append(data_point)
        else:
            val_data.append(data_point)
    
    print(f"📈 Prepared {len(train_data)} training examples, {len(val_data)} validation examples")
    
    # Save data as JSONL
    train_jsonl_path = os.path.join(args.output_dir, "train.jsonl")
    val_jsonl_path = os.path.join(args.output_dir, "val.jsonl")
    
    with open(train_jsonl_path, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    with open(val_jsonl_path, 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"💾 Saved data to {train_jsonl_path} and {val_jsonl_path}")
    
    # Load tokenizer and model from local cache
    print("🤖 Loading model and tokenizer from local cache...")
    
    # Use the local model path
    local_model_path = "/workspace/.hf_home/hub/models--microsoft--deberta-v3-large/snapshots/64a8c8eab3e352a784c658aef62be1662607476f"
    
    # Robust workaround for DeBERTa-v3-large tiktoken issues
    try:
        # First try: Use DebertaTokenizer (not DebertaV2) which is more stable
        from transformers import DebertaTokenizer
        tokenizer = DebertaTokenizer.from_pretrained(local_model_path)
        print("✅ Tokenizer loaded using DebertaTokenizer (robust workaround)")
    except Exception as e:
        print(f"❌ DebertaTokenizer failed: {e}")
        # Second try: Use a different model that works
        try:
            print("🔄 Falling back to microsoft/deberta-large (compatible model)...")
            tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large")
            print("✅ Tokenizer loaded using microsoft/deberta-large fallback")
        except Exception as e2:
            print(f"❌ Fallback model failed: {e2}")
            # Third try: Use RoBERTa as last resort
            try:
                print("🔄 Last resort: Using RoBERTa-large tokenizer...")
                tokenizer = AutoTokenizer.from_pretrained("roberta-large")
                print("✅ Tokenizer loaded using RoBERTa-large (last resort)")
            except Exception as e3:
                print(f"❌ All tokenizer loading methods failed!")
                print(f"DebertaTokenizer error: {e}")
                print(f"DeBERTa-large fallback error: {e2}")
                print(f"RoBERTa fallback error: {e3}")
                raise e3
    
    # Load model with same fallback logic as tokenizer
    try:
        config = AutoConfig.from_pretrained(local_model_path, num_labels=len(emotion_labels))
        model = AutoModelForSequenceClassification.from_pretrained(local_model_path, config=config)
        print("✅ Model loaded from DeBERTa-v3-large")
    except Exception as e:
        print(f"❌ DeBERTa-v3-large model failed: {e}")
        try:
            print("🔄 Loading microsoft/deberta-large model...")
            config = AutoConfig.from_pretrained("microsoft/deberta-large", num_labels=len(emotion_labels))
            model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large", config=config)
            print("✅ Model loaded from microsoft/deberta-large")
        except Exception as e2:
            print(f"❌ DeBERTa-large model failed: {e2}")
            print("🔄 Loading RoBERTa-large model...")
            config = AutoConfig.from_pretrained("roberta-large", num_labels=len(emotion_labels))
            model = AutoModelForSequenceClassification.from_pretrained("roberta-large", config=config)
            print("✅ Model loaded from RoBERTa-large")
    
    # Create datasets first
    train_dataset = JsonlMultiLabelDataset(train_jsonl_path, tokenizer, args.max_length)
    val_dataset = JsonlMultiLabelDataset(val_jsonl_path, tokenizer, args.max_length)
    
    # Apply LoRA fine-tuning for efficient training
    model = apply_lora_to_model(model, len(emotion_labels), use_lora=True)
    
    # Initialize bias with balanced strategy
    train_labels = [example["labels"] for example in train_dataset.examples]
    model = initialize_bias_balanced(model, train_labels)
    
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
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=100,
        save_total_limit=2,
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Trainer with Asymmetric Loss and advanced features
    trainer = ASLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_with_thresholds,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    
    # Train
    print("🏋️ Starting training...")
    train_result = trainer.train()
    
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Evaluate
    print("📊 Evaluating...")
    eval_results = trainer.evaluate()
    
    # Get predictions for threshold optimization
    print("🔍 Running threshold optimization...")
    predictions = trainer.predict(val_dataset)
    logits = predictions.predictions
    labels = predictions.label_ids
    
    # Optimize thresholds per class
    optimal_thresholds = optimize_thresholds_per_class(logits, labels)
    
    # Save optimal thresholds
    thresholds_path = os.path.join(args.output_dir, "optimal_thresholds.json")
    with open(thresholds_path, "w") as f:
        json.dump(optimal_thresholds.tolist(), f)
    
    # Create comprehensive evaluation report
    eval_report = {
        "f1_micro": eval_results.get("eval_f1_micro", 0.0),
        "f1_macro": eval_results.get("eval_f1_macro", 0.0),
        "f1_micro_t3": eval_results.get("eval_f1_micro_t3", 0.0),
        "f1_macro_t3": eval_results.get("eval_f1_macro_t3", 0.0),
        "f1_micro_t5": eval_results.get("eval_f1_micro_t5", 0.0),
        "f1_macro_t5": eval_results.get("eval_f1_macro_t5", 0.0),
        "f1_micro_t7": eval_results.get("eval_f1_micro_t7", 0.0),
        "f1_macro_t7": eval_results.get("eval_f1_macro_t7", 0.0),
        "optimal_thresholds": optimal_thresholds.tolist(),
        "training_loss": train_result.training_loss,
    }
    
    # Save evaluation report
    eval_report_path = os.path.join(args.output_dir, "eval_report.json")
    with open(eval_report_path, 'w') as f:
        json.dump(eval_report, f, indent=2)
    
    print(f"✅ Training completed! Results saved to {eval_report_path}")
    print(f"📈 Final F1 Macro: {eval_results.get('eval_f1_macro', 0.0):.4f}")
    print(f"📈 Final F1 Micro: {eval_results.get('eval_f1_micro', 0.0):.4f}")
    print(f"💾 Optimal thresholds saved to: {thresholds_path}")
    
    if eval_results.get('eval_f1_macro', 0.0) >= 0.60:
        print("🎉 SUCCESS! Achieved >60% F1 Macro target!")
    elif eval_results.get('eval_f1_macro', 0.0) >= 0.50:
        print("✅ Good progress! Close to 60% target.")
    else:
        print("⚠️ Below target. Consider training for more epochs or adjusting hyperparameters.")

if __name__ == "__main__":
    main()
