#!/usr/bin/env python3
"""
Fast evaluation script for GoEmotions DeBERTa-v3-large model
Generates evaluation report without slow threshold optimization
"""

import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import f1_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings("ignore")

# Set HF token for authentication
import os
os.environ["HF_TOKEN"] = "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"

def load_model_and_tokenizer(checkpoint_path, model_name="microsoft/deberta-v3-large"):
    """Load the trained model and tokenizer from checkpoint"""
    print(f"🤖 Loading model from {checkpoint_path}")
    
    # Check if this is a LoRA checkpoint
    is_lora = os.path.exists(os.path.join(checkpoint_path, "adapter_config.json"))
    
    if is_lora:
        print("🔧 Detected LoRA checkpoint - loading base model + adapter")
        from peft import PeftModel
        
        # Load tokenizer from checkpoint
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=False)
        print("✅ Loaded tokenizer from checkpoint")
        
        # Load base model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=28,
            problem_type="multi_label_classification"
        )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        print("✅ Loaded LoRA adapter")
        
    else:
        # Load tokenizer from checkpoint (should be cached there)
        try:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=False)
            print("✅ Loaded tokenizer from checkpoint")
        except Exception as e:
            print(f"⚠️  Could not load tokenizer from checkpoint: {e}")
            print("🔄 Trying to load from original model name...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        
        # Load full model
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_path,
            num_labels=28,  # GoEmotions has 28 classes
            problem_type="multi_label_classification"
        )
    
    return model, tokenizer

def load_validation_data():
    """Load validation data"""
    print("📊 Loading validation data...")
    
    val_data = []
    with open("./samo_out/val.jsonl", "r") as f:
        for line in f:
            val_data.append(json.loads(line))
    
    print(f"✅ Loaded {len(val_data)} validation examples")
    return val_data

def tokenize_data(data, tokenizer, max_length=512):
    """Tokenize the data"""
    print("🔄 Tokenizing data...")
    
    texts = [item["text"] for item in data]
    labels = [item["labels"] for item in data]
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Convert to dataset
    dataset = Dataset.from_dict({
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels
    })
    
    return dataset

def compute_fast_metrics(eval_pred):
    """Fast metrics computation with multiple thresholds"""
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
    
    # Primary metrics (using 0.5 threshold)
    metrics["f1_micro"] = metrics["f1_micro_t5"]
    metrics["f1_macro"] = metrics["f1_macro_t5"]
    
    return metrics

def fast_threshold_optimization(logits, labels, num_samples=1000):
    """Fast threshold optimization using sampling"""
    print("🔍 Running fast threshold optimization...")
    
    # Sample subset for faster computation
    if len(logits) > num_samples:
        indices = np.random.choice(len(logits), num_samples, replace=False)
        logits_sample = logits[indices]
        labels_sample = labels[indices]
    else:
        logits_sample = logits
        labels_sample = labels
    
    probs = 1.0 / (1.0 + np.exp(-logits_sample))
    num_classes = labels_sample.shape[1]
    optimal_thresholds = np.full(num_classes, 0.5)  # Default threshold
    
    for j in range(num_classes):
        y_true = labels_sample[:, j]
        y_scores = probs[:, j]
        
        # Skip if no positive examples
        if y_true.sum() == 0:
            continue
        
        best_f1 = 0
        best_threshold = 0.5
        
        # Test fewer thresholds for speed
        for threshold in np.linspace(0.2, 0.8, 7):  # Only 7 thresholds instead of 17
            y_pred = (y_scores >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        optimal_thresholds[j] = best_threshold
    
    return optimal_thresholds

def main():
    """Main evaluation function"""
    print("🚀 Fast GoEmotions Evaluation")
    
    # Use the latest checkpoint
    checkpoint_path = "./samo_out/checkpoint-1833"
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(checkpoint_path)
    
    # Load validation data
    val_data = load_validation_data()
    val_dataset = tokenize_data(val_data, tokenizer)
    
    # Create trainer
    training_args = TrainingArguments(
        output_dir="./temp_eval",
        per_device_eval_batch_size=16,
        dataloader_drop_last=False,
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=val_dataset,
        compute_metrics=compute_fast_metrics,
    )
    
    # Evaluate
    print("📊 Evaluating...")
    eval_results = trainer.evaluate()
    
    # Get predictions for threshold optimization
    print("🔍 Getting predictions...")
    predictions = trainer.predict(val_dataset)
    logits = predictions.predictions
    labels = predictions.label_ids
    
    # Fast threshold optimization
    optimal_thresholds = fast_threshold_optimization(logits, labels)
    
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
        "eval_loss": eval_results.get("eval_loss", 0.0),
        "eval_runtime": eval_results.get("eval_runtime", 0.0),
        "eval_samples_per_second": eval_results.get("eval_samples_per_second", 0.0),
    }
    
    # Save evaluation report
    report_path = "./samo_out/eval_report.json"
    with open(report_path, "w") as f:
        json.dump(eval_report, f, indent=2)
    
    # Save optimal thresholds
    thresholds_path = "./samo_out/optimal_thresholds.json"
    with open(thresholds_path, "w") as f:
        json.dump(optimal_thresholds.tolist(), f)
    
    # Print results
    print("\n" + "="*50)
    print("📈 EVALUATION RESULTS")
    print("="*50)
    print(f"📊 F1 Micro: {eval_results.get('eval_f1_micro', 0.0):.4f}")
    print(f"📊 F1 Macro: {eval_results.get('eval_f1_macro', 0.0):.4f}")
    print(f"📊 F1 Micro (t=0.3): {eval_results.get('eval_f1_micro_t3', 0.0):.4f}")
    print(f"📊 F1 Macro (t=0.3): {eval_results.get('eval_f1_macro_t3', 0.0):.4f}")
    print(f"📊 F1 Micro (t=0.5): {eval_results.get('eval_f1_micro_t5', 0.0):.4f}")
    print(f"📊 F1 Macro (t=0.5): {eval_results.get('eval_f1_macro_t5', 0.0):.4f}")
    print(f"📊 F1 Micro (t=0.7): {eval_results.get('eval_f1_micro_t7', 0.0):.4f}")
    print(f"📊 F1 Macro (t=0.7): {eval_results.get('eval_f1_macro_t7', 0.0):.4f}")
    print(f"💾 Evaluation report saved to: {report_path}")
    print(f"💾 Optimal thresholds saved to: {thresholds_path}")
    
    # Performance assessment
    f1_macro = eval_results.get('eval_f1_macro', 0.0)
    if f1_macro >= 0.60:
        print("🎉 EXCELLENT performance! F1 Macro >= 0.60")
    elif f1_macro >= 0.50:
        print("✅ GOOD performance! F1 Macro >= 0.50")
    elif f1_macro >= 0.30:
        print("⚠️  MODERATE performance. Consider threshold tuning or more training.")
    else:
        print("❌ LOW performance. May need more training or different approach.")
    
    print("="*50)

if __name__ == "__main__":
    main()
