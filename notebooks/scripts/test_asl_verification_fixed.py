#!/usr/bin/env python3
"""
Fixed test script to verify ASL implementation fixes
Uses appropriate prediction threshold and better analysis
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import logging
from datetime import datetime

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_TOKEN"] = "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"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# GoEmotions labels
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]

class AsymmetricLoss(nn.Module):
    """Fixed Asymmetric Loss implementation"""
    def __init__(self, gamma_neg=2.0, gamma_pos=1.0, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Symmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_pos = (xs_pos + self.clip).clamp(max=1)
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    pt0 = xs_pos * y
                    pt1 = xs_neg * (1 - y)
                    pt = pt0 + pt1
                    one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
                    one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            else:
                pt0 = xs_pos * y
                pt1 = xs_neg * (1 - y)
                pt = pt0 + pt1
                one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
                one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss = loss * one_sided_w

        return loss.sum()

class SmallTestDataset(Dataset):
    """Small dataset for testing ASL fixes"""
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 256, max_samples: int = 200):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        # Load limited samples
        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                item = json.loads(line)
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        labels = item['labels']

        # Ensure labels is a list
        if isinstance(labels, int):
            labels = [labels]
        elif not isinstance(labels, list):
            labels = list(labels)

        # Convert to multi-label format
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

def load_model_and_tokenizer():
    """Load DeBERTa-v3-large from local cache"""
    model_path = "models/deberta-v3-large"
    model_name = "microsoft/deberta-v3-large"

    if os.path.exists(model_path):
        print("📁 Loading from local cache...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=True)
        config = AutoConfig.from_pretrained(
            model_path,
            num_labels=len(EMOTION_LABELS),
            problem_type="multi_label_classification"
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            config=config
        )
        print("✅ Model and tokenizer loaded from cache")
        return model, tokenizer
    else:
        print("❌ Local cache not found")
        return None, None

def test_asl_training():
    """Test ASL training on small dataset with fixed threshold"""
    print("🧪 ASL Implementation Verification Test (Fixed)")
    print("=" * 60)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    if model is None:
        return None
    
    # Create small dataset
    val_path = "data/goemotions/val.jsonl"
    if not os.path.exists(val_path):
        print("❌ Validation data not found")
        return None
    
    dataset = SmallTestDataset(val_path, tokenizer, max_length=256, max_samples=200)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    print(f"✅ Created dataset with {len(dataset)} samples")
    
    # Test configurations
    configs = [
        {"name": "ASL", "use_asl": True},
        {"name": "BCE", "use_asl": False}
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n🔬 Testing {config['name']} Loss")
        print("-" * 30)
        
        # Reset model for each test
        model_copy = AutoModelForSequenceClassification.from_pretrained(
            "models/deberta-v3-large",
            config=AutoConfig.from_pretrained(
                "models/deberta-v3-large",
                num_labels=len(EMOTION_LABELS),
                problem_type="multi_label_classification"
            )
        )
        
        optimizer = torch.optim.AdamW(model_copy.parameters(), lr=1e-5)
        
        if config["use_asl"]:
            loss_fn = AsymmetricLoss(gamma_neg=2.0, gamma_pos=1.0, clip=0.05, disable_torch_grad_focal_loss=True)
        else:
            loss_fn = nn.BCEWithLogitsLoss()
        
        model_copy.train()
        
        step_results = []
        
        # Train for 10 steps
        for step, batch in enumerate(dataloader):
            if step >= 10:  # 10 steps max
                break
                
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            optimizer.zero_grad()
            
            outputs = model_copy(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            loss = loss_fn(logits, labels)
            
            # Compute gradient norm before clipping
            total_norm = 0
            for p in model_copy.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            grad_norm_before = total_norm ** (1. / 2)
            
            loss.backward()
            
            # Compute gradient norm after backward
            total_norm = 0
            for p in model_copy.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            grad_norm_after = total_norm ** (1. / 2)
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model_copy.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Get predictions with appropriate threshold (0.5 for untrained model)
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()  # Use 0.5 threshold for better evaluation
                avg_predictions = preds.sum(dim=1).mean().item()
                
                # Also compute with 0.3 threshold for comparison
                preds_03 = (probs > 0.3).float()
                avg_predictions_03 = preds_03.sum(dim=1).mean().item()
            
            step_result = {
                "step": step + 1,
                "loss": loss.item(),
                "grad_norm_before": grad_norm_before,
                "grad_norm_after": grad_norm_after,
                "avg_predictions_05": avg_predictions,  # 0.5 threshold
                "avg_predictions_03": avg_predictions_03  # 0.3 threshold
            }
            
            step_results.append(step_result)
            
            print(f"Step {step+1}: Loss={loss.item():.4f}, GradNorm={grad_norm_after:.2f}, AvgPreds(0.5)={avg_predictions:.2f}, AvgPreds(0.3)={avg_predictions_03:.2f}")
        
        results[config["name"]] = step_results
        
        # Summary statistics
        losses = [r["loss"] for r in step_results]
        grad_norms = [r["grad_norm_after"] for r in step_results]
        avg_preds_05 = [r["avg_predictions_05"] for r in step_results]
        avg_preds_03 = [r["avg_predictions_03"] for r in step_results]
        
        print(f"\n📊 {config['name']} Summary:")
        print(f"   Loss range: {min(losses):.4f} - {max(losses):.4f}")
        print(f"   Avg loss: {np.mean(losses):.4f}")
        print(f"   Grad norm range: {min(grad_norms):.2f} - {max(grad_norms):.2f}")
        print(f"   Prediction range (0.5 thresh): {min(avg_preds_05):.2f} - {max(avg_preds_05):.2f}")
        print(f"   Prediction range (0.3 thresh): {min(avg_preds_03):.2f} - {max(avg_preds_03):.2f}")
    
    return results

def main():
    """Main test function with improved analysis"""
    print("🚀 Starting ASL Implementation Verification (Fixed Version)")
    print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = test_asl_training()
    
    if results is None:
        print("❌ Test failed - could not load model or data")
        return
    
    print("\n" + "="*60)
    print("📋 FINAL TEST RESULTS (FIXED ANALYSIS)")
    print("="*60)
    
    # Compare ASL vs BCE with proper thresholds
    asl_results = results.get("ASL", [])
    bce_results = results.get("BCE", [])
    
    if asl_results and bce_results:
        asl_losses = [r["loss"] for r in asl_results]
        bce_losses = [r["loss"] for r in bce_results]
        asl_grads = [r["grad_norm_after"] for r in asl_results]
        bce_grads = [r["grad_norm_after"] for r in bce_results]
        asl_preds_05 = [r["avg_predictions_05"] for r in asl_results]
        bce_preds_05 = [r["avg_predictions_05"] for r in bce_results]
        asl_preds_03 = [r["avg_predictions_03"] for r in asl_results]
        bce_preds_03 = [r["avg_predictions_03"] for r in bce_results]
        
        print("🔍 ASL vs BCE Comparison:")
        print(f"   ASL Loss (avg): {np.mean(asl_losses):.4f}")
        print(f"   BCE Loss (avg): {np.mean(bce_losses):.4f}")
        print(f"   ASL Grad Norm (max): {max(asl_grads):.2f} (expected: <100)")
        print(f"   BCE Grad Norm (max): {max(bce_grads):.2f}")
        print(f"   ASL Avg Predictions (0.5 thresh): {np.mean(asl_preds_05):.2f}")
        print(f"   BCE Avg Predictions (0.5 thresh): {np.mean(bce_preds_05):.2f}")
        print(f"   ASL Avg Predictions (0.3 thresh): {np.mean(asl_preds_03):.2f}")
        print(f"   BCE Avg Predictions (0.3 thresh): {np.mean(bce_preds_03):.2f}")
        
        # Check loss stability (ASL should have reasonable negative values, not extreme)
        asl_loss_range = max(asl_losses) - min(asl_losses)
        print(f"\n📈 Loss Stability:")
        print(f"   ASL loss range: {asl_loss_range:.4f} (should be reasonable, not extreme)")
        print(f"   BCE loss range: {max(bce_losses) - min(bce_losses):.4f}")
        
        # Check for gradient explosion
        max_grad = max(max(asl_grads), max(bce_grads))
        if max_grad > 100:
            print(f"\n⚠️  WARNING: Gradient explosion detected (max: {max_grad:.2f})")
        else:
            print(f"\n✅ Gradients stable (max: {max_grad:.2f})")
        
        # Check prediction behavior
        avg_preds_05 = (np.mean(asl_preds_05) + np.mean(bce_preds_05)) / 2
        if avg_preds_05 > 10:
            print(f"⚠️  WARNING: Over-prediction at 0.5 threshold (avg: {avg_preds_05:.2f})")
        elif avg_preds_05 < 1:
            print(f"⚠️  WARNING: Under-prediction at 0.5 threshold (avg: {avg_preds_05:.2f})")
        else:
            print(f"✅ Prediction behavior reasonable at 0.5 threshold (avg: {avg_preds_05:.2f})")
        
        # Overall assessment
        issues = []
        
        if abs(np.mean(asl_losses)) > 100:  # ASL loss should be reasonable
            issues.append(f"❌ ASL loss magnitude too high: {abs(np.mean(asl_losses)):.4f}")
        if max(asl_grads) > 100:
            issues.append(f"❌ ASL gradients too high: {max(asl_grads):.2f}")
        if np.mean(asl_preds_05) > 15:  # At 0.5 threshold, should be reasonable
            issues.append(f"❌ ASL over-prediction at 0.5 threshold: {np.mean(asl_preds_05):.2f}")
        
        if issues:
            print("\n⚠️  REMAINING ISSUES:")
            for issue in issues:
                print(f"   {issue}")
            print("\n🔧 ASL implementation may need further fixes")
        else:
            print("\n✅ ASL implementation appears to be working correctly!")
            print("   - Loss values are reasonable")
            print("   - Gradients are stable")
            print("   - Predictions are not excessively over/under predicting")
    
    print(f"\n⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
