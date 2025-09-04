#!/usr/bin/env python3
"""
Comprehensive diagnostic script for GoEmotions training issues
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, precision_recall_fscore_support

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# GoEmotions labels
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]

class DiagnosticDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=256, max_samples=200):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        print(f"🔍 Loading data from {jsonl_path}...")

        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                try:
                    item = json.loads(line.strip())
                    self.data.append(item)
                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error at line {i}: {e}")
                    continue

        print(f"✅ Loaded {len(self.data)} samples")

        # Analyze label distribution
        self.analyze_labels()

    def analyze_labels(self):
        """Analyze label distribution in dataset"""
        label_counts = np.zeros(len(EMOTION_LABELS))

        for item in self.data:
            labels = item.get('labels', [])
            if isinstance(labels, int):
                labels = [labels]
            elif not isinstance(labels, list):
                continue

            for label_idx in labels:
                if 0 <= label_idx < len(EMOTION_LABELS):
                    label_counts[label_idx] += 1

        print("📊 Label Distribution:")
        for i, count in enumerate(label_counts):
            if count > 0:
                print(".1f")

        print(".1f")
        print(".1f")

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
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_vector, dtype=torch.float),
            'text': text
        }

def diagnose_data_loading():
    """Diagnose data loading issues"""
    print("🔍 DIAGNOSING DATA LOADING")
    print("=" * 50)

    try:
        tokenizer = AutoTokenizer.from_pretrained("models/deberta-v3-large", use_fast=False, local_files_only=True)
        print("✅ Tokenizer loaded successfully")
    except Exception as e:
        print(f"❌ Tokenizer loading failed: {e}")
        return

    # Test data loading
    try:
        dataset = DiagnosticDataset("data/goemotions/val.jsonl", tokenizer, max_samples=100)
        print("✅ Dataset created successfully")

        # Test a few samples
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"Sample {i}:")
            print(f"  Text length: {len(sample['text'])}")
            print(f"  Input shape: {sample['input_ids'].shape}")
            print(f"  Labels positive: {sample['labels'].sum().item()}")
            print(f"  Text preview: {sample['text'][:100]}...")

    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")

def diagnose_model_initialization():
    """Diagnose model initialization issues"""
    print("\n🔍 DIAGNOSING MODEL INITIALIZATION")
    print("=" * 50)

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            "models/deberta-v3-large",
            num_labels=len(EMOTION_LABELS),
            problem_type="multi_label_classification"
        )
        print("✅ Model loaded successfully")
        print(f"   Model type: {type(model)}")
        print(f"   Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Check classifier head
        if hasattr(model, 'classifier'):
            print(f"   Classifier: {model.classifier}")
            print(f"   Classifier output features: {model.classifier.out_features}")

        # Check model config
        print(f"   Problem type: {model.config.problem_type}")
        print(f"   Num labels: {model.config.num_labels}")

    except Exception as e:
        print(f"❌ Model loading failed: {e}")

def diagnose_training_setup():
    """Diagnose training setup issues"""
    print("\n🔍 DIAGNOSING TRAINING SETUP")
    print("=" * 50)

    try:
        tokenizer = AutoTokenizer.from_pretrained("models/deberta-v3-large", use_fast=False, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            "models/deberta-v3-large",
            num_labels=len(EMOTION_LABELS),
            problem_type="multi_label_classification"
        )
        dataset = DiagnosticDataset("data/goemotions/val.jsonl", tokenizer, max_samples=50)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

        print("✅ Training components initialized")

        # Test forward pass
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']

                print(f"Batch input shape: {input_ids.shape}")
                print(f"Batch labels shape: {labels.shape}")
                print(f"Labels in batch: {labels.sum(dim=1)}")

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                print(f"Logits shape: {logits.shape}")
                print(f"Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")

                # Test BCE loss
                bce_loss = nn.BCEWithLogitsLoss()
                loss = bce_loss(logits, labels)
                print(f"BCE Loss: {loss.item():.6f}")

                # Test predictions
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                print(f"Predictions shape: {preds.shape}")
                print(f"Average predictions per sample: {preds.sum(dim=1).mean().item():.2f}")

                break  # Just test one batch

    except Exception as e:
        print(f"❌ Training setup failed: {e}")
        import traceback
        traceback.print_exc()

def diagnose_gradient_flow():
    """Diagnose gradient flow issues"""
    print("\n🔍 DIAGNOSING GRADIENT FLOW")
    print("=" * 50)

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            "models/deberta-v3-large",
            num_labels=len(EMOTION_LABELS),
            problem_type="multi_label_classification"
        )

        # Check if gradients are enabled
        grad_enabled = []
        for name, param in model.named_parameters():
            grad_enabled.append(param.requires_grad)

        print(f"Parameters requiring gradients: {sum(grad_enabled)}/{len(grad_enabled)}")

        # Check classifier gradients specifically
        classifier_grads = []
        for name, param in model.named_parameters():
            if 'classifier' in name:
                classifier_grads.append(param.requires_grad)

        print(f"Classifier parameters requiring gradients: {sum(classifier_grads)}/{len(classifier_grads)}")

        if sum(grad_enabled) == 0:
            print("❌ WARNING: No parameters require gradients!")
        else:
            print("✅ Gradients are properly enabled")

    except Exception as e:
        print(f"❌ Gradient flow diagnosis failed: {e}")

def main():
    """Run all diagnostics"""
    print("🚀 COMPREHENSIVE TRAINING DIAGNOSTIC")
    print("=" * 60)

    diagnose_data_loading()
    diagnose_model_initialization()
    diagnose_training_setup()
    diagnose_gradient_flow()

    print("\n" + "=" * 60)
    print("📋 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print("🔍 Check the output above for specific issues")
    print("🔧 Common fixes:")
    print("   - Ensure data files exist and are properly formatted")
    print("   - Check model cache is complete")
    print("   - Verify tokenizer compatibility")
    print("   - Check gradient flow is enabled")
    print("   - Validate loss function is working")

if __name__ == "__main__":
    main()