#!/usr/bin/env python3
"""
Debug script to check ASL training behavior
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# GoEmotions labels
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=1.0, gamma_pos=1.0, clip=0.2, eps=1e-8, disable_torch_grad_focal_loss=False):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        if self.clip is not None and self.clip > 0:
            xs_pos = (xs_pos + self.clip).clamp(max=1)
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

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

        result = -loss.mean()
        print(f"DEBUG ASL: loss before mean = {loss.mean().item():.6f}, final loss = {result.item():.6f}")
        return result

class SimpleDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=256, max_samples=100):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                item = eval(line.strip())
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
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_vector, dtype=torch.float)
        }

def debug_asl_training():
    print("🔍 DEBUG: ASL Training Investigation")
    print("=" * 50)

    # Load model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("models/deberta-v3-large", use_fast=False, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            "models/deberta-v3-large",
            num_labels=len(EMOTION_LABELS),
            problem_type="multi_label_classification"
        )
        print("✅ Model and tokenizer loaded")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Create dataset
    dataset = SimpleDataset("data/goemotions/val.jsonl", tokenizer, max_samples=50)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    print(f"✅ Created dataset with {len(dataset)} samples")

    # Test ASL
    asl_loss = AsymmetricLoss(gamma_neg=1.0, gamma_pos=1.0, clip=0.2, disable_torch_grad_focal_loss=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    print("\n🚀 Testing ASL training for 3 steps...")

    model.train()
    for step, batch in enumerate(dataloader):
        if step >= 3:
            break

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        print(f"\nStep {step + 1}:")
        print(f"  Batch shape: {input_ids.shape}")
        print(f"  Labels positive count: {labels.sum().item()}")

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        print(f"  Logits shape: {logits.shape}")
        print(f"  Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")

        loss = asl_loss(logits, labels)
        print(f"  Loss: {loss.item():.6f}")
        print(f"  Loss is positive: {loss.item() > 0}")

        loss.backward()

        # Check gradients
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        grad_norm = total_norm ** (1. / 2)
        print(f"  Gradient norm: {grad_norm:.4f}")

        optimizer.step()

        # Check predictions
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            avg_preds = preds.sum(dim=1).mean().item()
            print(f"  Average predictions (threshold=0.5): {avg_preds:.2f}")

    print("\n✅ DEBUG completed!")

if __name__ == "__main__":
    debug_asl_training()