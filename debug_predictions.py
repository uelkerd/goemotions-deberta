import torch
import torch.nn as nn
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

# Load model
model_path = "models/deberta-v3-large"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=True)
config = AutoConfig.from_pretrained(model_path, num_labels=28, problem_type="multi_label_classification")
model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)

# Load one sample
with open("data/goemotions/val.jsonl", 'r') as f:
    sample = json.loads(f.readline())

print(f"Sample text: {sample['text']}")
print(f"Sample labels: {sample['labels']}")

# Tokenize
inputs = tokenizer(sample['text'], return_tensors='pt', truncation=True, padding='max_length', max_length=256)

# Get predictions
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.sigmoid(logits)
    
print(f"Logits shape: {logits.shape}")
print(f"Logits range: {logits.min().item():.4f} - {logits.max().item():.4f}")
print(f"Probs range: {probs.min().item():.4f} - {probs.max().item():.4f}")

# Check predictions at different thresholds
for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
    preds = (probs > threshold).float()
    num_preds = preds.sum().item()
    print(f"Threshold {threshold}: {num_preds} predictions")

# Show top 5 probabilities
top_probs, top_indices = torch.topk(probs.squeeze(), 5)
print(f"Top 5 probabilities: {top_probs.tolist()}")
print(f"Top 5 indices: {top_indices.tolist()}")
