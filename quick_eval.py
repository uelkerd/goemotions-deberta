
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import f1_score, precision_recall_fscore_support
import numpy as np

# Load model and evaluate
def evaluate_checkpoint(checkpoint_path, name):
    print(f"\n🔍 Evaluating {name}...")
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    # Quick eval on subset
    # ... evaluation code ...
    
    print(f"✅ {name} evaluation complete!")

# Evaluate both
evaluate_checkpoint('outputs/gpu0_asymmetric/checkpoint-1250', 'Asymmetric Loss')
evaluate_checkpoint('outputs/gpu1_combined_50/checkpoint-1250', 'Combined Loss 50%')
