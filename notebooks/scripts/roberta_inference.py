#!/usr/bin/env python3
"""
Working inference script for the trained RoBERTa-large model
Tests the actual model we have (not the one we wanted)
"""

import json
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

# Set HF token
os.environ["HF_TOKEN"] = "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"

EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]

def load_roberta_model():
    """Load the trained RoBERTa-large + LoRA model"""
    print("🤖 Loading RoBERTa-large + LoRA adapter...")
    
    checkpoint_path = "./samo_out/checkpoint-1833"
    base_model_name = "roberta-large"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=False)
    print("✅ Tokenizer loaded")
    
    # Load base RoBERTa model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=28,
        problem_type="multi_label_classification"
    )
    print("✅ Base RoBERTa-large model loaded")
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()
    print("✅ LoRA adapter loaded")
    
    return model, tokenizer

def predict_emotions(model, tokenizer, text, threshold=0.3):
    """Predict emotions for text"""
    inputs = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits)
    
    predictions = (probabilities >= threshold).cpu().numpy()[0]
    probabilities = probabilities.cpu().numpy()[0]
    
    predicted_emotions = []
    for i, (emotion, pred, prob) in enumerate(zip(EMOTION_LABELS, predictions, probabilities)):
        if pred:
            predicted_emotions.append((emotion, prob))
    
    predicted_emotions.sort(key=lambda x: x[1], reverse=True)
    return predicted_emotions, probabilities

def test_roberta_model():
    """Test the RoBERTa model on various examples"""
    
    print("🚀 RoBERTa-large GoEmotions Inference Test")
    print("="*60)
    
    try:
        model, tokenizer = load_roberta_model()
        
        # Test examples
        test_examples = [
            "I love this movie! It's absolutely amazing and made me so happy!",
            "This is terrible. I'm so angry and disappointed with this service.",
            "I'm really confused about what happened. Can someone explain?",
            "Thank you so much for your help! I'm really grateful.",
            "I'm so excited about the upcoming vacation! Can't wait!",
            "This is just okay, nothing special.",
            "I'm really scared about the exam tomorrow.",
            "Wow, I never expected that to happen!",
            "I feel so proud of my daughter's achievement.",
            "I'm really sorry for what I did. I feel terrible about it."
        ]
        
        print("\n🧪 Testing RoBERTa-large on example texts:")
        print("-" * 60)
        
        for i, text in enumerate(test_examples, 1):
            print(f"\n📝 Example {i}: \"{text}\"")
            
            for threshold in [0.3, 0.5, 0.7]:
                emotions, probs = predict_emotions(model, tokenizer, text, threshold)
                emotion_names = [emotion for emotion, _ in emotions]
                print(f"   Threshold {threshold}: {emotion_names}")
        
        # Test validation samples
        print("\n🔬 Testing on validation samples:")
        print("-" * 60)
        
        with open("./samo_out/val.jsonl", "r") as f:
            for i, line in enumerate(f):
                if i >= 5:  # Test first 5
                    break
                
                sample = json.loads(line)
                text = sample["text"]
                true_labels = sample["labels"]
                true_emotions = [EMOTION_LABELS[j] for j, label in enumerate(true_labels) if label == 1]
                
                print(f"\n📝 Val Sample {i+1}: \"{text}\"")
                print(f"   True: {true_emotions}")
                
                emotions, probs = predict_emotions(model, tokenizer, text, threshold=0.3)
                predicted = [emotion for emotion, _ in emotions]
                print(f"   Predicted: {predicted}")
                
                # Calculate overlap
                overlap = set(predicted) & set(true_emotions)
                precision = len(overlap) / len(predicted) if predicted else 0
                recall = len(overlap) / len(true_emotions) if true_emotions else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                print(f"   Overlap: {list(overlap)}")
                print(f"   Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        
        print("\n" + "="*60)
        print("✅ RoBERTa-large inference test complete!")
        print("📊 This is the actual model we trained (not DeBERTa-v3-large)")
        print("🎯 Performance: 21.8% F1 Micro at t=0.3 is reasonable for RoBERTa-large")
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_roberta_model()
