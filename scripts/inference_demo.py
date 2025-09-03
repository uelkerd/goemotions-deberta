#!/usr/bin/env python3
"""
Rigorous inference demo for GoEmotions DeBERTa-v3-large model
Tests the model on real examples to validate performance claims
"""

import json
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

# Set HF token for authentication
os.environ["HF_TOKEN"] = "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"

# GoEmotions class labels (28 emotions)
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]

def load_model_and_tokenizer():
    """Load the trained LoRA model and tokenizer"""
    print("🤖 Loading DeBERTa-v3-large + LoRA adapter...")
    
    checkpoint_path = "./samo_out/checkpoint-1833"
    model_name = "microsoft/deberta-v3-large"
    
    # Load tokenizer from checkpoint
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=False)
    print("✅ Tokenizer loaded")
    
    # Load base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=28,
        problem_type="multi_label_classification"
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    print("✅ LoRA adapter loaded")
    
    # Set to evaluation mode
    model.eval()
    
    return model, tokenizer

def predict_emotions(model, tokenizer, text, threshold=0.5):
    """Predict emotions for a given text"""
    
    # Tokenize input
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits)
    
    # Apply threshold
    predictions = (probabilities >= threshold).cpu().numpy()[0]
    probabilities = probabilities.cpu().numpy()[0]
    
    # Get predicted emotions
    predicted_emotions = []
    for i, (emotion, pred, prob) in enumerate(zip(EMOTION_LABELS, predictions, probabilities)):
        if pred:
            predicted_emotions.append((emotion, prob))
    
    # Sort by probability
    predicted_emotions.sort(key=lambda x: x[1], reverse=True)
    
    return predicted_emotions, probabilities

def test_inference_examples():
    """Test the model on various example texts"""
    
    print("🧪 RIGOROUS INFERENCE TESTING")
    print("="*60)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer()
    
    # Test examples covering different emotion types
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
    
    # Test different thresholds
    thresholds = [0.3, 0.5, 0.7]
    
    for i, text in enumerate(test_examples, 1):
        print(f"\n📝 Example {i}: \"{text}\"")
        print("-" * 60)
        
        for threshold in thresholds:
            emotions, probs = predict_emotions(model, tokenizer, text, threshold)
            
            print(f"🎯 Threshold {threshold}:")
            if emotions:
                for emotion, prob in emotions:
                    print(f"   • {emotion}: {prob:.3f}")
            else:
                print("   • No emotions predicted")
        
        print()

def test_validation_samples():
    """Test on actual validation samples"""
    
    print("\n🔬 TESTING ON VALIDATION SAMPLES")
    print("="*60)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer()
    
    # Load a few validation samples
    val_samples = []
    with open("./samo_out/val.jsonl", "r") as f:
        for i, line in enumerate(f):
            if i >= 5:  # Test first 5 samples
                break
            val_samples.append(json.loads(line))
    
    for i, sample in enumerate(val_samples, 1):
        text = sample["text"]
        true_labels = sample["labels"]
        
        # Get true emotions
        true_emotions = [EMOTION_LABELS[j] for j, label in enumerate(true_labels) if label == 1]
        
        print(f"\n📝 Validation Sample {i}: \"{text}\"")
        print(f"✅ True emotions: {true_emotions}")
        
        # Test different thresholds
        for threshold in [0.3, 0.5, 0.7]:
            emotions, probs = predict_emotions(model, tokenizer, text, threshold)
            predicted_emotions = [emotion for emotion, _ in emotions]
            
            # Calculate precision and recall
            if predicted_emotions:
                precision = len(set(predicted_emotions) & set(true_emotions)) / len(predicted_emotions)
                recall = len(set(predicted_emotions) & set(true_emotions)) / len(true_emotions) if true_emotions else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            else:
                precision = recall = f1 = 0
            
            print(f"🎯 Threshold {threshold}:")
            print(f"   Predicted: {predicted_emotions}")
            print(f"   Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        
        print()

def analyze_prediction_patterns():
    """Analyze what the model is actually predicting"""
    
    print("\n📊 PREDICTION PATTERN ANALYSIS")
    print("="*60)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer()
    
    # Test on neutral/ambiguous text
    neutral_texts = [
        "The weather is nice today.",
        "I went to the store.",
        "This is a test.",
        "Hello, how are you?",
        "I don't know what to say."
    ]
    
    print("🔍 Testing on neutral/ambiguous texts:")
    for text in neutral_texts:
        emotions, probs = predict_emotions(model, tokenizer, text, threshold=0.3)
        print(f"   \"{text}\" -> {[emotion for emotion, _ in emotions]}")
    
    # Test on strong emotional text
    strong_texts = [
        "I AM SO ANGRY RIGHT NOW!!!",
        "I love you so much! You make me so happy!",
        "This is absolutely terrifying and I'm scared!",
        "I'm so grateful and thankful for everything!",
        "I'm really disappointed and sad about this."
    ]
    
    print("\n🔍 Testing on strong emotional texts:")
    for text in strong_texts:
        emotions, probs = predict_emotions(model, tokenizer, text, threshold=0.3)
        print(f"   \"{text}\" -> {[emotion for emotion, _ in emotions]}")

def main():
    """Main inference demo"""
    
    print("🚀 GoEmotions DeBERTa-v3-large Inference Demo")
    print("🔬 Rigorous testing to validate performance claims")
    print("="*60)
    
    try:
        # Test 1: Inference examples
        test_inference_examples()
        
        # Test 2: Validation samples
        test_validation_samples()
        
        # Test 3: Pattern analysis
        analyze_prediction_patterns()
        
        print("\n" + "="*60)
        print("✅ Inference testing complete!")
        print("📊 Review results above to validate model performance")
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
