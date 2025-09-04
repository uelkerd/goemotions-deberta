#!/usr/bin/env python3
"""
Simple inference demo that uses the same approach as the training script
Avoids PEFT import issues by using the training script's model loading method
"""

import json
import os
import subprocess
import sys

# GoEmotions class labels (28 emotions)
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]

def create_inference_script():
    """Create a simple inference script that uses the training environment"""
    
    inference_code = '''
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

def load_model():
    """Load the trained model"""
    print("🤖 Loading DeBERTa-v3-large + LoRA adapter...")
    
    checkpoint_path = "./samo_out/checkpoint-1833"
    model_name = "microsoft/deberta-v3-large"
    
    # Load tokenizer
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
    model.eval()
    print("✅ Model loaded")
    
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
    return predicted_emotions

def main():
    """Run inference tests"""
    print("🚀 GoEmotions Inference Demo")
    print("="*50)
    
    try:
        model, tokenizer = load_model()
        
        # Test examples
        test_texts = [
            "I love this movie! It's absolutely amazing!",
            "This is terrible. I'm so angry and disappointed.",
            "I'm really confused about what happened.",
            "Thank you so much! I'm really grateful.",
            "I'm so excited about the vacation!",
            "This is just okay, nothing special.",
            "I'm really scared about the exam.",
            "Wow, I never expected that!",
            "I feel so proud of my achievement.",
            "I'm really sorry. I feel terrible."
        ]
        
        print("\\n🧪 Testing on example texts:")
        print("-" * 50)
        
        for i, text in enumerate(test_texts, 1):
            print(f"\\n📝 Example {i}: \\"{text}\\"")
            
            for threshold in [0.3, 0.5, 0.7]:
                emotions = predict_emotions(model, tokenizer, text, threshold)
                emotion_names = [emotion for emotion, _ in emotions]
                print(f"   Threshold {threshold}: {emotion_names}")
        
        # Test validation samples
        print("\\n🔬 Testing on validation samples:")
        print("-" * 50)
        
        with open("./samo_out/val.jsonl", "r") as f:
            for i, line in enumerate(f):
                if i >= 3:  # Test first 3
                    break
                
                sample = json.loads(line)
                text = sample["text"]
                true_labels = sample["labels"]
                true_emotions = [EMOTION_LABELS[j] for j, label in enumerate(true_labels) if label == 1]
                
                print(f"\\n📝 Val Sample {i+1}: \\"{text}\\"")
                print(f"   True: {true_emotions}")
                
                emotions = predict_emotions(model, tokenizer, text, threshold=0.3)
                predicted = [emotion for emotion, _ in emotions]
                print(f"   Predicted: {predicted}")
                
                # Calculate overlap
                overlap = set(predicted) & set(true_emotions)
                print(f"   Overlap: {list(overlap)}")
        
        print("\\n✅ Inference testing complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
'''
    
    with open("temp_inference.py", "w") as f:
        f.write(inference_code)
    
    print("✅ Created temporary inference script")

def run_inference_demo():
    """Run the inference demo using the training environment"""
    
    print("🚀 GoEmotions Inference Demo")
    print("🔬 Testing model performance on real examples")
    print("="*60)
    
    # Create the inference script
    create_inference_script()
    
    # Run it using the same environment as training
    print("\\n🧪 Running inference tests...")
    print("-" * 60)
    
    try:
        # Use the same Python environment that was used for training
        result = subprocess.run([
            sys.executable, "temp_inference.py"
        ], capture_output=True, text=True, timeout=300)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\\nSTDERR:")
            print(result.stderr)
        
        if result.returncode != 0:
            print(f"\\n❌ Inference failed with return code {result.returncode}")
        else:
            print("\\n✅ Inference completed successfully!")
            
    except subprocess.TimeoutExpired:
        print("❌ Inference timed out after 5 minutes")
    except Exception as e:
        print(f"❌ Error running inference: {e}")
    
    finally:
        # Clean up
        if os.path.exists("temp_inference.py"):
            os.remove("temp_inference.py")

def main():
    """Main function"""
    run_inference_demo()

if __name__ == "__main__":
    main()
