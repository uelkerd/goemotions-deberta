#!/usr/bin/env python3
"""
Direct inference using the training script's approach
Avoids PEFT import issues by using the same method as train_samo.py
"""

import json
import os
import subprocess
import sys

def create_direct_inference_script():
    """Create inference script that uses the training script's model loading"""
    
    inference_code = '''
import json
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
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

def load_model_direct():
    """Load model using the same approach as training script"""
    print("🤖 Loading model directly from checkpoint...")
    
    checkpoint_path = "./samo_out/checkpoint-1833"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=False)
    print("✅ Tokenizer loaded")
    
    # Try to load the model directly (this might work if it's a full model)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_path,
            num_labels=28,
            problem_type="multi_label_classification"
        )
        print("✅ Model loaded directly")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Direct loading failed: {e}")
        return None, tokenizer

def test_tokenizer_only(tokenizer):
    """Test just the tokenizer to see if it works"""
    print("\\n🧪 Testing tokenizer only...")
    
    test_texts = [
        "I love this movie!",
        "This is terrible.",
        "I'm confused.",
        "Thank you so much!",
        "I'm excited!"
    ]
    
    for text in test_texts:
        try:
            tokens = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt")
            print(f"✅ \\"{text}\\" -> {tokens['input_ids'].shape}")
        except Exception as e:
            print(f"❌ \\"{text}\\" -> Error: {e}")

def analyze_checkpoint():
    """Analyze what's in the checkpoint"""
    print("\\n🔍 Analyzing checkpoint contents...")
    
    checkpoint_path = "./samo_out/checkpoint-1833"
    
    if os.path.exists(checkpoint_path):
        files = os.listdir(checkpoint_path)
        print(f"📁 Checkpoint contains {len(files)} files:")
        for file in sorted(files):
            size = os.path.getsize(os.path.join(checkpoint_path, file))
            print(f"   • {file} ({size:,} bytes)")
        
        # Check if it's a LoRA checkpoint
        if "adapter_config.json" in files:
            print("\\n🔧 This is a LoRA checkpoint")
            try:
                with open(os.path.join(checkpoint_path, "adapter_config.json"), "r") as f:
                    config = json.load(f)
                print(f"   • Base model: {config.get('base_model_name_or_path', 'Unknown')}")
                print(f"   • Task type: {config.get('task_type', 'Unknown')}")
                print(f"   • Target modules: {config.get('target_modules', 'Unknown')}")
            except Exception as e:
                print(f"   • Error reading config: {e}")
        else:
            print("\\n🤖 This appears to be a full model checkpoint")
    else:
        print("❌ Checkpoint directory not found")

def test_validation_data():
    """Test loading validation data"""
    print("\\n📊 Testing validation data...")
    
    val_file = "./samo_out/val.jsonl"
    if os.path.exists(val_file):
        with open(val_file, "r") as f:
            samples = []
            for i, line in enumerate(f):
                if i >= 3:  # Test first 3 samples
                    break
                samples.append(json.loads(line))
        
        print(f"✅ Loaded {len(samples)} validation samples")
        
        for i, sample in enumerate(samples, 1):
            text = sample["text"]
            labels = sample["labels"]
            true_emotions = [EMOTION_LABELS[j] for j, label in enumerate(labels) if label == 1]
            print(f"   Sample {i}: \\"{text[:50]}...\\" -> {true_emotions}")
    else:
        print("❌ Validation file not found")

def main():
    """Main function"""
    print("🚀 Direct Inference Test")
    print("="*50)
    
    # Analyze checkpoint
    analyze_checkpoint()
    
    # Test validation data
    test_validation_data()
    
    # Try to load model
    model, tokenizer = load_model_direct()
    
    if tokenizer:
        test_tokenizer_only(tokenizer)
    
    if model:
        print("\\n✅ Model loaded successfully - ready for inference!")
    else:
        print("\\n❌ Model loading failed - need to investigate LoRA loading")
    
    print("\\n" + "="*50)
    print("✅ Direct inference test complete!")

if __name__ == "__main__":
    main()
'''
    
    with open("temp_direct_inference.py", "w") as f:
        f.write(inference_code)
    
    print("✅ Created direct inference script")

def run_direct_inference():
    """Run the direct inference test"""
    
    print("🚀 Direct Inference Test")
    print("🔬 Testing model loading and basic functionality")
    print("="*60)
    
    # Create the inference script
    create_direct_inference_script()
    
    # Run it
    print("\\n🧪 Running direct inference test...")
    print("-" * 60)
    
    try:
        result = subprocess.run([
            sys.executable, "temp_direct_inference.py"
        ], capture_output=True, text=True, timeout=120)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\\nSTDERR:")
            print(result.stderr)
        
        if result.returncode != 0:
            print(f"\\n❌ Test failed with return code {result.returncode}")
        else:
            print("\\n✅ Direct inference test completed!")
            
    except subprocess.TimeoutExpired:
        print("❌ Test timed out after 2 minutes")
    except Exception as e:
        print(f"❌ Error running test: {e}")
    
    finally:
        # Clean up
        if os.path.exists("temp_direct_inference.py"):
            os.remove("temp_direct_inference.py")

def main():
    """Main function"""
    run_direct_inference()

if __name__ == "__main__":
    main()
