#!/usr/bin/env python3
"""
Setup local cache for GoEmotions dataset and DeBERTa-v3-large model
Downloads and caches everything locally for faster access
"""

import os
import json
import shutil
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Set up environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_TOKEN"] = "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"

def setup_directories():
    """Create necessary directories"""
    print("📁 Setting up directory structure...")
    
    directories = [
        "data/goemotions",
        "models/deberta-v3-large",
        "models/roberta-large", 
        "outputs/deberta",
        "outputs/roberta",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    return True

def cache_goemotions_dataset():
    """Download and cache GoEmotions dataset locally"""
    print("\n📊 Caching GoEmotions dataset...")
    
    try:
        from datasets import load_dataset
        
        # Check if already cached
        cache_path = "data/goemotions"
        if os.path.exists(f"{cache_path}/train.jsonl"):
            print("✅ GoEmotions dataset already cached")
            return True
        
        print("🔄 Downloading GoEmotions dataset...")
        dataset = load_dataset("go_emotions")
        
        # Save train and validation splits
        train_data = dataset["train"]
        val_data = dataset["validation"]
        
        print(f"✅ Downloaded {len(train_data)} training examples")
        print(f"✅ Downloaded {len(val_data)} validation examples")
        
        # Save as JSONL files
        with open(f"{cache_path}/train.jsonl", "w") as f:
            for example in train_data:
                f.write(json.dumps(example) + "\n")
        
        with open(f"{cache_path}/val.jsonl", "w") as f:
            for example in val_data:
                f.write(json.dumps(example) + "\n")
        
        # Save metadata
        metadata = {
            "dataset": "go_emotions",
            "train_size": len(train_data),
            "val_size": len(val_data),
            "total_size": len(train_data) + len(val_data),
            "emotions": train_data.features["labels"].feature.names
        }
        
        with open(f"{cache_path}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ GoEmotions dataset cached to {cache_path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to cache GoEmotions dataset: {e}")
        return False

def cache_deberta_v3_model():
    """Download and cache DeBERTa-v3-large model locally"""
    print("\n🤖 Caching DeBERTa-v3-large model...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
        
        model_name = "microsoft/deberta-v3-large"
        cache_path = "models/deberta-v3-large"
        
        # Check if already cached
        if os.path.exists(f"{cache_path}/config.json"):
            print("✅ DeBERTa-v3-large model already cached")
            return True
        
        print("🔄 Downloading DeBERTa-v3-large model...")
        
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        tokenizer.save_pretrained(cache_path)
        print("✅ Tokenizer cached")
        
        # Download model config
        config = AutoConfig.from_pretrained(model_name)
        config.save_pretrained(cache_path)
        print("✅ Config cached")
        
        # Download model (this is the big one)
        print("🔄 Downloading model weights (this may take a while)...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=28,  # GoEmotions has 28 emotions
            problem_type="multi_label_classification"
        )
        model.save_pretrained(cache_path)
        print("✅ Model weights cached")
        
        # Save metadata
        metadata = {
            "model_name": model_name,
            "model_type": "DeBERTa-v3-large",
            "num_labels": 28,
            "problem_type": "multi_label_classification",
            "cached_at": str(Path().cwd())
        }
        
        with open(f"{cache_path}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ DeBERTa-v3-large model cached to {cache_path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to cache DeBERTa-v3-large model: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up local cache for GoEmotions DeBERTa project")
    print("="*60)
    
    # Setup directories
    if not setup_directories():
        print("❌ Failed to setup directories")
        return False
    
    # Cache datasets and models
    success = True
    
    if not cache_goemotions_dataset():
        success = False
    
    if not cache_deberta_v3_model():
        success = False
    
    if success:
        print("\n🎉 Local cache setup completed successfully!")
        print("📁 All models and datasets are now cached locally")
        print("🚀 Ready for fast training without internet dependency")
    else:
        print("\n⚠️  Local cache setup completed with some issues")
        print("💡 Check the errors above and retry if needed")
    
    return success

if __name__ == "__main__":
    main()
