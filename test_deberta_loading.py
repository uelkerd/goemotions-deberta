#!/usr/bin/env python3
"""
Test DeBERTa-v3-large loading without PEFT dependencies
"""

import os
import json
import torch
import warnings
warnings.filterwarnings("ignore")

# Set environment variables for DeBERTa-v3-large compatibility
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_TOKEN"] = "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"
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # Enable offline mode

def test_deberta_loading():
    """Test DeBERTa-v3-large loading using the working offline approach"""
    print("🧪 Testing DeBERTa-v3-large loading...")
    
    # Local cache path
    local_path = "/workspace/.hf_home/hub/models--microsoft--deberta-v3-large/snapshots/64a8c8eab3e352a784c658aef62be1662607476f"
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
        
        if os.path.exists(local_path):
            print(f"✅ Local cache found: {local_path}")
            
            # Test tokenizer loading
            print("🔄 Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                local_path,
                use_fast=False,
                local_files_only=True
            )
            print("✅ Tokenizer loaded successfully!")
            
            # Test tokenization
            test_text = "Hello world! This is a test."
            tokens = tokenizer(test_text, return_tensors="pt")
            print(f"✅ Tokenization test: {tokens['input_ids'].shape}")
            
            # Test model loading
            print("🔄 Loading model...")
            config = AutoConfig.from_pretrained(
                local_path,
                num_labels=28,
                problem_type="multi_label_classification"
            )
            
            model = AutoModelForSequenceClassification.from_pretrained(
                local_path,
                config=config
            )
            print("✅ DeBERTa-v3-large model loaded successfully!")
            
            # Test model inference
            print("🔄 Testing model inference...")
            with torch.no_grad():
                outputs = model(**tokens)
                logits = outputs.logits
                print(f"✅ Model inference test: {logits.shape}")
            
            print("\n🎉 SUCCESS: DeBERTa-v3-large is working!")
            print("🚀 Ready to train with the powerful model!")
            
            return True
            
        else:
            print(f"❌ Local cache not found at: {local_path}")
            return False
        
    except Exception as e:
        print(f"❌ Failed to load DeBERTa-v3-large: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🚀 DeBERTa-v3-large Loading Test")
    print("="*50)
    
    success = test_deberta_loading()
    
    if success:
        print("\n✅ CONCLUSION: DeBERTa-v3-large loading works!")
        print("💡 We can now create a training script without PEFT")
        print("🎯 Expected performance improvement over RoBERTa-large")
    else:
        print("\n❌ CONCLUSION: DeBERTa-v3-large loading failed")
        print("💡 Need to investigate further or use alternative approach")

if __name__ == "__main__":
    main()
