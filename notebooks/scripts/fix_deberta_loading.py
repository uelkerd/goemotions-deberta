#!/usr/bin/env python3
"""
Systematic approach to fix DeBERTa-v3-large loading issues
Tests multiple strategies to overcome tiktoken/SentencePiece problems
"""

import os
import sys
import subprocess
import warnings
warnings.filterwarnings("ignore")

def test_environment():
    """Test current environment and versions"""
    print("🔍 Testing current environment...")
    
    try:
        import transformers
        import torch
        import tiktoken
        import sentencepiece
        
        print(f"✅ Transformers: {transformers.__version__}")
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ Tiktoken: {tiktoken.__version__}")
        print(f"✅ SentencePiece: {sentencepiece.__version__}")
        
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        return False

def strategy_1_force_slow_tokenizer():
    """Strategy 1: Force slow tokenizer and disable fast tokenizer"""
    print("\n🧪 Strategy 1: Force slow tokenizer")
    print("-" * 50)
    
    test_code = '''
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_TOKEN"] = "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"

try:
    from transformers import AutoTokenizer
    print("Testing DeBERTa-v3-large with use_fast=False...")
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/deberta-v3-large", 
        use_fast=False,
        trust_remote_code=True
    )
    print("✅ SUCCESS: Slow tokenizer loaded!")
    
    # Test tokenization
    test_text = "Hello world!"
    tokens = tokenizer(test_text, return_tensors="pt")
    print(f"✅ Tokenization test: {tokens['input_ids'].shape}")
    
except Exception as e:
    print(f"❌ FAILED: {e}")
'''
    
    return run_test_code(test_code, "Strategy 1")

def strategy_2_offline_mode():
    """Strategy 2: Use offline mode and local cache"""
    print("\n🧪 Strategy 2: Offline mode with local cache")
    print("-" * 50)
    
    test_code = '''
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_TOKEN"] = "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"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

try:
    from transformers import AutoTokenizer
    print("Testing DeBERTa-v3-large in offline mode...")
    
    # Try to load from local cache first
    local_path = "/workspace/.hf_home/hub/models--microsoft--deberta-v3-large/snapshots/64a8c8eab3e352a784c658aef62be1662607476f"
    
    if os.path.exists(local_path):
        print(f"Loading from local cache: {local_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            local_path,
            use_fast=False,
            local_files_only=True
        )
    else:
        print("Local cache not found, trying online...")
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/deberta-v3-large",
            use_fast=False
        )
    
    print("✅ SUCCESS: Tokenizer loaded!")
    
    # Test tokenization
    test_text = "Hello world!"
    tokens = tokenizer(test_text, return_tensors="pt")
    print(f"✅ Tokenization test: {tokens['input_ids'].shape}")
    
except Exception as e:
    print(f"❌ FAILED: {e}")
'''
    
    return run_test_code(test_code, "Strategy 2")

def strategy_3_manual_tiktoken_fix():
    """Strategy 3: Manual tiktoken environment fix"""
    print("\n🧪 Strategy 3: Manual tiktoken environment fix")
    print("-" * 50)
    
    test_code = '''
import os
import sys

# Set environment variables to fix tiktoken issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_TOKEN"] = "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"

# Try to fix tiktoken import issues
try:
    import tiktoken
    print(f"Tiktoken version: {tiktoken.__version__}")
    
    # Test tiktoken directly
    enc = tiktoken.get_encoding("cl100k_base")
    test_tokens = enc.encode("Hello world!")
    print(f"✅ Tiktoken test: {len(test_tokens)} tokens")
    
except Exception as e:
    print(f"❌ Tiktoken issue: {e}")

try:
    from transformers import AutoTokenizer
    print("Testing DeBERTa-v3-large with manual tiktoken fix...")
    
    # Try different tokenizer classes
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/deberta-v3-large",
        use_fast=False,
        trust_remote_code=True,
        use_auth_token=True
    )
    
    print("✅ SUCCESS: Tokenizer loaded!")
    
    # Test tokenization
    test_text = "Hello world!"
    tokens = tokenizer(test_text, return_tensors="pt")
    print(f"✅ Tokenization test: {tokens['input_ids'].shape}")
    
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
'''
    
    return run_test_code(test_code, "Strategy 3")

def strategy_4_alternative_model():
    """Strategy 4: Use alternative DeBERTa model"""
    print("\n🧪 Strategy 4: Alternative DeBERTa model")
    print("-" * 50)
    
    test_code = '''
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_TOKEN"] = "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"

try:
    from transformers import AutoTokenizer
    print("Testing microsoft/deberta-large (not v3)...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/deberta-large",
        use_fast=False
    )
    
    print("✅ SUCCESS: DeBERTa-large tokenizer loaded!")
    
    # Test tokenization
    test_text = "Hello world!"
    tokens = tokenizer(test_text, return_tensors="pt")
    print(f"✅ Tokenization test: {tokens['input_ids'].shape}")
    
except Exception as e:
    print(f"❌ FAILED: {e}")
'''
    
    return run_test_code(test_code, "Strategy 4")

def strategy_5_package_downgrade():
    """Strategy 5: Try specific package downgrades"""
    print("\n🧪 Strategy 5: Package downgrade test")
    print("-" * 50)
    
    print("Testing with current versions first...")
    
    # Test current versions
    test_code = '''
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_TOKEN"] = "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"

try:
    from transformers import AutoTokenizer
    print("Testing with current package versions...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/deberta-v3-large",
        use_fast=False,
        trust_remote_code=True
    )
    
    print("✅ SUCCESS: Current versions work!")
    
except Exception as e:
    print(f"❌ Current versions failed: {e}")
    print("Would need to test specific downgrades...")
'''
    
    return run_test_code(test_code, "Strategy 5")

def run_test_code(code, strategy_name):
    """Run test code and return success status"""
    try:
        result = subprocess.run([
            sys.executable, "-c", code
        ], capture_output=True, text=True, timeout=60)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        success = "✅ SUCCESS" in result.stdout
        if success:
            print(f"🎉 {strategy_name} WORKED!")
        else:
            print(f"❌ {strategy_name} failed")
        
        return success
        
    except subprocess.TimeoutExpired:
        print(f"⏰ {strategy_name} timed out")
        return False
    except Exception as e:
        print(f"❌ {strategy_name} error: {e}")
        return False

def main():
    """Main function to test all strategies"""
    print("🚀 DeBERTa-v3-large Loading Fix - Systematic Testing")
    print("="*70)
    
    # Test environment
    if not test_environment():
        print("❌ Environment issues detected")
        return
    
    # Test all strategies
    strategies = [
        strategy_1_force_slow_tokenizer,
        strategy_2_offline_mode,
        strategy_3_manual_tiktoken_fix,
        strategy_4_alternative_model,
        strategy_5_package_downgrade
    ]
    
    successful_strategies = []
    
    for strategy in strategies:
        try:
            success = strategy()
            if success:
                successful_strategies.append(strategy.__name__)
        except Exception as e:
            print(f"❌ Strategy {strategy.__name__} crashed: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("📊 TESTING SUMMARY")
    print("="*70)
    
    if successful_strategies:
        print(f"✅ {len(successful_strategies)} strategies worked:")
        for strategy in successful_strategies:
            print(f"   • {strategy}")
        
        print("\n💡 RECOMMENDATION:")
        print("   Use the first successful strategy to fix the training script")
        
    else:
        print("❌ No strategies worked")
        print("\n💡 NEXT STEPS:")
        print("   1. Check if DeBERTa-v3-large is actually available")
        print("   2. Try different package versions")
        print("   3. Consider using microsoft/deberta-large instead")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
