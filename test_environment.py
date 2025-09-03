#!/usr/bin/env python3
"""
Environment Compatibility Test for DeBERTa-v3-large
Run this script to verify your environment supports DeBERTa-v3-large training
"""

import os
import sys

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")
    
    imports_passed = 0
    total_imports = 6
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        imports_passed += 1
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
        imports_passed += 1
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
    
    try:
        import datasets
        print(f"✅ Datasets: {datasets.__version__}")
        imports_passed += 1
    except ImportError as e:
        print(f"❌ Datasets import failed: {e}")
    
    try:
        import sentencepiece
        print(f"✅ SentencePiece: {sentencepiece.__version__}")
        imports_passed += 1
    except ImportError as e:
        print(f"❌ SentencePiece import failed: {e}")
    
    try:
        import tiktoken
        print(f"✅ Tiktoken: {tiktoken.__version__}")
        imports_passed += 1
    except ImportError as e:
        print(f"❌ Tiktoken import failed: {e}")
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
        imports_passed += 1
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
    
    print(f"📊 Imports: {imports_passed}/{total_imports} passed")
    return imports_passed == total_imports

def test_cuda():
    """Test CUDA availability"""
    print("\n🔍 Testing CUDA...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.device_count()} GPU(s)")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("❌ CUDA not available")
            return False
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
        return False

def test_deberta_tokenizer():
    """Test DeBERTa-v3-large tokenizer loading"""
    print("\n🔍 Testing DeBERTa-v3-large tokenizer...")
    
    try:
        from transformers import AutoTokenizer
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
        print("✅ DeBERTa-v3-large tokenizer loaded successfully!")
        
        # Test tokenization
        test_text = "I love this movie! It's amazing."
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"✅ Tokenization test passed: {tokens['input_ids'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ DeBERTa-v3-large tokenizer test failed: {e}")
        return False

def test_goemotions_dataset():
    """Test GoEmotions dataset loading"""
    print("\n🔍 Testing GoEmotions dataset...")
    
    try:
        from datasets import load_dataset
        import os
        # Set environment variable to avoid pattern issues
        os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
        
        # Try loading with specific configuration to avoid pattern issues
        dataset = load_dataset("go_emotions", trust_remote_code=True)
        print(f"✅ GoEmotions dataset loaded: {len(dataset['train'])} train examples")
        return True
        
    except Exception as e:
        print(f"❌ GoEmotions dataset test failed: {e}")
        # Try alternative loading method
        try:
            print("🔄 Trying alternative dataset loading method...")
            from datasets import load_dataset
            dataset = load_dataset("go_emotions", split="train")
            print(f"✅ GoEmotions dataset loaded (alternative method): {len(dataset)} examples")
            return True
        except Exception as e2:
            print(f"❌ Alternative method also failed: {e2}")
            return False

def test_training_script():
    """Test if training script can be imported"""
    print("\n🔍 Testing training script...")
    
    try:
        # Check if train_samo.py exists
        if not os.path.exists("train_samo.py"):
            print("❌ train_samo.py not found")
            return False
        
        # Try to import the main function
        import importlib.util
        spec = importlib.util.spec_from_file_location("train_samo", "train_samo.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, 'main'):
            print("✅ Training script imported successfully")
            return True
        else:
            print("❌ Training script missing main function")
            return False
            
    except Exception as e:
        print(f"❌ Training script test failed: {e}")
        return False

def main():
    """Run all compatibility tests"""
    print("🚀 DeBERTa-v3-large Environment Compatibility Test")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("CUDA", test_cuda),
        ("DeBERTa Tokenizer", test_deberta_tokenizer),
        ("GoEmotions Dataset", test_goemotions_dataset),
        ("Training Script", test_training_script),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("🎉 All tests passed! Your environment is ready for DeBERTa-v3-large training!")
        print("\nYou can now run:")
        print("accelerate launch --num_processes=2 --mixed_precision=fp16 train_samo.py --output_dir './samo_out' --model_name 'microsoft/deberta-v3-large' --per_device_train_batch_size 8 --per_device_eval_batch_size 16 --gradient_accumulation_steps 4 --num_train_epochs 3 --learning_rate 1e-5 --lr_scheduler_type cosine --warmup_ratio 0.1 --weight_decay 0.01 --fp16 true --tf32 true --gradient_checkpointing true")
    else:
        print("⚠️  Some tests failed. Please check the environment setup guide.")
        print("\nRecommended actions:")
        print("1. Install compatible package versions (see environment_setup_guide.md)")
        print("2. Restart the runtime/kernel")
        print("3. Clear Hugging Face cache: rm -rf ~/.cache/huggingface/")
        print("4. Run this test again")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
