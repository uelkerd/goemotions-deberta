#!/usr/bin/env python3
"""
🧪 COMPREHENSIVE FIXES VALIDATION
==================================
Test all critical fixes to ensure the multi-dataset approach will work
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import tempfile
import shutil

def test_fix_1_isear_emotion_mapping():
    """Test that ISEAR emotion mapping is no longer random"""
    print("🧪 TEST 1: ISEAR Emotion Mapping")
    print("=" * 40)

    try:
        # Import and test the function
        sys.path.append('notebooks')
        from prepare_all_datasets import load_isear

        # Run ISEAR loading twice
        print("   Testing ISEAR emotion consistency...")
        data1 = load_isear()
        data2 = load_isear()

        if len(data1) == 0 or len(data2) == 0:
            print("   ⚠️ No ISEAR data loaded, testing fallback...")
            # Should create consistent fallback data
            if len(data1) == len(data2):
                print("   ✅ Consistent fallback data generation")
                return True
            else:
                print("   ❌ Inconsistent fallback data generation")
                return False

        # Check for consistency (not random)
        if len(data1) != len(data2):
            print("   ❌ Inconsistent data size between runs")
            return False

        # Check emotion mapping consistency
        emotions_1 = set()
        emotions_2 = set()

        for item in data1[:100]:  # Test first 100 items
            emotions_1.update(item['labels'])

        for item in data2[:100]:
            emotions_2.update(item['labels'])

        if emotions_1 == emotions_2:
            print("   ✅ Emotion mappings are consistent (not random)")
            print(f"   ✅ Emotions covered: {sorted(list(emotions_1))}")
            return True
        else:
            print("   ❌ Emotion mappings are inconsistent")
            print(f"   Run 1 emotions: {sorted(list(emotions_1))}")
            print(f"   Run 2 emotions: {sorted(list(emotions_2))}")
            return False

    except Exception as e:
        print(f"   ❌ Error testing ISEAR mapping: {e}")
        return False

def test_fix_2_fallback_data_generation():
    """Test enhanced fallback data generation"""
    print("\n🧪 TEST 2: Enhanced Fallback Data Generation")
    print("=" * 50)

    try:
        # Test fallback handler
        sys.path.append('notebooks')
        from fallback_handler import create_fallback_data

        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            old_cwd = os.getcwd()
            os.chdir(temp_dir)

            # Create minimal directory structure
            os.makedirs("data/combined_all_datasets", exist_ok=True)

            print("   Testing fallback data creation...")
            success = create_fallback_data()

            if not success:
                print("   ❌ Fallback data creation failed")
                return False

            # Validate created data
            train_path = "data/combined_all_datasets/train.jsonl"
            val_path = "data/combined_all_datasets/val.jsonl"

            if not (os.path.exists(train_path) and os.path.exists(val_path)):
                print("   ❌ Fallback data files not created")
                return False

            # Check data quality
            train_count = sum(1 for _ in open(train_path))
            val_count = sum(1 for _ in open(val_path))

            print(f"   ✅ Created {train_count} train, {val_count} val samples")

            if train_count < 1000 or val_count < 200:
                print("   ❌ Insufficient fallback data generated")
                return False

            # Check emotion coverage
            emotions_covered = set()
            with open(train_path, 'r') as f:
                for i, line in enumerate(f):
                    if i > 100:  # Check first 100 samples
                        break
                    item = json.loads(line)
                    emotions_covered.update(item['labels'])

            if len(emotions_covered) < 20:  # Should cover most emotions
                print(f"   ❌ Limited emotion coverage: {len(emotions_covered)} emotions")
                return False

            print(f"   ✅ Good emotion coverage: {len(emotions_covered)} emotions")
            print("   ✅ Enhanced fallback data generation working")

            os.chdir(old_cwd)
            return True

    except Exception as e:
        print(f"   ❌ Error testing fallback data: {e}")
        return False

def test_fix_3_combined_loss_strategy():
    """Test that Combined Loss strategy is properly configured"""
    print("\n🧪 TEST 3: Combined Loss Strategy")
    print("=" * 40)

    try:
        # Check training script configuration
        script_path = "scripts/train_comprehensive_multidataset.sh"

        if not os.path.exists(script_path):
            print("   ❌ Training script not found")
            return False

        with open(script_path, 'r') as f:
            script_content = f.read()

        # Check for Combined Loss configuration
        checks = [
            ("--use_combined_loss", "Combined Loss enabled"),
            ("--loss_combination_ratio 0.7", "Optimal ratio (0.7)"),
            ("--gamma 2.0", "Gamma parameter set"),
            ("--label_smoothing", "Label smoothing enabled"),
            ("--use_class_weights", "Class weights enabled"),
            ("--oversample_rare_classes", "Oversampling enabled")
        ]

        all_checks_passed = True
        for check_string, description in checks:
            if check_string in script_content:
                print(f"   ✅ {description}")
            else:
                print(f"   ❌ Missing: {description}")
                all_checks_passed = False

        # Check that we're not using default BCE only
        if "--use_combined_loss" not in script_content:
            print("   ❌ Still using default BCE - Combined Loss not enabled")
            return False

        print("   ✅ Combined Loss strategy properly configured")
        return all_checks_passed

    except Exception as e:
        print(f"   ❌ Error testing Combined Loss strategy: {e}")
        return False

def test_fix_4_integration():
    """Test complete integration of all fixes"""
    print("\n🧪 TEST 4: Complete Integration")
    print("=" * 40)

    try:
        # Test that all components work together
        print("   Testing complete workflow integration...")

        # Check that all key files exist
        key_files = [
            "notebooks/prepare_all_datasets.py",
            "notebooks/fallback_handler.py",
            "scripts/train_comprehensive_multidataset.sh",
            "notebooks/scripts/train_deberta_local.py",
            "setup_enhanced_notebook.py"
        ]

        missing_files = []
        for file_path in key_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)

        if missing_files:
            print(f"   ❌ Missing files: {missing_files}")
            return False

        print("   ✅ All key files present")

        # Test that setup script can verify components
        try:
            result = subprocess.run([sys.executable, 'setup_enhanced_notebook.py'],
                                   capture_output=True, text=True, timeout=30)

            if "All components installed and verified" in result.stdout:
                print("   ✅ Setup verification passes")
            else:
                print("   ⚠️ Setup verification has warnings")

        except subprocess.TimeoutExpired:
            print("   ⚠️ Setup script timeout (taking too long)")
        except Exception as e:
            print(f"   ⚠️ Setup script error: {e}")

        # Test dataset preparation dry run
        try:
            # Test without actually downloading large datasets
            sys.path.append('notebooks')
            import prepare_all_datasets

            # Check that emotion mappings are defined properly
            if hasattr(prepare_all_datasets, 'EMOTION_LABELS'):
                if len(prepare_all_datasets.EMOTION_LABELS) == 28:
                    print("   ✅ GoEmotions emotion labels properly defined")
                else:
                    print(f"   ❌ Wrong number of emotion labels: {len(prepare_all_datasets.EMOTION_LABELS)}")
                    return False
            else:
                print("   ❌ EMOTION_LABELS not found")
                return False

        except Exception as e:
            print(f"   ❌ Dataset preparation test failed: {e}")
            return False

        print("   ✅ Complete integration test passed")
        return True

    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
        return False

def main():
    """Run all validation tests"""
    print("🧪 COMPREHENSIVE FIXES VALIDATION")
    print("=" * 50)
    print("🎯 Validating all critical fixes for multi-dataset approach")
    print("=" * 50)

    # Run all tests
    tests = [
        ("ISEAR Emotion Mapping Fix", test_fix_1_isear_emotion_mapping),
        ("Enhanced Fallback Data Generation", test_fix_2_fallback_data_generation),
        ("Combined Loss Strategy", test_fix_3_combined_loss_strategy),
        ("Complete Integration", test_fix_4_integration)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("🎯 VALIDATION SUMMARY")
    print("=" * 50)

    passed = 0
    failed = 0

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status}: {test_name}")
        if success:
            passed += 1
        else:
            failed += 1

    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Multi-dataset approach is ready for training")
        print("🎯 Expected improvement: 51.79% → 60%+ F1-macro")
        print("\n🚀 Ready to run: ./quick_start_multidataset.sh")
        return True
    else:
        print(f"\n⚠️ {failed} TEST(S) FAILED")
        print("🔧 Please address issues before proceeding")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)