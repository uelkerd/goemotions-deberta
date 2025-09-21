#!/usr/bin/env python3
"""
⚡ QUICK VALIDATION TEST
======================
Fast validation of our performance improvements (5-10 minutes)

TESTS:
1. Environment check (30 seconds)
2. Threshold optimization (2-3 minutes)
3. Immediate fixes validation (5 minutes)
4. Dual GPU utilization (2 minutes)

GOAL: Rapid validation before full optimization
"""

import os
import subprocess
import json
import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_environment_check():
    """Quick environment validation"""
    logger.info("🔍 QUICK ENVIRONMENT CHECK")

    checks = []

    # GPU check
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=count', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True)
        gpu_count = int(result.stdout.strip())
        checks.append(f"✅ {gpu_count} GPUs detected")
    except:
        checks.append("❌ GPU detection failed")

    # Key files check
    key_files = [
        'notebooks/scripts/train_deberta_local.py',
        'immediate_performance_fixes.py',
        'quick_threshold_optimizer.py',
        'data/combined_all_datasets/train.jsonl'
    ]

    for file_path in key_files:
        if Path(file_path).exists():
            checks.append(f"✅ {file_path}")
        else:
            checks.append(f"❌ {file_path} missing")

    for check in checks:
        logger.info(f"   {check}")

    return all("✅" in check for check in checks)

def quick_threshold_test():
    """Quick threshold optimization test"""
    logger.info("🎯 QUICK THRESHOLD OPTIMIZATION TEST")

    try:
        start_time = time.time()
        result = subprocess.run([
            'python3', 'quick_threshold_optimizer.py'
        ], capture_output=True, text=True, timeout=300)  # 5 min timeout

        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            logger.info(f"✅ Threshold optimization completed in {elapsed_time:.1f}s")

            # Check if config was created
            if Path("configs/optimal_thresholds.json").exists():
                logger.info("✅ Optimal thresholds generated")
                return True
            else:
                logger.warning("⚠️ No thresholds file created")
                return False
        else:
            logger.error(f"❌ Threshold optimization failed")
            logger.error(f"Error: {result.stderr[-200:]}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("⏰ Threshold optimization timed out")
        return False
    except Exception as e:
        logger.error(f"💥 Threshold test crashed: {str(e)}")
        return False

def quick_training_test():
    """Quick training test with immediate fixes"""
    logger.info("🔧 QUICK TRAINING TEST (Immediate Fixes)")

    output_dir = "./outputs/quick_training_test"

    try:
        cmd = [
            'python3', 'notebooks/scripts/train_deberta_local.py',
            '--output_dir', output_dir,
            '--model_type', 'deberta-v3-large',

            # Immediate fixes
            '--learning_rate', '2e-5',
            '--lr_scheduler_type', 'polynomial',
            '--warmup_ratio', '0.2',
            '--per_device_train_batch_size', '3',
            '--gradient_accumulation_steps', '3',
            '--weight_decay', '0.005',
            '--fp16',

            # Quick test settings
            '--max_train_samples', '3000',
            '--max_eval_samples', '500',
            '--num_train_epochs', '1',
            '--threshold', '0.2',
            '--early_stopping_patience', '1',
        ]

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0,1'

        logger.info("⚡ Running quick training test (expected: 3-5 minutes)...")
        start_time = time.time()

        result = subprocess.run(
            cmd, env=env, timeout=600,  # 10 min timeout
            capture_output=True, text=True
        )

        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            logger.info(f"✅ Quick training completed in {elapsed_time/60:.1f} minutes")

            # Check results
            eval_file = Path(output_dir) / 'eval_report.json'
            if eval_file.exists():
                with open(eval_file, 'r') as f:
                    data = json.load(f)

                f1_macro = data.get('f1_macro', 0.0)
                baseline_f1 = 0.5179
                regression_f1 = 0.3943

                improvement_vs_regression = ((f1_macro - regression_f1) / regression_f1) * 100

                logger.info(f"📊 Quick Training Results:")
                logger.info(f"   F1-macro: {f1_macro:.4f}")
                logger.info(f"   vs Regression: {improvement_vs_regression:+.1f}%")

                success = f1_macro > regression_f1
                if success:
                    logger.info("✅ Performance improved over regression!")
                else:
                    logger.warning("⚠️ Still below regression baseline")

                return success, f1_macro, improvement_vs_regression
            else:
                logger.error("❌ No evaluation results found")
                return False, 0.0, 0.0
        else:
            logger.error(f"❌ Quick training failed")
            logger.error(f"Error: {result.stderr[-300:]}")
            return False, 0.0, 0.0

    except subprocess.TimeoutExpired:
        logger.error("⏰ Quick training timed out")
        return False, 0.0, 0.0
    except Exception as e:
        logger.error(f"💥 Quick training crashed: {str(e)}")
        return False, 0.0, 0.0

def quick_gpu_test():
    """Quick dual GPU utilization test"""
    logger.info("🚀 QUICK GPU UTILIZATION TEST")

    try:
        # Check initial GPU state
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)

        if result.returncode == 0:
            utils = [int(x.strip()) for x in result.stdout.strip().split('\\n')]
            logger.info(f"📊 GPU count detected: {len(utils)}")
            logger.info(f"   Current utilization: {utils}")

            if len(utils) >= 2:
                logger.info("✅ Dual GPU setup confirmed")
                return True
            else:
                logger.warning("⚠️ Only single GPU detected")
                return False
        else:
            logger.error("❌ GPU utilization check failed")
            return False

    except Exception as e:
        logger.error(f"💥 GPU test crashed: {str(e)}")
        return False

def run_quick_validation():
    """Run all quick validation tests"""
    logger.info("⚡ QUICK VALIDATION TEST SUITE")
    logger.info("=" * 50)
    logger.info("🎯 Goal: Validate improvements in 5-10 minutes")

    start_time = time.time()

    # Test 1: Environment
    logger.info("\\n1️⃣ Environment Check...")
    env_ok = quick_environment_check()

    if not env_ok:
        logger.error("❌ Environment check failed - stopping tests")
        return False

    # Test 2: Threshold Optimization
    logger.info("\\n2️⃣ Threshold Optimization...")
    threshold_ok = quick_threshold_test()

    # Test 3: GPU Setup
    logger.info("\\n3️⃣ GPU Setup...")
    gpu_ok = quick_gpu_test()

    # Test 4: Training with Fixes
    logger.info("\\n4️⃣ Training with Immediate Fixes...")
    training_ok, f1_score, improvement = quick_training_test()

    total_time = time.time() - start_time

    # Results Summary
    logger.info("\\n" + "="*50)
    logger.info("⚡ QUICK VALIDATION RESULTS")
    logger.info("="*50)

    tests = [
        ("Environment", env_ok),
        ("Threshold Optimization", threshold_ok),
        ("Dual GPU Setup", gpu_ok),
        ("Training with Fixes", training_ok)
    ]

    passed = sum(1 for _, success in tests if success)
    total = len(tests)

    logger.info(f"📊 Tests passed: {passed}/{total}")
    for test_name, success in tests:
        status = "✅" if success else "❌"
        logger.info(f"   {status} {test_name}")

    if training_ok:
        logger.info(f"\\n📈 PERFORMANCE RESULTS:")
        logger.info(f"   F1-macro: {f1_score:.4f}")
        logger.info(f"   Improvement: {improvement:+.1f}%")

    logger.info(f"\\n⏱️ Total validation time: {total_time/60:.1f} minutes")

    # Recommendations
    if passed == total and training_ok:
        logger.info("\\n🎉 ALL QUICK TESTS PASSED!")
        logger.info("🚀 Ready for comprehensive optimization!")
        logger.info("💡 Next: Run full optimization framework for 60% F1 target")
    elif passed >= 3:
        logger.info("\\n📈 MOSTLY SUCCESSFUL!")
        logger.info("🔧 Minor issues - proceed with caution")
        logger.info("💡 Consider fixing issues then run comprehensive optimization")
    else:
        logger.info("\\n⚠️ MULTIPLE ISSUES DETECTED!")
        logger.info("🔍 Fix critical problems before optimization")
        logger.info("💡 Review failed tests and resolve issues")

    return passed == total

def main():
    """Execute quick validation"""
    success = run_quick_validation()

    if success:
        logger.info("\\n✅ QUICK VALIDATION SUCCESSFUL!")
        logger.info("🚀 System ready for full optimization!")
    else:
        logger.info("\\n⚠️ VALIDATION ISSUES DETECTED!")
        logger.info("🔧 Resolve issues before proceeding")

if __name__ == "__main__":
    main()