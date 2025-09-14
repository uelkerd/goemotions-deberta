#!/usr/bin/env python3
"""
🎯 BASELINE REPRODUCTION SCRIPT
===============================
Reproduce the 51.79% F1-macro baseline from your published HuggingFace model
duelker/samo-goemotions-deberta-v3-large before proceeding with multi-dataset training
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import time
from datetime import datetime

def check_baseline_requirements():
    """Check requirements for baseline reproduction"""
    print("🔍 BASELINE REQUIREMENTS CHECK")
    print("=" * 40)

    requirements = []

    # Check 1: GoEmotions dataset
    if os.path.exists("data/goemotions/train.jsonl"):
        train_count = sum(1 for _ in open("data/goemotions/train.jsonl"))
        requirements.append(f"✅ GoEmotions train: {train_count} samples")
    else:
        requirements.append("❌ GoEmotions train: Missing")

    if os.path.exists("data/goemotions/val.jsonl"):
        val_count = sum(1 for _ in open("data/goemotions/val.jsonl"))
        requirements.append(f"✅ GoEmotions val: {val_count} samples")
    else:
        requirements.append("❌ GoEmotions val: Missing")

    # Check 2: Training script
    if os.path.exists("notebooks/scripts/train_deberta_local.py"):
        requirements.append("✅ Training script: Available")
    else:
        requirements.append("❌ Training script: Missing")

    # Check 3: Model cache
    if os.path.exists("models/deberta-v3-large"):
        requirements.append("✅ Model cache: Available")
    else:
        requirements.append("⚠️ Model cache: Will download")

    for req in requirements:
        print(f"   {req}")

    missing_count = sum(1 for req in requirements if req.startswith("❌"))
    return missing_count == 0

def create_baseline_training_command():
    """Create the exact training command used for the successful baseline"""
    print("\\n🎯 BASELINE TRAINING CONFIGURATION")
    print("=" * 45)

    # These are the exact parameters from your successful 51.79% model
    baseline_config = {
        'model_type': 'deberta-v3-large',
        'output_dir': 'checkpoints_baseline_reproduction',
        'num_train_epochs': 3,
        'per_device_train_batch_size': 4,
        'per_device_eval_batch_size': 8,
        'gradient_accumulation_steps': 4,
        'learning_rate': '3e-5',
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'fp16': True,
        'max_length': 256,
        'loss_function': 'BCE',  # Your proven winner
        'threshold': 0.2,        # Optimal for GoEmotions
        'label_smoothing': 0.0,
        'early_stopping_patience': 3
    }

    print("📊 Baseline Configuration (51.79% F1-macro):")
    for key, value in baseline_config.items():
        print(f"   {key}: {value}")

    # Create training command
    cmd = [
        'python3', 'notebooks/scripts/train_deberta_local.py',
        '--output_dir', baseline_config['output_dir'],
        '--model_type', baseline_config['model_type'],
        '--per_device_train_batch_size', str(baseline_config['per_device_train_batch_size']),
        '--per_device_eval_batch_size', str(baseline_config['per_device_eval_batch_size']),
        '--gradient_accumulation_steps', str(baseline_config['gradient_accumulation_steps']),
        '--num_train_epochs', str(baseline_config['num_train_epochs']),
        '--learning_rate', baseline_config['learning_rate'],
        '--lr_scheduler_type', 'cosine',
        '--warmup_ratio', str(baseline_config['warmup_ratio']),
        '--weight_decay', str(baseline_config['weight_decay']),
        '--fp16',
        '--max_length', str(baseline_config['max_length']),
        '--augment_prob', '0.0',
        '--label_smoothing', str(baseline_config['label_smoothing']),
        '--early_stopping_patience', str(baseline_config['early_stopping_patience'])
        # Note: Using default BCE loss (no --use_asymmetric_loss or --use_combined_loss)
    ]

    return cmd, baseline_config

def run_baseline_reproduction():
    """Run the baseline reproduction"""
    print("\\n🚀 BASELINE REPRODUCTION")
    print("=" * 30)

    # Get training command
    cmd, config = create_baseline_training_command()

    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)

    # Create log file
    log_file = "logs/baseline_reproduction.log"
    os.makedirs("logs", exist_ok=True)

    print(f"📊 Training command:")
    print(f"   {' '.join(cmd)}")
    print(f"📝 Logging to: {log_file}")
    print(f"⏱️ Expected time: ~2-3 hours")
    print(f"🎯 Expected F1-macro: ~51.79%")

    # Set environment
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'  # Use single GPU for reproducibility

    print("\\n🚀 Starting baseline reproduction...")
    print("=" * 40)

    start_time = time.time()

    # Run training
    with open(log_file, 'w') as f:
        f.write(f"Baseline Reproduction Started: {datetime.now()}\\n")
        f.write(f"Command: {' '.join(cmd)}\\n\\n")

    try:
        result = subprocess.run(cmd, env=env, cwd=os.getcwd())

        elapsed = time.time() - start_time
        hours = elapsed / 3600

        print(f"\\n⏱️ Training completed in {hours:.1f} hours")

        if result.returncode == 0:
            print("✅ Training completed successfully!")
            return True
        else:
            print(f"❌ Training failed with exit code: {result.returncode}")
            return False

    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        return False

def analyze_baseline_results():
    """Analyze the baseline results"""
    print("\\n📊 BASELINE RESULTS ANALYSIS")
    print("=" * 40)

    results_file = "checkpoints_baseline_reproduction/eval_report.json"

    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        return False

    try:
        with open(results_file, 'r') as f:
            results = json.load(f)

        f1_macro = results.get('f1_macro', 0.0)
        f1_micro = results.get('f1_micro', 0.0)
        f1_weighted = results.get('f1_weighted', 0.0)

        print(f"📈 BASELINE RESULTS:")
        print(f"   F1 Macro: {f1_macro:.4f}")
        print(f"   F1 Micro: {f1_micro:.4f}")
        print(f"   F1 Weighted: {f1_weighted:.4f}")

        # Compare with expected baseline
        expected_f1_macro = 0.5179
        difference = f1_macro - expected_f1_macro
        percentage_diff = (difference / expected_f1_macro) * 100

        print(f"\\n🎯 BASELINE VALIDATION:")
        print(f"   Expected F1-macro: {expected_f1_macro:.4f}")
        print(f"   Achieved F1-macro: {f1_macro:.4f}")
        print(f"   Difference: {difference:+.4f} ({percentage_diff:+.1f}%)")

        if abs(percentage_diff) <= 5:  # Within 5% is acceptable
            print("✅ BASELINE REPRODUCED: Within acceptable range!")
            print("🚀 Ready to proceed with multi-dataset training")
            return True
        elif f1_macro >= expected_f1_macro:
            print("🎉 BASELINE EXCEEDED: Better than expected!")
            print("🚀 Excellent foundation for multi-dataset training")
            return True
        else:
            print("⚠️ BASELINE BELOW EXPECTED: Need to investigate")
            print("🔧 Check training logs and configuration")
            return False

    except Exception as e:
        print(f"❌ Error reading results: {e}")
        return False

def create_baseline_summary():
    """Create a summary of baseline reproduction"""
    print("\\n📋 BASELINE SUMMARY")
    print("=" * 25)

    summary_file = "baseline_reproduction_summary.json"

    try:
        with open("checkpoints_baseline_reproduction/eval_report.json", 'r') as f:
            results = json.load(f)

        summary = {
            "baseline_reproduction": {
                "timestamp": datetime.now().isoformat(),
                "model": "microsoft/deberta-v3-large",
                "dataset": "GoEmotions (28 emotions)",
                "loss_function": "BCE",
                "results": {
                    "f1_macro": results.get('f1_macro', 0.0),
                    "f1_micro": results.get('f1_micro', 0.0),
                    "f1_weighted": results.get('f1_weighted', 0.0)
                },
                "target": {
                    "f1_macro": 0.5179,
                    "source": "duelker/samo-goemotions-deberta-v3-large"
                },
                "status": "success" if results.get('f1_macro', 0) >= 0.50 else "needs_review"
            }
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"✅ Summary saved: {summary_file}")

    except Exception as e:
        print(f"⚠️ Could not create summary: {e}")

def main():
    """Main baseline reproduction workflow"""
    print("🎯 BASELINE REPRODUCTION SCRIPT")
    print("=" * 40)
    print("🎯 Goal: Reproduce 51.79% F1-macro baseline")
    print("📊 Model: DeBERTa-v3-large with BCE loss")
    print("🔬 Purpose: Validate foundation before multi-dataset training")
    print("=" * 40)

    # Step 1: Check requirements
    if not check_baseline_requirements():
        print("❌ Requirements not met. Please address issues above.")
        return False

    # Step 2: Run baseline reproduction
    print("\\n🚀 STARTING BASELINE REPRODUCTION")
    print("⚠️ This will take 2-3 hours...")

    # Uncomment the line below to actually run training
    # For now, we'll skip to avoid long execution
    print("🔄 Training execution prepared (uncomment to run)")
    # success = run_baseline_reproduction()

    # For demonstration, assume success
    success = True

    if success:
        print("✅ Baseline reproduction would complete here")
        # analyze_baseline_results()
        # create_baseline_summary()

    print("\\n🎯 NEXT STEPS:")
    print("   1. Uncomment training execution when ready")
    print("   2. Monitor training with: watch -n 5 'nvidia-smi'")
    print("   3. Check results: checkpoints_baseline_reproduction/")
    print("   4. If successful, proceed with multi-dataset training")

    print("\\n✅ BASELINE REPRODUCTION SCRIPT READY!")
    return True

if __name__ == "__main__":
    main()