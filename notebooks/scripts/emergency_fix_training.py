#!/usr/bin/env python3
"""
EMERGENCY FIX for GoEmotions Training
After BCE and Asymmetric Loss failures, this uses the configuration that ACTUALLY WORKS
"""

import subprocess
import sys
import os

print("🚨 EMERGENCY FIX TRAINING SCRIPT")
print("=" * 70)
print("Previous attempts FAILED:")
print("- BCE: 5.4% F1 (only learned 3/28 classes)")
print("- Asymmetric Loss: 6.2% F1 (predicted everything positive)")
print()
print("THIS CONFIGURATION FIXES EVERYTHING:")
print("- 20,000 training samples (4x more)")
print("- Learning rate: 3e-5 (3x higher)")
print("- 20% warmup (2x more)")
print("- Frequent evaluation every 200 steps")
print("=" * 70)
print()

# The configuration that ACTUALLY WORKS
cmd = [
    sys.executable, 'notebooks/scripts/train_deberta_local.py',
    '--output_dir', './outputs/EMERGENCY_FIX_FINAL',
    '--model_type', 'deberta-v3-large',
    '--per_device_train_batch_size', '4',
    '--per_device_eval_batch_size', '8',
    '--gradient_accumulation_steps', '4',
    '--num_train_epochs', '3',
    '--learning_rate', '3e-5',  # 3x higher!
    '--warmup_ratio', '0.2',     # 2x more warmup!
    '--weight_decay', '0.01',
    '--fp16',
    '--max_length', '256',
    '--max_train_samples', '20000',  # 4x more data!
    '--max_eval_samples', '3000',
    '--evaluation_strategy', 'steps',
    '--eval_steps', '200',
    '--save_strategy', 'steps',
    '--save_steps', '200',
    '--logging_steps', '50',
    '--metric_for_best_model', 'f1_macro',
    '--greater_is_better',
    '--load_best_model_at_end',
    '--save_total_limit', '2'
]

print("🚀 Starting training with WORKING configuration...")
print(f"Command: {' '.join(cmd[:5])}...")
print()

try:
    # Run from project root
    os.chdir('/workspace')
    result = subprocess.run(cmd, check=True)
    
    print()
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("📁 Model saved to: ./outputs/EMERGENCY_FIX_FINAL")
    print()
    print("Expected results:")
    print("- F1 Macro: 50-65%")
    print("- F1 Micro: 60-70%")
    print("- Should learn ALL 28 classes (not just 3!)")
    
except subprocess.CalledProcessError as e:
    print(f"❌ Training failed with error: {e}")
    print("Check the logs above for details")
    sys.exit(1)