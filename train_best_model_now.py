#!/usr/bin/env python3
"""
PRAGMATIC TRAINING SCRIPT - NO MORE BS!
We know Combined Loss works best from literature. Let's just train it properly!
"""

import subprocess
import sys
import os

print("🔥 PRAGMATIC TRAINING - NO MORE WASTED TIME!")
print("=" * 60)
print("BCE failed at 5.4% F1. We're going straight to what works!")
print()

# Best configuration based on literature and common sense
# Combined Loss with 70% ASL for extreme imbalance
cmd = [
    sys.executable, 'notebooks/scripts/train_deberta_local.py',
    '--output_dir', './outputs/FINAL_MODEL',
    '--model_type', 'deberta-v3-large',
    '--per_device_train_batch_size', '2',
    '--per_device_eval_batch_size', '4', 
    '--gradient_accumulation_steps', '8',  # Effective batch = 16
    '--num_train_epochs', '3',  # Full training
    '--learning_rate', '2e-5',
    '--lr_scheduler_type', 'cosine',
    '--warmup_ratio', '0.1',
    '--weight_decay', '0.01',
    '--use_combined_loss',
    '--loss_combination_ratio', '0.7',  # 70% ASL, 30% Focal
    '--fp16',
    '--max_length', '256',
    '--evaluation_strategy', 'steps',
    '--eval_steps', '500',
    '--save_strategy', 'steps',
    '--save_steps', '500',
    '--metric_for_best_model', 'f1_macro',
    '--greater_is_better', 'true',
    '--load_best_model_at_end',
    '--save_total_limit', '3',
    '--logging_steps', '50'
]

print("🎯 Training Configuration:")
print("   Model: DeBERTa-v3-large")
print("   Loss: Combined (70% ASL + 30% Focal)")
print("   Epochs: 3 (full training)")
print("   Effective batch size: 16")
print("   Learning rate: 2e-5")
print()
print("📊 Expected Results:")
print("   F1 Macro: 55-65% (10-12x better than BCE)")
print("   Training time: ~3-4 hours")
print()
print("🚀 Starting training NOW...")
print("-" * 60)

# Run it!
result = subprocess.run(cmd)

if result.returncode == 0:
    print()
    print("✅ TRAINING COMPLETE!")
    print("📁 Model saved to: ./outputs/FINAL_MODEL")
    print("🎯 Next step: Run inference on test set")
else:
    print("❌ Training failed. Check logs.")
    sys.exit(1)
