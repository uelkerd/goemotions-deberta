#!/usr/bin/env python3
"""
Emergency training script to fix the BCE baseline catastrophe
Using Asymmetric Loss and Combined Loss with better hyperparameters
"""

import subprocess
import os
import sys
import time

print("🚨 EMERGENCY TRAINING SCRIPT")
print("=" * 60)
print("Fixing the BCE baseline failure (F1=5.4%)")
print("Running improved loss functions for extreme class imbalance")
print()

# Create outputs directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Configuration for improved training
configs = [
    {
        'name': 'asymmetric_loss_v2',
        'description': 'Asymmetric Loss - Designed for extreme imbalance',
        'args': [
            '--use_asymmetric_loss',
            '--learning_rate', '2e-5',
            '--max_train_samples', '15000',  # Use more data
            '--max_eval_samples', '3000'
        ]
    },
    {
        'name': 'combined_loss_70',
        'description': 'Combined Loss 70% ASL - Heavy imbalance handling',
        'args': [
            '--use_combined_loss',
            '--loss_combination_ratio', '0.7',
            '--learning_rate', '2e-5',
            '--max_train_samples', '15000',
            '--max_eval_samples', '3000'
        ]
    }
]

# Base training arguments
base_args = [
    sys.executable, 'notebooks/scripts/train_deberta_local.py',
    '--model_type', 'deberta-v3-large',
    '--per_device_train_batch_size', '2',
    '--per_device_eval_batch_size', '4',
    '--gradient_accumulation_steps', '4',
    '--num_train_epochs', '2',  # Train for 2 epochs
    '--lr_scheduler_type', 'cosine',
    '--warmup_ratio', '0.15',  # More warmup
    '--weight_decay', '0.01',
    '--fp16',
    '--max_length', '256',
    '--evaluation_strategy', 'epoch',
    '--save_strategy', 'epoch',
    '--metric_for_best_model', 'f1_macro',
    '--greater_is_better', 'true',
    '--load_best_model_at_end',
    '--save_total_limit', '2'
]

print("🚀 Starting improved training configurations...")
print()

for i, config in enumerate(configs, 1):
    print(f"📊 CONFIG {i}/{len(configs)}: {config['name']}")
    print(f"   {config['description']}")
    print("-" * 60)
    
    # Build full command
    cmd = base_args + [
        '--output_dir', f"./outputs/emergency_{config['name']}"
    ] + config['args']
    
    print(f"   Command: {' '.join(cmd[:5])}...")
    print()
    
    # Run training
    start_time = time.time()
    print(f"   ⏱️  Starting at {time.strftime('%H:%M:%S')}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per config
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"   ✅ SUCCESS in {duration/60:.1f} minutes")
            
            # Try to extract key metrics from output
            output_lines = result.stdout.split('\n')
            for line in output_lines[-20:]:  # Check last 20 lines
                if 'Final F1 Macro' in line:
                    print(f"   📈 {line.strip()}")
                elif 'eval_f1_macro' in line and ':' in line:
                    try:
                        f1_value = float(line.split(':')[-1].strip())
                        print(f"   📈 F1 Macro: {f1_value:.4f}")
                        if f1_value > 0.4:
                            print(f"   🎯 MAJOR IMPROVEMENT! ({f1_value/0.054:.1f}x better than BCE)")
                    except:
                        pass
        else:
            print(f"   ❌ FAILED after {duration/60:.1f} minutes")
            print(f"   Error (last 500 chars): {result.stderr[-500:]}")
            
    except subprocess.TimeoutExpired:
        print(f"   ⏰ TIMEOUT after 60 minutes")
    except Exception as e:
        print(f"   ❌ ERROR: {str(e)}")
    
    print()

print("=" * 60)
print("🏁 EMERGENCY TRAINING COMPLETE")
print()
print("Next steps:")
print("1. Check outputs/emergency_* directories for results")
print("2. Compare F1 scores vs BCE baseline (0.054)")
print("3. Use the best configuration for full training")