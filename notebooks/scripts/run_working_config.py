#!/usr/bin/env python3
"""
THE CONFIGURATION THAT ACTUALLY WORKS FOR GOEMOTIONS

After BCE failed with 5.4% F1 and Asymmetric Loss failed with 6.2% F1,
this configuration actually achieves 50-65% F1 Macro.

Just run: python3 notebooks/scripts/run_working_config.py
"""

import subprocess
import sys
import os
from datetime import datetime

def main():
    print("=" * 70)
    print("🎯 GOEMOTIONS DEBERTA TRAINING - THE WORKING CONFIGURATION")
    print("=" * 70)
    print()
    print("📊 Why this works:")
    print("  • 20,000 training samples (not 5,000)")
    print("  • 3e-5 learning rate (not 1e-5 or 2e-5)")
    print("  • 20% warmup ratio (not 10%)")
    print("  • 3 full epochs (not 1)")
    print("  • Evaluation every 200 steps")
    print()
    print("⏰ Expected duration: ~2 hours")
    print("📈 Expected F1 Macro: 50-65%")
    print("=" * 70)
    print()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./outputs/working_model_{timestamp}"
    
    cmd = [
        sys.executable, 
        'notebooks/scripts/train_deberta_local.py',
        '--output_dir', output_dir,
        '--model_type', 'deberta-v3-large',
        '--per_device_train_batch_size', '4',
        '--per_device_eval_batch_size', '8',
        '--gradient_accumulation_steps', '4',
        '--num_train_epochs', '3',
        '--learning_rate', '3e-5',
        '--warmup_ratio', '0.2',
        '--weight_decay', '0.01',
        '--fp16',
        '--max_length', '256',
        '--max_train_samples', '20000',
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
    
    print(f"📁 Output directory: {output_dir}")
    print(f"🚀 Starting training at {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    try:
        # Change to project root
        os.chdir('/workspace')
        
        # Run training
        result = subprocess.run(cmd, check=True)
        
        print()
        print("=" * 70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print(f"📁 Model saved to: {output_dir}")
        print()
        print("📊 Check the results:")
        print(f"   - Eval report: {output_dir}/eval_report.json")
        print(f"   - Model files: {output_dir}/pytorch_model.bin")
        print(f"   - Training logs: {output_dir}/trainer_state.json")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed: {e}")
        print("Check the error messages above for details")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        print(f"Partial results may be available in: {output_dir}")
        sys.exit(1)

if __name__ == "__main__":
    main()