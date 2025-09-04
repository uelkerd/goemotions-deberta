#!/usr/bin/env python3
"""
RIGOROUS 5-CONFIG COMPARISON - FIXED VERSION
This actually tests loss functions properly with correct baseline settings
"""

import subprocess
import sys
import os
import json
import time
from datetime import datetime

# THE CORRECT BASELINE SETTINGS (that actually work!)
BASE_CONFIG = {
    'model_type': 'deberta-v3-large',
    'per_device_train_batch_size': '4',
    'per_device_eval_batch_size': '8', 
    'gradient_accumulation_steps': '4',
    'num_train_epochs': '2',  # 2 epochs for screening, 3 for final
    'learning_rate': '3e-5',  # CORRECT learning rate for DeBERTa-v3
    'warmup_ratio': '0.15',   # Proper warmup
    'weight_decay': '0.01',
    'fp16': '',
    'max_length': '256',
    'max_train_samples': '15000',  # Enough data to learn all classes
    'max_eval_samples': '2000',
    'evaluation_strategy': 'steps',
    'eval_steps': '250',
    'save_strategy': 'steps',
    'save_steps': '250',
    'logging_steps': '50',
    'metric_for_best_model': 'f1_macro',
    'greater_is_better': '',
    'load_best_model_at_end': '',
    'save_total_limit': '1'
}

# CONFIGURATIONS TO COMPARE (all with PROPER baseline)
CONFIGS = [
    {
        'name': 'bce_with_class_weights',
        'description': 'BCE with computed class weights',
        'extra_args': {
            'use_class_weights': ''  # Enable class weighting
        }
    },
    {
        'name': 'focal_loss',
        'description': 'Focal Loss (gamma=2.0, alpha=0.25)',
        'extra_args': {
            'use_focal_loss': '',
            'focal_gamma': '2.0',
            'focal_alpha': '0.25'
        }
    },
    {
        'name': 'asymmetric_loss_fixed',
        'description': 'Asymmetric Loss with proper scaling',
        'extra_args': {
            'use_asymmetric_loss': '',
            'asl_gamma_neg': '3.0',
            'asl_gamma_pos': '1.0',
            'asl_clip': '0.05'
        }
    },
    {
        'name': 'combined_focal_weighted',
        'description': 'Focal Loss + Class Weights',
        'extra_args': {
            'use_focal_loss': '',
            'use_class_weights': '',
            'focal_gamma': '2.0'
        }
    },
    {
        'name': 'label_smoothing',
        'description': 'BCE with label smoothing (0.1)',
        'extra_args': {
            'label_smoothing': '0.1'
        }
    }
]

def run_config(config, base_args):
    """Run a single configuration"""
    timestamp = datetime.now().strftime("%H%M%S")
    output_dir = f"./outputs/rigorous_{config['name']}_{timestamp}"
    
    print(f"\n{'='*70}")
    print(f"🔬 TESTING: {config['name'].upper()}")
    print(f"   {config['description']}")
    print(f"   Output: {output_dir}")
    print(f"   Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}\n")
    
    # Build command
    cmd = [sys.executable, 'notebooks/scripts/train_deberta_local.py']
    cmd.extend(['--output_dir', output_dir])
    
    # Add base configuration
    for key, value in base_args.items():
        cmd.append(f'--{key}')
        if value:  # Only add value if not empty
            cmd.append(value)
    
    # Add config-specific arguments
    for key, value in config.get('extra_args', {}).items():
        cmd.append(f'--{key}')
        if value:
            cmd.append(value)
    
    # Run training
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd='/workspace',
            timeout=3600  # 1 hour timeout
        )
        
        duration = time.time() - start_time
        
        # Extract results
        eval_report = f"{output_dir}/eval_report.json"
        if os.path.exists(eval_report):
            with open(eval_report, 'r') as f:
                metrics = json.load(f)
            
            return {
                'name': config['name'],
                'success': True,
                'duration': duration,
                'f1_macro': metrics.get('f1_macro', 0.0),
                'f1_micro': metrics.get('f1_micro', 0.0),
                'f1_weighted': metrics.get('f1_weighted', 0.0),
                'precision_macro': metrics.get('precision_macro', 0.0),
                'recall_macro': metrics.get('recall_macro', 0.0)
            }
        else:
            return {
                'name': config['name'],
                'success': False,
                'duration': duration,
                'error': 'No eval report found'
            }
            
    except subprocess.TimeoutExpired:
        return {
            'name': config['name'],
            'success': False,
            'duration': 3600,
            'error': 'Timeout'
        }
    except Exception as e:
        return {
            'name': config['name'],
            'success': False,
            'duration': time.time() - start_time,
            'error': str(e)
        }

def main():
    print("🔬 RIGOROUS LOSS FUNCTION COMPARISON - FIXED VERSION")
    print("=" * 70)
    print("\n📊 Key Improvements Over Original:")
    print("  • 15,000 training samples (3x more)")
    print("  • Learning rate: 3e-5 (optimal for DeBERTa-v3)")
    print("  • 2 epochs screening (not 1)")
    print("  • Class weighting enabled where appropriate")
    print("  • Proper loss function parameters")
    print("\n⏰ Expected: ~30 min per config, ~2.5 hours total")
    print("📈 Expected F1 Macro: 40-60% (not 5%!)")
    print("=" * 70)
    
    results = []
    
    # Run all configurations
    for i, config in enumerate(CONFIGS, 1):
        print(f"\n\n📊 CONFIGURATION {i}/{len(CONFIGS)}")
        result = run_config(config, BASE_CONFIG)
        results.append(result)
        
        if result['success']:
            print(f"✅ SUCCESS: F1 Macro = {result['f1_macro']:.4f}")
        else:
            print(f"❌ FAILED: {result.get('error', 'Unknown error')}")
    
    # Summarize results
    print("\n\n" + "=" * 70)
    print("📊 FINAL COMPARISON RESULTS")
    print("=" * 70)
    
    # Sort by F1 macro
    successful = [r for r in results if r['success']]
    successful.sort(key=lambda x: x['f1_macro'], reverse=True)
    
    print("\n🏆 RANKINGS:")
    for i, result in enumerate(successful, 1):
        print(f"\n{i}. {result['name'].upper()}")
        print(f"   F1 Macro: {result['f1_macro']:.4f}")
        print(f"   F1 Micro: {result['f1_micro']:.4f}")
        print(f"   Precision: {result['precision_macro']:.4f}")
        print(f"   Recall: {result['recall_macro']:.4f}")
        print(f"   Duration: {result['duration']/60:.1f} min")
    
    if successful:
        winner = successful[0]
        print(f"\n\n🎯 WINNER: {winner['name'].upper()}")
        print(f"   Best F1 Macro: {winner['f1_macro']:.4f}")
        print(f"   Improvement over BCE baseline: {(winner['f1_macro']/0.054 - 1)*100:.1f}%")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"outputs/comparison_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'configs': CONFIGS,
            'results': results,
            'winner': successful[0] if successful else None,
            'timestamp': timestamp
        }, f, indent=2)
    
    print(f"\n📁 Full results saved to: {results_file}")

if __name__ == "__main__":
    main()