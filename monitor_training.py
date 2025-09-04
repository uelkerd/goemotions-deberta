#!/usr/bin/env python3
"""
Monitor training progress and compare with BCE baseline disaster
"""

import os
import json
import glob
import time
from datetime import datetime

BCE_BASELINE = {
    'f1_macro': 0.054,
    'f1_micro': 0.2813,
    'f1_weighted': 0.1876
}

def check_training_status():
    """Check status of all training runs"""
    print("\n" + "=" * 70)
    print(f"📊 TRAINING STATUS CHECK - {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 70)
    
    # Check for output directories
    output_dirs = glob.glob("outputs/emergency_*")
    
    if not output_dirs:
        print("⏳ No emergency training outputs found yet...")
        print("   Training may still be initializing.")
        return
    
    results = []
    for output_dir in output_dirs:
        config_name = output_dir.replace("outputs/emergency_", "")
        
        # Check for eval_report.json
        eval_report = os.path.join(output_dir, "eval_report.json")
        
        # Check for trainer_state.json for progress
        trainer_state = os.path.join(output_dir, "trainer_state.json")
        
        if os.path.exists(eval_report):
            with open(eval_report, 'r') as f:
                data = json.load(f)
            
            f1_macro = data.get('f1_macro', 0.0)
            f1_micro = data.get('f1_micro', 0.0)
            improvement = (f1_macro / BCE_BASELINE['f1_macro'] - 1) * 100
            
            results.append({
                'name': config_name,
                'status': 'COMPLETE',
                'f1_macro': f1_macro,
                'f1_micro': f1_micro,
                'improvement': improvement
            })
            
        elif os.path.exists(trainer_state):
            with open(trainer_state, 'r') as f:
                state = json.load(f)
            
            current_step = state.get('global_step', 0)
            total_steps = state.get('max_steps', 0)
            if total_steps > 0:
                progress = (current_step / total_steps) * 100
            else:
                progress = 0
            
            results.append({
                'name': config_name,
                'status': 'RUNNING',
                'progress': f"{progress:.1f}%",
                'step': f"{current_step}/{total_steps}"
            })
        else:
            # Check if directory has any files (training started)
            files = os.listdir(output_dir) if os.path.exists(output_dir) else []
            if files:
                results.append({
                    'name': config_name,
                    'status': 'INITIALIZING',
                    'files': len(files)
                })
    
    # Display results
    print("\n🔥 BCE BASELINE (CATASTROPHIC FAILURE):")
    print(f"   F1 Macro: {BCE_BASELINE['f1_macro']:.4f} (5.4%)")
    print(f"   F1 Micro: {BCE_BASELINE['f1_micro']:.4f}")
    print(f"   Status: Only 3/28 classes learned!")
    
    print("\n🚀 IMPROVED CONFIGURATIONS:")
    
    if not results:
        print("   ⏳ Waiting for training to start...")
    else:
        for result in sorted(results, key=lambda x: x['name']):
            print(f"\n   📦 {result['name'].upper()}:")
            
            if result['status'] == 'COMPLETE':
                print(f"      Status: ✅ COMPLETE")
                print(f"      F1 Macro: {result['f1_macro']:.4f}")
                print(f"      F1 Micro: {result['f1_micro']:.4f}")
                
                if result['improvement'] > 0:
                    print(f"      🎯 IMPROVEMENT: {result['improvement']:.1f}% better than BCE!")
                    if result['improvement'] > 500:
                        print(f"      🏆 MASSIVE SUCCESS! {result['f1_macro']/BCE_BASELINE['f1_macro']:.1f}x better!")
                        
            elif result['status'] == 'RUNNING':
                print(f"      Status: 🔄 RUNNING")
                print(f"      Progress: {result['progress']}")
                print(f"      Step: {result['step']}")
                
            elif result['status'] == 'INITIALIZING':
                print(f"      Status: ⏳ INITIALIZING")
                print(f"      Files created: {result['files']}")
    
    # Check for logs
    print("\n📝 RECENT LOGS:")
    log_files = glob.glob("outputs/emergency_*/scientific_log_*.json")
    if log_files:
        latest_log = max(log_files, key=os.path.getctime)
        print(f"   Latest log: {latest_log}")
        with open(latest_log, 'r') as f:
            log_data = json.load(f)
        if 'training_history' in log_data and log_data['training_history']:
            latest_entry = log_data['training_history'][-1]
            if 'loss' in latest_entry:
                print(f"   Latest loss: {latest_entry['loss']:.4f}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    print("🔍 MONITORING EMERGENCY TRAINING")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            check_training_status()
            time.sleep(30)  # Check every 30 seconds
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")