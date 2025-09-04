#!/usr/bin/env python3
"""
Simple monitoring for the FINAL training
"""

import os
import json
import time
from datetime import datetime

def check_progress():
    """Check training progress"""
    output_dir = "./outputs/FINAL_MODEL"
    
    print(f"\n{'='*60}")
    print(f"📊 TRAINING MONITOR - {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    
    # Check if training started
    if not os.path.exists(output_dir):
        print("⏳ Training not started yet...")
        return
    
    # Check trainer state
    trainer_state = os.path.join(output_dir, "trainer_state.json")
    if os.path.exists(trainer_state):
        with open(trainer_state, 'r') as f:
            state = json.load(f)
        
        current_step = state.get('global_step', 0)
        total_steps = state.get('max_steps', 0)
        best_metric = state.get('best_metric', 0)
        
        if total_steps > 0:
            progress = (current_step / total_steps) * 100
            print(f"🔄 Progress: {progress:.1f}% ({current_step}/{total_steps} steps)")
        
        if best_metric > 0:
            print(f"📈 Best F1 Macro so far: {best_metric:.4f}")
            
            # Compare with BCE baseline
            bce_baseline = 0.054
            improvement = (best_metric / bce_baseline)
            print(f"🎯 That's {improvement:.1f}x better than BCE baseline!")
    
    # Check for checkpoint folders
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        print(f"💾 Checkpoints saved: {len(checkpoints)}")
        latest = max(checkpoints, key=lambda x: int(x.split('-')[1]))
        print(f"   Latest: {latest}")
    
    # Check for eval results
    eval_report = os.path.join(output_dir, "eval_report.json")
    if os.path.exists(eval_report):
        print("\n✅ TRAINING COMPLETE!")
        with open(eval_report, 'r') as f:
            results = json.load(f)
        
        print(f"📊 Final Results:")
        print(f"   F1 Macro: {results.get('f1_macro', 0):.4f}")
        print(f"   F1 Micro: {results.get('f1_micro', 0):.4f}")
        print(f"   F1 Weighted: {results.get('f1_weighted', 0):.4f}")
        
        # Show improvement
        final_f1 = results.get('f1_macro', 0)
        if final_f1 > 0:
            improvement = ((final_f1 - 0.054) / 0.054) * 100
            print(f"\n🏆 FINAL IMPROVEMENT: {improvement:.1f}% better than BCE!")
            print(f"   ({final_f1:.4f} vs 0.054)")

if __name__ == "__main__":
    print("🔍 Monitoring FINAL MODEL training...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            check_progress()
            
            # Check if training is complete
            if os.path.exists("./outputs/FINAL_MODEL/eval_report.json"):
                print("\n🎉 Training complete! Exiting monitor.")
                break
                
            time.sleep(30)  # Check every 30 seconds
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")
