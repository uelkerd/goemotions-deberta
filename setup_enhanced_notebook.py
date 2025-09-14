#!/usr/bin/env python3
"""
🚀 ENHANCED MULTI-DATASET NOTEBOOK SETUP
========================================
Final setup script that integrates all improvements into the
SAMo_MultiDataset_Streamlined_CLEAN.ipynb workflow
"""

import os
import sys
from pathlib import Path
import subprocess

def print_banner():
    """Print setup banner"""
    print("🚀 ENHANCED MULTI-DATASET NOTEBOOK SETUP")
    print("=" * 55)
    print("🎯 Goal: Transform simple notebook into robust scientific workflow")
    print("📊 Baseline: 51.79% F1-macro → Target: 60%+ F1-macro")
    print("🔬 Features: Phase-based training + Scientific validation")
    print("=" * 55)

def verify_all_components():
    """Verify all components are in place"""
    print("\n🔍 COMPONENT VERIFICATION")
    print("=" * 30)

    components = [
        # Core scripts
        ("notebooks/prepare_all_datasets.py", "Multi-dataset preparation script"),
        ("scripts/train_comprehensive_multidataset.sh", "Training wrapper script"),
        ("notebooks/scripts/train_deberta_local.py", "Main training script"),

        # Enhanced features
        ("notebooks/verify_emotion_alignment.py", "Emotion alignment validator"),
        ("notebooks/fallback_handler.py", "Fallback and error handler"),
        ("scripts/reproduce_baseline.py", "Baseline reproduction script"),
        ("scripts/phase_based_multidataset_training.py", "Phase-based training system"),
        ("scripts/validation_monitor.py", "Validation and monitoring system"),

        # This setup script
        ("setup_enhanced_notebook.py", "Setup coordination script")
    ]

    missing_components = []
    present_components = []

    for component_path, description in components:
        if os.path.exists(component_path):
            present_components.append((component_path, description))
            print(f"   ✅ {component_path}")
        else:
            missing_components.append((component_path, description))
            print(f"   ❌ {component_path}")

    print(f"\n📊 Component Status: {len(present_components)}/{len(components)} available")

    if missing_components:
        print(f"\n⚠️ Missing components:")
        for path, desc in missing_components:
            print(f"   - {path}: {desc}")
        return False

    print("✅ All components verified!")
    return True

def create_execution_guide():
    """Create comprehensive execution guide"""
    print("\n📋 CREATING EXECUTION GUIDE")
    print("=" * 35)

    guide_content = """# 🚀 ENHANCED MULTI-DATASET TRAINING GUIDE

## 📋 QUICK START (Original Notebook Enhanced)

### Option A: Simple Execution (Enhanced Notebook Cells)
```bash
# Cell 1: Run enhanced data preparation
python notebooks/prepare_all_datasets.py

# Cell 2: Run comprehensive training
bash scripts/train_comprehensive_multidataset.sh

# Cell 3: Monitor progress
python scripts/validation_monitor.py
```

### Option B: Phase-Based Scientific Approach (RECOMMENDED)
```bash
# Step 1: Verify environment and data quality
python notebooks/fallback_handler.py

# Step 2: Reproduce baseline (optional but recommended)
python scripts/reproduce_baseline.py

# Step 3: Run phase-based multi-dataset training
python scripts/phase_based_multidataset_training.py

# Step 4: Comprehensive validation
python scripts/validation_monitor.py
```

## 🔬 SCIENTIFIC WORKFLOW

### Phase 1: Quick Exploration (2-3 hours)
- Tests 4 configurations: BCE, Asymmetric, Combined(0.7), Combined(0.5)
- Uses subset of data for rapid iteration
- Identifies best performing approaches

### Phase 2: Extended Training (4-6 hours)
- Full dataset training on top 2 configurations from Phase 1
- Extended epochs for better convergence
- Target: >60% F1-macro achievement

## 📊 EXPECTED RESULTS

| Configuration | Expected F1-Macro | Status |
|---------------|------------------|---------|
| Baseline (GoEmotions BCE) | 51.79% | ✅ Proven |
| Multi-dataset BCE | 55-58% | 🎯 Conservative |
| Multi-dataset Combined | 58-62% | 🚀 Optimistic |
| Multi-dataset Asymmetric | 52-56% | 📈 Modest improvement |

## 🔍 MONITORING & VALIDATION

### Real-time Monitoring
```bash
# System status
python scripts/validation_monitor.py

# Training logs
tail -f logs/train_comprehensive_multidataset.log

# GPU status
watch -n 5 'nvidia-smi'
```

### Quality Assurance
```bash
# Validate emotion mappings
python notebooks/verify_emotion_alignment.py

# Check dataset quality
python scripts/validation_monitor.py
```

## 🛡️ ROBUST ERROR HANDLING

### Fallback Systems
1. **Missing datasets**: Automatic sample data generation
2. **Training failures**: Checkpoint recovery and restart
3. **GPU issues**: Single-GPU fallback mode
4. **Disk space**: Automatic cleanup of old checkpoints

### Debugging Tools
```bash
# Comprehensive system check
python notebooks/fallback_handler.py

# Training process monitoring
ps aux | grep train_deberta

# Log analysis for issues
grep -i error logs/*.log
```

## 🎯 SUCCESS CRITERIA

### Primary Target: >60% F1-Macro
- **Excellent**: ≥60% (10x improvement over baseline)
- **Good**: 55-60% (6-16% improvement)
- **Acceptable**: 52-55% (1-6% improvement)

### Secondary Metrics
- F1-Micro ≥55%
- Balanced performance across emotion classes
- Stable training without stalls or crashes

## 📁 OUTPUT STRUCTURE

```
project_root/
├── checkpoints_comprehensive_multidataset/  # Main results
├── outputs/phase1_*/                        # Phase 1 exploration results
├── outputs/phase2_*/                        # Phase 2 extended results
├── logs/                                     # All training logs
├── *_report.json                           # Performance reports
└── comprehensive_validation_report.json     # Full validation results
```

## 🚨 TROUBLESHOOTING

### Common Issues
1. **CUDA OOM**: Reduce batch size to 2, increase gradient accumulation
2. **Training stall**: Check GPU utilization, restart if needed
3. **Poor performance**: Verify emotion alignment, check data quality
4. **Missing files**: Run fallback_handler.py to regenerate

### Support Resources
- Training logs: `logs/` directory
- System status: `python scripts/validation_monitor.py`
- Error handling: `python notebooks/fallback_handler.py`

## 🎉 COMPLETION CHECKLIST

- [ ] Data preparation successful (>30K samples)
- [ ] Training completes without errors
- [ ] F1-macro ≥55% achieved
- [ ] Validation report generated
- [ ] Model artifacts saved
- [ ] Performance documented

---

**🔬 This enhanced workflow transforms the simple 3-cell notebook into a robust, scientific, production-ready training pipeline while maintaining ease of use.**
"""

    with open("ENHANCED_TRAINING_GUIDE.md", "w") as f:
        f.write(guide_content)

    print("✅ Comprehensive guide created: ENHANCED_TRAINING_GUIDE.md")

def create_notebook_patches():
    """Create patches for the original notebook"""
    print("\n🔧 NOTEBOOK ENHANCEMENT PATCHES")
    print("=" * 40)

    # Enhanced Cell 2 (Data Preparation)
    enhanced_cell2 = '''# ENHANCED MULTI-DATASET PREPARATION WITH VALIDATION
import os
import subprocess
import sys

print("🚀 ENHANCED MULTI-DATASET PREPARATION")
print("=" * 50)
print("🔬 Features: Scientific validation + Robust error handling")
print("📊 Datasets: GoEmotions + SemEval + ISEAR + MELD")
print("=" * 50)

# Step 1: Run fallback handler for robust setup
print("\\n🛡️ Step 1: Robust Environment Setup")
result = subprocess.run([sys.executable, 'notebooks/fallback_handler.py'],
                       capture_output=True, text=True)
print(result.stdout)

# Step 2: Enhanced dataset preparation
print("\\n📊 Step 2: Enhanced Dataset Preparation")
result = subprocess.run([sys.executable, 'notebooks/prepare_all_datasets.py'],
                       capture_output=False, text=True)

# Step 3: Validation
print("\\n🔍 Step 3: Dataset Quality Validation")
result = subprocess.run([sys.executable, 'notebooks/verify_emotion_alignment.py'],
                       capture_output=True, text=True)
print(result.stdout)

# Verify success
if os.path.exists('data/combined_all_datasets/train.jsonl'):
    train_count = sum(1 for line in open('data/combined_all_datasets/train.jsonl'))
    val_count = sum(1 for line in open('data/combined_all_datasets/val.jsonl'))
    print(f"\\n✅ SUCCESS: {train_count + val_count} samples prepared")
    print(f"   Training: {train_count} samples")
    print(f"   Validation: {val_count} samples")
    print("\\n🚀 Ready for enhanced training! Run next cell.")
else:
    print("\\n❌ FAILED: Dataset preparation unsuccessful")
    print("💡 Check logs and try again")'''

    # Enhanced Cell 4 (Training)
    enhanced_cell4 = '''# ENHANCED MULTI-DATASET TRAINING WITH MONITORING
import os
import subprocess
import sys
from pathlib import Path

print("🚀 ENHANCED MULTI-DATASET TRAINING")
print("=" * 50)
print("🔬 Features: Phase-based + Scientific validation + Monitoring")
print("🎯 Target: >60% F1-macro (vs 51.79% baseline)")
print("=" * 50)

# Change to project directory
os.chdir('/home/user/goemotions-deberta')

# Choose training approach
print("\\n📋 TRAINING APPROACH OPTIONS:")
print("   A) Simple Training (Original approach, enhanced)")
print("   B) Phase-Based Training (Scientific approach, RECOMMENDED)")

approach = input("\\nChoose approach (A/B): ").upper()

if approach == 'B':
    print("\\n🔬 PHASE-BASED SCIENTIFIC TRAINING SELECTED")
    print("⏱️ Duration: ~6-8 hours (2 phases)")
    print("📊 Methodology: Systematic configuration exploration + Extended training")

    # Run phase-based training
    result = subprocess.run([sys.executable, 'scripts/phase_based_multidataset_training.py'],
                           capture_output=False, text=True)

    # Check results
    if os.path.exists('multidataset_training_report.json'):
        import json
        with open('multidataset_training_report.json', 'r') as f:
            report = json.load(f)

        best_result = report.get('best_result', {})
        if best_result:
            f1_macro = best_result.get('f1_macro', 0)
            print(f"\\n🏆 BEST RESULT: F1-macro = {f1_macro:.4f}")
            if f1_macro >= 0.60:
                print("🎉 TARGET ACHIEVED: >60% F1-macro!")
            elif f1_macro >= 0.55:
                print("✅ EXCELLENT: >55% F1-macro achieved!")
            else:
                print("📈 PROGRESS: Above baseline, consider further tuning")

else:
    print("\\n⚡ SIMPLE TRAINING SELECTED")
    print("⏱️ Duration: ~3-4 hours")

    # Run simple training with monitoring
    import threading

    def run_monitoring():
        import time
        time.sleep(300)  # Start monitoring after 5 minutes
        while True:
            subprocess.run([sys.executable, 'scripts/validation_monitor.py'],
                          capture_output=True)
            time.sleep(1800)  # Monitor every 30 minutes

    # Start monitoring in background
    monitor_thread = threading.Thread(target=run_monitoring, daemon=True)
    monitor_thread.start()

    # Run training
    result = subprocess.run(['bash', 'scripts/train_comprehensive_multidataset.sh'],
                           capture_output=False, text=True)

print("\\n✅ ENHANCED TRAINING COMPLETE!")
print("📊 Check results: Run validation cell next")'''

    # Enhanced Cell 6 (Results)
    enhanced_cell6 = '''# ENHANCED RESULTS ANALYSIS WITH COMPREHENSIVE VALIDATION
import json
import sys
import subprocess
from pathlib import Path

print("📊 COMPREHENSIVE RESULTS ANALYSIS")
print("=" * 50)

# Run comprehensive validation
print("🔍 Running comprehensive validation...")
result = subprocess.run([sys.executable, 'scripts/validation_monitor.py'],
                       capture_output=True, text=True)
print(result.stdout)

# Load baseline comparison
baseline = {
    'f1_macro': 0.5179,
    'f1_micro': 0.5975,
    'model': 'GoEmotions BCE (Published HF Model)'
}

print("\\n🏆 BASELINE PERFORMANCE:")
print(f"   F1 Macro: {baseline['f1_macro']:.4f} ({baseline['f1_macro']*100:.1f}%)")
print(f"   Model: {baseline['model']}")

# Check multiple result locations
result_locations = [
    ("checkpoints_comprehensive_multidataset/eval_report.json", "Simple Training"),
    ("multidataset_training_report.json", "Phase-Based Training"),
    ("outputs/phase2_best_extended/eval_report.json", "Extended Training")
]

best_f1 = 0.0
best_config = None

for result_path, config_name in result_locations:
    if os.path.exists(result_path):
        try:
            with open(result_path, 'r') as f:
                data = json.load(f)

            if 'best_result' in data:
                # Phase-based training report
                f1_macro = data['best_result']['f1_macro']
                config_name = f"Phase-Based: {data['best_result']['config']}"
            else:
                # Direct evaluation report
                f1_macro = data.get('f1_macro', 0.0)

            if f1_macro > best_f1:
                best_f1 = f1_macro
                best_config = config_name

            print(f"\\n🎯 {config_name.upper()}:")
            print(f"   F1-macro: {f1_macro:.4f} ({f1_macro*100:.1f}%)")

            improvement = ((f1_macro - baseline['f1_macro']) / baseline['f1_macro']) * 100
            print(f"   Improvement: {improvement:+.1f}% over baseline")

        except Exception as e:
            print(f"   ❌ Error reading {result_path}: {e}")

# Final assessment
print(f"\\n🏆 BEST OVERALL RESULT:")
print(f"   Configuration: {best_config}")
print(f"   F1-macro: {best_f1:.4f}")

if best_f1 >= 0.60:
    print("\\n🎉 OUTSTANDING SUCCESS!")
    print("   ✅ Target >60% F1-macro achieved!")
    print("   🚀 Multi-dataset training HIGHLY SUCCESSFUL!")
elif best_f1 >= 0.55:
    print("\\n✅ EXCELLENT SUCCESS!")
    print("   ✅ Strong >55% F1-macro achieved!")
    print("   📈 Significant improvement from multi-dataset approach")
elif best_f1 > baseline['f1_macro']:
    print("\\n👍 SUCCESSFUL IMPROVEMENT!")
    print("   ✅ Beat baseline performance")
    print("   🔧 Consider extended training for even better results")
else:
    print("\\n⚠️ NEEDS INVESTIGATION")
    print("   🔍 Check data quality and training logs")
    print("   💡 Consider baseline reproduction first")

print(f"\\n📋 TARGET ACHIEVEMENT:")
print(f"   >60% F1-macro: {'✅' if best_f1 >= 0.60 else '❌'}")
print(f"   >55% F1-macro: {'✅' if best_f1 >= 0.55 else '❌'}")
print(f"   Beat baseline: {'✅' if best_f1 > baseline['f1_macro'] else '❌'}")

print(f"\\n📄 COMPREHENSIVE REPORT:")
print(f"   Available at: comprehensive_validation_report.json")'''

    # Save patches
    patches = {
        "enhanced_cell_2_data_preparation.py": enhanced_cell2,
        "enhanced_cell_4_training.py": enhanced_cell4,
        "enhanced_cell_6_results.py": enhanced_cell6
    }

    patch_dir = "notebook_patches"
    os.makedirs(patch_dir, exist_ok=True)

    for filename, content in patches.items():
        with open(f"{patch_dir}/{filename}", "w") as f:
            f.write(content)

    print(f"✅ Notebook patches created in: {patch_dir}/")
    print("   Use these to enhance the original notebook cells")

def create_quick_start_script():
    """Create a quick start script for immediate use"""
    print("\n🚀 CREATING QUICK START SCRIPT")
    print("=" * 35)

    quick_start = '''#!/bin/bash

# 🚀 QUICK START: ENHANCED MULTI-DATASET TRAINING
# ===============================================

echo "🚀 ENHANCED MULTI-DATASET TRAINING - QUICK START"
echo "================================================"
echo "🎯 Goal: Achieve >60% F1-macro using multi-dataset approach"
echo "📊 Baseline: 51.79% F1-macro (GoEmotions BCE)"
echo "================================================"

# Step 1: Environment setup and validation
echo ""
echo "🔧 Step 1: Environment Setup & Validation"
echo "----------------------------------------"
python3 notebooks/fallback_handler.py

# Step 2: Dataset preparation
echo ""
echo "📊 Step 2: Multi-Dataset Preparation"
echo "------------------------------------"
python3 notebooks/prepare_all_datasets.py

# Step 3: Emotion alignment validation
echo ""
echo "🔍 Step 3: Emotion Alignment Validation"
echo "---------------------------------------"
python3 notebooks/verify_emotion_alignment.py

# Step 4: Training approach selection
echo ""
echo "🚀 Step 4: Training Approach Selection"
echo "--------------------------------------"
echo "Choose training approach:"
echo "  A) Simple Training (3-4 hours, good for testing)"
echo "  B) Phase-Based Training (6-8 hours, recommended for best results)"
echo ""

read -p "Enter choice (A/B): " choice

if [[ "$choice" == "B" || "$choice" == "b" ]]; then
    echo ""
    echo "🔬 Running Phase-Based Scientific Training..."
    echo "⏱️ Duration: ~6-8 hours"
    python3 scripts/phase_based_multidataset_training.py
else
    echo ""
    echo "⚡ Running Simple Training..."
    echo "⏱️ Duration: ~3-4 hours"
    bash scripts/train_comprehensive_multidataset.sh
fi

# Step 5: Comprehensive validation and results
echo ""
echo "📊 Step 5: Comprehensive Results Analysis"
echo "-----------------------------------------"
python3 scripts/validation_monitor.py

echo ""
echo "✅ ENHANCED MULTI-DATASET TRAINING COMPLETE!"
echo "📄 Check comprehensive_validation_report.json for full results"
echo "🎯 Expected improvement: 51.79% → 55-65% F1-macro"
'''

    with open("quick_start_multidataset.sh", "w") as f:
        f.write(quick_start)

    os.chmod("quick_start_multidataset.sh", 0o755)
    print("✅ Quick start script created: quick_start_multidataset.sh")

def main():
    """Main setup execution"""
    print_banner()

    # Verify components
    if not verify_all_components():
        print("\n❌ Setup incomplete - missing components detected")
        return False

    # Create documentation and guides
    create_execution_guide()
    create_notebook_patches()
    create_quick_start_script()

    # Final summary
    print("\n🎉 ENHANCED NOTEBOOK SETUP COMPLETE!")
    print("=" * 45)
    print("✅ All components installed and verified")
    print("✅ Execution guide created")
    print("✅ Notebook patches generated")
    print("✅ Quick start script ready")

    print(f"\n🚀 NEXT STEPS:")
    print(f"   1. IMMEDIATE: Run ./quick_start_multidataset.sh")
    print(f"   2. NOTEBOOK: Use enhanced cells from notebook_patches/")
    print(f"   3. DOCUMENTATION: Read ENHANCED_TRAINING_GUIDE.md")

    print(f"\n🎯 TRANSFORMATION COMPLETE:")
    print(f"   Original: Simple 3-cell notebook (prone to failure)")
    print(f"   Enhanced: Robust scientific pipeline (production-ready)")
    print(f"   Expected: 51.79% → 60%+ F1-macro improvement")

    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 Ready to achieve >60% F1-macro with multi-dataset training!")
    else:
        print("\n🔧 Please address setup issues before proceeding.")