#!/usr/bin/env python3
"""
🚀 PHASE-BASED MULTI-DATASET TRAINING
=====================================
Implements the proven phase-based workflow from GoEmotions_DeBERTa_ALL_PHASES_FIXED.ipynb
for multi-dataset training with scientific rigor and robustness
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime

class PhaseBasedTrainer:
    """Phase-based training implementation"""

    def __init__(self):
        self.phases_completed = []
        self.results = {}
        self.baseline_f1 = 0.5179  # Your published model baseline

    def setup_phase_environment(self):
        """Setup environment for phase-based training"""
        print("🔧 PHASE ENVIRONMENT SETUP")
        print("=" * 35)

        # Create phase directories
        phase_dirs = [
            "outputs/phase1_multidataset_bce",
            "outputs/phase1_multidataset_asymmetric",
            "outputs/phase1_multidataset_combined_07",
            "outputs/phase1_multidataset_combined_05",
            "outputs/phase2_best_extended",
            "logs/phases"
        ]

        for dir_path in phase_dirs:
            os.makedirs(dir_path, exist_ok=True)

        print("✅ Phase directories created")

        # Check prerequisites
        prerequisites = [
            "data/combined_all_datasets/train.jsonl",
            "data/combined_all_datasets/val.jsonl",
            "notebooks/scripts/train_deberta_local.py"
        ]

        missing = [p for p in prerequisites if not os.path.exists(p)]
        if missing:
            print(f"❌ Missing prerequisites: {missing}")
            return False

        print("✅ All prerequisites available")
        return True

    def run_phase1_exploration(self):
        """Phase 1: Quick exploration with multiple configurations"""
        print("\\n🚀 PHASE 1: MULTI-DATASET CONFIGURATION EXPLORATION")
        print("=" * 65)
        print("🎯 Goal: Test 4 configurations on multi-dataset")
        print("⏱️ Duration: ~3-4 hours total")
        print("📊 Configs: BCE, Asymmetric, Combined(0.7), Combined(0.5)")

        configs = [
            {
                'name': 'bce',
                'output_dir': 'outputs/phase1_multidataset_bce',
                'use_asymmetric_loss': False,
                'use_combined_loss': False,
                'loss_combination_ratio': None
            },
            {
                'name': 'asymmetric',
                'output_dir': 'outputs/phase1_multidataset_asymmetric',
                'use_asymmetric_loss': True,
                'use_combined_loss': False,
                'loss_combination_ratio': None
            },
            {
                'name': 'combined_07',
                'output_dir': 'outputs/phase1_multidataset_combined_07',
                'use_asymmetric_loss': False,
                'use_combined_loss': True,
                'loss_combination_ratio': 0.7
            },
            {
                'name': 'combined_05',
                'output_dir': 'outputs/phase1_multidataset_combined_05',
                'use_asymmetric_loss': False,
                'use_combined_loss': True,
                'loss_combination_ratio': 0.5
            }
        ]

        phase1_results = {}

        for config in configs:
            print(f"\\n🔄 Training {config['name'].upper()} configuration...")

            success, f1_score = self.run_single_config(
                config,
                epochs=2,  # Quick exploration
                max_train_samples=15000,  # Subset for speed
                max_eval_samples=2000
            )

            if success:
                phase1_results[config['name']] = {
                    'f1_macro': f1_score,
                    'success': f1_score > self.baseline_f1,
                    'improvement': ((f1_score - self.baseline_f1) / self.baseline_f1) * 100
                }
                print(f"✅ {config['name']}: F1-macro = {f1_score:.4f}")
            else:
                phase1_results[config['name']] = {
                    'f1_macro': 0.0,
                    'success': False,
                    'improvement': -100
                }
                print(f"❌ {config['name']}: Failed")

        # Save phase 1 results
        with open("logs/phases/phase1_results.json", "w") as f:
            json.dump(phase1_results, f, indent=2)

        self.results['phase1'] = phase1_results
        self.phases_completed.append('phase1')

        # Analyze phase 1
        successful_configs = [name for name, result in phase1_results.items()
                            if result['success']]

        print(f"\\n📊 PHASE 1 RESULTS:")
        print("=" * 25)
        for name, result in phase1_results.items():
            status = "SUCCESS" if result['success'] else "BELOW BASELINE"
            print(f"   {name}: {result['f1_macro']:.4f} ({status})")

        print(f"\\n🏆 Successful configs: {len(successful_configs)}/{len(configs)}")

        return successful_configs

    def run_phase2_extended_training(self, best_configs):
        """Phase 2: Extended training on best configurations"""
        print("\\n🚀 PHASE 2: EXTENDED TRAINING ON BEST CONFIGS")
        print("=" * 55)
        print(f"🎯 Goal: Train top configs with full dataset and extended epochs")
        print(f"📊 Configs: {best_configs}")

        if not best_configs:
            print("⚠️ No successful configs from Phase 1 - skipping Phase 2")
            return []

        # Select top 2 configs by F1 score
        sorted_configs = sorted(
            [(name, self.results['phase1'][name]) for name in best_configs],
            key=lambda x: x[1]['f1_macro'],
            reverse=True
        )[:2]

        phase2_results = {}

        for name, phase1_result in sorted_configs:
            print(f"\\n🔄 Extended training for {name.upper()}...")
            print(f"   Phase 1 F1-macro: {phase1_result['f1_macro']:.4f}")

            # Get original config
            config = next(c for c in [
                {'name': 'bce', 'use_asymmetric_loss': False, 'use_combined_loss': False, 'loss_combination_ratio': None},
                {'name': 'asymmetric', 'use_asymmetric_loss': True, 'use_combined_loss': False, 'loss_combination_ratio': None},
                {'name': 'combined_07', 'use_asymmetric_loss': False, 'use_combined_loss': True, 'loss_combination_ratio': 0.7},
                {'name': 'combined_05', 'use_asymmetric_loss': False, 'use_combined_loss': True, 'loss_combination_ratio': 0.5}
            ] if c['name'] == name)

            config['output_dir'] = f"outputs/phase2_{name}_extended"

            success, f1_score = self.run_single_config(
                config,
                epochs=3,  # Extended training
                max_train_samples=None,  # Full dataset
                max_eval_samples=None
            )

            if success:
                phase2_results[name] = {
                    'f1_macro': f1_score,
                    'improvement_over_phase1': f1_score - phase1_result['f1_macro'],
                    'improvement_over_baseline': ((f1_score - self.baseline_f1) / self.baseline_f1) * 100,
                    'target_60_achieved': f1_score >= 0.60
                }
                print(f"✅ {name}: F1-macro = {f1_score:.4f}")
            else:
                phase2_results[name] = {
                    'f1_macro': 0.0,
                    'improvement_over_phase1': 0.0,
                    'improvement_over_baseline': -100,
                    'target_60_achieved': False
                }
                print(f"❌ {name}: Failed")

        # Save phase 2 results
        with open("logs/phases/phase2_results.json", "w") as f:
            json.dump(phase2_results, f, indent=2)

        self.results['phase2'] = phase2_results
        self.phases_completed.append('phase2')

        # Analyze phase 2
        print(f"\\n📊 PHASE 2 RESULTS:")
        print("=" * 25)
        for name, result in phase2_results.items():
            target_status = "🎯 TARGET ACHIEVED" if result['target_60_achieved'] else "📈 PROGRESS"
            print(f"   {name}: {result['f1_macro']:.4f} ({target_status})")
            print(f"      Improvement over baseline: {result['improvement_over_baseline']:+.1f}%")

        return list(phase2_results.keys())

    def run_single_config(self, config, epochs=3, max_train_samples=None, max_eval_samples=None):
        """Run a single training configuration"""

        # Build training command
        cmd = [
            'python3', 'notebooks/scripts/train_deberta_local.py',
            '--output_dir', config['output_dir'],
            '--model_type', 'deberta-v3-large',
            '--per_device_train_batch_size', '4',
            '--per_device_eval_batch_size', '8',
            '--gradient_accumulation_steps', '4',
            '--num_train_epochs', str(epochs),
            '--learning_rate', '3e-5',
            '--lr_scheduler_type', 'cosine',
            '--warmup_ratio', '0.1',
            '--weight_decay', '0.01',
            '--fp16',
            '--max_length', '256',
            '--augment_prob', '0.0',
            '--label_smoothing', '0.0',
            '--early_stopping_patience', '3'
        ]

        # Add configuration-specific parameters
        if config['use_asymmetric_loss']:
            cmd.append('--use_asymmetric_loss')

        if config['use_combined_loss'] and config['loss_combination_ratio']:
            cmd.extend(['--use_combined_loss', '--loss_combination_ratio', str(config['loss_combination_ratio'])])

        if max_train_samples:
            cmd.extend(['--max_train_samples', str(max_train_samples)])

        if max_eval_samples:
            cmd.extend(['--max_eval_samples', str(max_eval_samples)])

        # Set environment
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'  # Single GPU for stability

        # Create log file
        log_file = f"logs/phases/{config['name']}_training.log"

        print(f"   📝 Logging to: {log_file}")
        print(f"   ⏱️ Expected duration: ~{epochs * 45} minutes")

        # Run training
        try:
            with open(log_file, 'w') as f:
                f.write(f"Training started: {datetime.now()}\\n")
                f.write(f"Command: {' '.join(cmd)}\\n\\n")

            result = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True)

            # Log output
            with open(log_file, 'a') as f:
                f.write(result.stdout)
                f.write(f"\\nTraining completed: {datetime.now()}\\n")
                f.write(f"Exit code: {result.returncode}\\n")

            if result.returncode == 0:
                # Read results
                results_file = f"{config['output_dir']}/eval_report.json"
                if os.path.exists(results_file):
                    with open(results_file, 'r') as f:
                        eval_results = json.load(f)
                    return True, eval_results.get('f1_macro', 0.0)
                else:
                    return False, 0.0
            else:
                return False, 0.0

        except Exception as e:
            with open(log_file, 'a') as f:
                f.write(f"\\nError: {e}\\n")
            return False, 0.0

    def generate_final_report(self):
        """Generate comprehensive final report"""
        print("\\n📊 FINAL MULTI-DATASET TRAINING REPORT")
        print("=" * 50)

        report = {
            "training_summary": {
                "timestamp": datetime.now().isoformat(),
                "baseline_f1_macro": self.baseline_f1,
                "target_f1_macro": 0.60,
                "phases_completed": self.phases_completed,
                "total_configs_tested": len(self.results.get('phase1', {}))
            },
            "results": self.results
        }

        # Find best overall result
        all_f1_scores = []

        if 'phase1' in self.results:
            all_f1_scores.extend([(f"phase1_{name}", result['f1_macro'])
                                for name, result in self.results['phase1'].items()])

        if 'phase2' in self.results:
            all_f1_scores.extend([(f"phase2_{name}", result['f1_macro'])
                                for name, result in self.results['phase2'].items()])

        if all_f1_scores:
            best_config, best_f1 = max(all_f1_scores, key=lambda x: x[1])

            report["best_result"] = {
                "config": best_config,
                "f1_macro": best_f1,
                "improvement_over_baseline": ((best_f1 - self.baseline_f1) / self.baseline_f1) * 100,
                "target_60_achieved": best_f1 >= 0.60
            }

            print(f"🏆 BEST RESULT:")
            print(f"   Configuration: {best_config}")
            print(f"   F1-macro: {best_f1:.4f}")
            print(f"   Improvement: {report['best_result']['improvement_over_baseline']:+.1f}%")
            print(f"   Target achieved: {'✅' if best_f1 >= 0.60 else '❌'}")
        else:
            print("⚠️ No valid results to report")

        # Save report
        with open("multidataset_training_report.json", "w") as f:
            json.dump(report, f, indent=2)

        print(f"\\n📄 Full report saved: multidataset_training_report.json")

        return report

def main():
    """Main phase-based training execution"""
    print("🚀 PHASE-BASED MULTI-DATASET TRAINING")
    print("=" * 50)
    print("🎯 Goal: Achieve >60% F1-macro using multi-dataset approach")
    print("🔬 Method: Phase-based systematic training with proven robustness")
    print("📊 Baseline: 51.79% F1-macro (GoEmotions BCE)")
    print("=" * 50)

    trainer = PhaseBasedTrainer()

    # Setup environment
    if not trainer.setup_phase_environment():
        print("❌ Environment setup failed")
        return False

    # Phase 1: Configuration exploration
    successful_configs = trainer.run_phase1_exploration()

    # Phase 2: Extended training (if Phase 1 had successes)
    if successful_configs:
        trainer.run_phase2_extended_training(successful_configs)
    else:
        print("\\n⚠️ PHASE 2 SKIPPED: No successful configs from Phase 1")

    # Generate final report
    report = trainer.generate_final_report()

    # Final recommendations
    print("\\n🎯 RECOMMENDATIONS:")
    if report.get('best_result', {}).get('target_60_achieved', False):
        print("   ✅ SUCCESS: Target achieved! Deploy the best model")
    elif report.get('best_result', {}).get('f1_macro', 0) > trainer.baseline_f1:
        print("   📈 IMPROVEMENT: Beat baseline, consider further tuning")
    else:
        print("   🔧 DEBUGGING: Review logs and adjust approach")

    print("\\n✅ PHASE-BASED TRAINING COMPLETE!")
    return True

if __name__ == "__main__":
    main()