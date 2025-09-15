#!/usr/bin/env python3
"""
🔬 SCIENTIFIC LOSS FUNCTION COMPARISON
=====================================
Systematic testing of BCE, Asymmetric, and Combined Loss functions
Based on ALL_PHASES_FIXED methodology for rigorous validation

GOAL: Identify optimal loss function for multi-dataset emotion classification
"""

import os
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LossFunctionTester:
    def __init__(self, base_output_dir="./outputs/loss_comparison"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        self.results = {}
        self.baseline_f1 = 0.5179  # Known baseline from previous experiments

        # Test configurations
        self.test_configs = [
            {
                'name': 'BCE_Pure',
                'description': 'Pure BCE Loss (baseline reproduction)',
                'args': [],
                'expected_f1': 0.52,  # Should reproduce baseline
            },
            {
                'name': 'Asymmetric_Tuned',
                'description': 'Asymmetric Loss with optimal parameters',
                'args': ['--use_asymmetric_loss'],
                'expected_f1': 0.55,  # Should improve on baseline
            },
            {
                'name': 'Combined_Conservative',
                'description': 'Combined Loss with conservative ratio',
                'args': ['--use_combined_loss', '--loss_combination_ratio', '0.3'],
                'expected_f1': 0.50,  # Safer approach
            },
            {
                'name': 'Combined_Balanced',
                'description': 'Combined Loss with balanced ratio',
                'args': ['--use_combined_loss', '--loss_combination_ratio', '0.5'],
                'expected_f1': 0.52,  # Balanced approach
            },
            {
                'name': 'Combined_Aggressive',
                'description': 'Combined Loss with aggressive ratio',
                'args': ['--use_combined_loss', '--loss_combination_ratio', '0.7'],
                'expected_f1': 0.48,  # May be unstable
            }
        ]

    def run_training_config(self, config):
        """Run training for a specific loss function configuration"""
        output_dir = self.base_output_dir / config['name']
        output_dir.mkdir(exist_ok=True)

        logger.info(f"🚀 Starting {config['name']}: {config['description']}")

        # Base training command with optimized parameters
        base_cmd = [
            'python3', 'notebooks/scripts/train_deberta_local.py',
            '--output_dir', str(output_dir),
            '--model_type', 'deberta-v3-large',
            '--per_device_train_batch_size', '4',
            '--per_device_eval_batch_size', '8',
            '--gradient_accumulation_steps', '2',  # Reduced for faster testing
            '--num_train_epochs', '2',  # Quick test first
            '--learning_rate', '3e-5',
            '--lr_scheduler_type', 'cosine',
            '--warmup_ratio', '0.1',
            '--weight_decay', '0.01',
            '--fp16',
            '--max_length', '256',
            '--threshold', '0.2',  # Critical for F1@0.2 evaluation
            '--max_train_samples', '15000',  # Subset for faster testing
            '--max_eval_samples', '3000',
            '--augment_prob', '0.0',
            '--freeze_layers', '0',
            '--early_stopping_patience', '3'
        ]

        # Add loss-specific arguments
        cmd = base_cmd + config['args']

        # Set environment for single GPU
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'  # Use single GPU for stability

        logger.info(f"Command: {' '.join(cmd)}")

        # Run training with timeout protection
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                env=env,
                timeout=3600,  # 1 hour timeout
                capture_output=True,
                text=True
            )

            elapsed_time = time.time() - start_time

            if result.returncode == 0:
                logger.info(f"✅ {config['name']} completed successfully in {elapsed_time:.1f}s")
                return self.extract_results(output_dir, config, elapsed_time)
            else:
                logger.error(f"❌ {config['name']} failed with return code: {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            logger.error(f"⏰ {config['name']} timed out after 1 hour")
            return None
        except Exception as e:
            logger.error(f"💥 {config['name']} crashed: {str(e)}")
            return None

    def extract_results(self, output_dir, config, elapsed_time):
        """Extract and analyze results from training output"""
        eval_file = output_dir / 'eval_report.json'

        if not eval_file.exists():
            logger.warning(f"⚠️ No eval_report.json found for {config['name']}")
            return None

        try:
            with open(eval_file, 'r') as f:
                eval_data = json.load(f)

            # Extract key metrics
            f1_macro = eval_data.get('f1_macro', 0.0)
            f1_micro = eval_data.get('f1_micro', 0.0)

            # Calculate improvement over baseline
            improvement = ((f1_macro - self.baseline_f1) / self.baseline_f1) * 100

            # Determine success status
            success = f1_macro > 0.50  # Target threshold
            beats_baseline = f1_macro > self.baseline_f1
            meets_expectation = f1_macro >= (config['expected_f1'] - 0.02)  # 2% tolerance

            results = {
                'name': config['name'],
                'description': config['description'],
                'f1_macro': f1_macro,
                'f1_micro': f1_micro,
                'improvement_pct': improvement,
                'elapsed_time': elapsed_time,
                'success': success,
                'beats_baseline': beats_baseline,
                'meets_expectation': meets_expectation,
                'expected_f1': config['expected_f1'],
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"📊 {config['name']} Results:")
            logger.info(f"   F1 Macro: {f1_macro:.4f}")
            logger.info(f"   Improvement: {improvement:+.1f}%")
            logger.info(f"   Status: {'✅ SUCCESS' if success else '⚠️ NEEDS WORK'}")

            return results

        except Exception as e:
            logger.error(f"❌ Error extracting results for {config['name']}: {str(e)}")
            return None

    def run_all_tests(self):
        """Run systematic testing of all loss function configurations"""
        logger.info("🔬 SCIENTIFIC LOSS FUNCTION COMPARISON")
        logger.info("=" * 50)
        logger.info(f"Target: >50% F1-macro (vs {self.baseline_f1:.1%} baseline)")
        logger.info(f"Testing {len(self.test_configs)} configurations...")

        all_results = []

        for i, config in enumerate(self.test_configs, 1):
            logger.info(f"\n📋 Test {i}/{len(self.test_configs)}: {config['name']}")
            result = self.run_training_config(config)

            if result:
                all_results.append(result)
                self.results[config['name']] = result
            else:
                logger.error(f"❌ {config['name']} failed - skipping")

        # Analyze and report results
        self.analyze_results(all_results)

        return all_results

    def analyze_results(self, results):
        """Analyze and report comprehensive results"""
        if not results:
            logger.error("❌ No successful results to analyze!")
            return

        logger.info("\n" + "=" * 60)
        logger.info("🧪 SCIENTIFIC ANALYSIS COMPLETE")
        logger.info("=" * 60)

        # Sort by F1 score
        sorted_results = sorted(results, key=lambda x: x['f1_macro'], reverse=True)

        # Summary statistics
        f1_scores = [r['f1_macro'] for r in results]
        best_f1 = max(f1_scores)
        worst_f1 = min(f1_scores)
        avg_f1 = sum(f1_scores) / len(f1_scores)
        success_count = sum(1 for r in results if r['success'])
        baseline_beat_count = sum(1 for r in results if r['beats_baseline'])

        logger.info(f"📊 SUMMARY STATISTICS:")
        logger.info(f"   Best F1: {best_f1:.4f}")
        logger.info(f"   Worst F1: {worst_f1:.4f}")
        logger.info(f"   Average F1: {avg_f1:.4f}")
        logger.info(f"   Success rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
        logger.info(f"   Beat baseline: {baseline_beat_count}/{len(results)} ({baseline_beat_count/len(results)*100:.1f}%)")

        logger.info(f"\n🏆 DETAILED RESULTS:")
        for i, result in enumerate(sorted_results, 1):
            status = "🎉" if result['success'] else "📈" if result['beats_baseline'] else "📉"
            logger.info(f"   {i}. {result['name']}: {result['f1_macro']:.4f} ({result['improvement_pct']:+.1f}%) {status}")

        # Recommendations
        best_result = sorted_results[0]
        logger.info(f"\n💡 RECOMMENDATIONS:")

        if best_result['success']:
            logger.info(f"✅ WINNER: {best_result['name']} achieved {best_result['f1_macro']:.1%} F1!")
            logger.info(f"🚀 Proceed with {best_result['name']} for full multi-dataset training")
        elif best_result['beats_baseline']:
            logger.info(f"📈 IMPROVEMENT: {best_result['name']} beats baseline but needs tuning")
            logger.info(f"🔧 Consider extended training or hyperparameter optimization")
        else:
            logger.info(f"🔍 DEBUGGING NEEDED: All configurations underperformed")
            logger.info(f"🔧 Check data quality, loss implementations, and training setup")

        # Save comprehensive report
        self.save_report(results)

    def save_report(self, results):
        """Save comprehensive analysis report"""
        report_file = self.base_output_dir / f"loss_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'baseline_f1': self.baseline_f1,
                'target_f1': 0.50,
                'total_configs_tested': len(self.test_configs),
                'successful_configs': len(results)
            },
            'results': results,
            'analysis': {
                'best_f1': max(r['f1_macro'] for r in results) if results else 0,
                'worst_f1': min(r['f1_macro'] for r in results) if results else 0,
                'avg_f1': sum(r['f1_macro'] for r in results) / len(results) if results else 0,
                'success_rate': sum(1 for r in results if r['success']) / len(results) if results else 0
            }
        }

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"📄 Comprehensive report saved: {report_file}")

def main():
    """Main execution function"""
    tester = LossFunctionTester()
    results = tester.run_all_tests()

    if results:
        logger.info(f"\n🎉 Testing complete! Check outputs/loss_comparison/ for detailed results")
    else:
        logger.error(f"\n❌ All tests failed! Check logs for debugging information")

if __name__ == "__main__":
    main()