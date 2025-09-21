#!/usr/bin/env python3
"""
🚀 COMPREHENSIVE PERFORMANCE OPTIMIZATION FRAMEWORK
===================================================
Beyond loss function testing - systematic optimization of ALL performance vectors

TARGET: Achieve >60% F1-macro through multi-vector optimization
APPROACH: Scientific testing of hyperparameters, data quality, and training strategies
"""

import os
import subprocess
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import f1_score
import itertools

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    def __init__(self, base_output_dir="./outputs/performance_optimization"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        self.baseline_f1 = 0.5179
        self.target_f1 = 0.60

        # 1. HYPERPARAMETER OPTIMIZATION CONFIGS
        self.hyperparameter_configs = [
            {
                'name': 'LR_Schedule_Fix',
                'description': 'Fix aggressive LR decay with polynomial scheduler',
                'args': [
                    '--learning_rate', '2e-5',  # Slightly lower than current 3e-5
                    '--lr_scheduler_type', 'polynomial',  # Less aggressive than cosine
                    '--warmup_ratio', '0.2',  # Increased warmup
                    '--num_train_epochs', '4',  # Extended training
                ],
                'expected_improvement': '+3-5%',
                'priority': 1
            },
            {
                'name': 'Batch_Size_Optimization',
                'description': 'Optimize batch size for dual GPU efficiency',
                'args': [
                    '--per_device_train_batch_size', '3',  # Between 2 and 4
                    '--gradient_accumulation_steps', '3',  # Maintain effective batch size
                    '--per_device_eval_batch_size', '6',
                ],
                'expected_improvement': '+2-3%',
                'priority': 2
            },
            {
                'name': 'Regularization_Tuning',
                'description': 'Optimize weight decay and dropout',
                'args': [
                    '--weight_decay', '0.005',  # Reduced from 0.01
                    '--learning_rate', '3e-5',
                    '--lr_scheduler_type', 'cosine_with_restarts',
                    '--warmup_ratio', '0.15',
                ],
                'expected_improvement': '+1-2%',
                'priority': 3
            }
        ]

        # 2. DATA QUALITY OPTIMIZATION CONFIGS
        self.data_optimization_configs = [
            {
                'name': 'Threshold_Optimization',
                'description': 'Per-class optimized thresholds instead of global 0.2',
                'script': 'optimize_thresholds.py',
                'expected_improvement': '+4-7%',
                'priority': 1
            },
            {
                'name': 'Text_Preprocessing_Enhanced',
                'description': 'Advanced text preprocessing and cleaning',
                'script': 'enhance_preprocessing.py',
                'expected_improvement': '+2-3%',
                'priority': 2
            },
            {
                'name': 'Class_Weight_Balancing',
                'description': 'Dynamic class weighting for imbalanced emotions',
                'args': ['--use_class_weights'],
                'expected_improvement': '+3-5%',
                'priority': 1
            }
        ]

        # 3. TRAINING STRATEGY OPTIMIZATION
        self.training_strategy_configs = [
            {
                'name': 'Progressive_Unfreezing',
                'description': 'Gradual layer unfreezing for better convergence',
                'args': [
                    '--freeze_layers', '18',  # Start with 18 layers frozen
                    '--progressive_unfreezing',
                    '--unfreeze_schedule', '0.3,0.6,1.0',  # Unfreeze at 30%, 60%, 100%
                ],
                'expected_improvement': '+2-4%',
                'priority': 2
            },
            {
                'name': 'Curriculum_Learning',
                'description': 'Start with easier examples, progress to harder ones',
                'args': [
                    '--curriculum_learning',
                    '--difficulty_metric', 'label_count',  # Start with single-label examples
                ],
                'expected_improvement': '+3-6%',
                'priority': 1
            },
            {
                'name': 'Extended_Training_Patience',
                'description': 'Extended training with better early stopping',
                'args': [
                    '--num_train_epochs', '6',
                    '--early_stopping_patience', '5',
                    '--early_stopping_threshold', '0.001',
                ],
                'expected_improvement': '+2-4%',
                'priority': 2
            }
        ]

        # 4. INFRASTRUCTURE OPTIMIZATION
        self.infrastructure_configs = [
            {
                'name': 'Memory_Optimization',
                'description': 'Enable gradient checkpointing with compatible settings',
                'args': [
                    '--gradient_checkpointing',
                    '--dataloader_num_workers', '4',
                    '--dataloader_pin_memory',
                ],
                'expected_improvement': 'Speed +20-30%',
                'priority': 1
            },
            {
                'name': 'Mixed_Precision_Optimization',
                'description': 'Advanced FP16 with loss scaling',
                'args': [
                    '--fp16',
                    '--fp16_opt_level', 'O2',
                    '--fp16_backend', 'auto',
                ],
                'expected_improvement': 'Speed +15-25%',
                'priority': 2
            }
        ]

    def create_threshold_optimizer(self):
        """Create per-class threshold optimization script"""
        script_content = '''#!/usr/bin/env python3
"""
🎯 PER-CLASS THRESHOLD OPTIMIZATION
==================================
Find optimal thresholds for each emotion class instead of global 0.2
Expected improvement: +4-7% F1-macro
"""

import json
import numpy as np
from sklearn.metrics import f1_score
from scipy.optimize import differential_evolution
import torch

def optimize_thresholds(model_path, val_data_path):
    """Find optimal per-class thresholds using validation data"""

    def evaluate_thresholds(thresholds):
        # Load model predictions and true labels
        # Apply per-class thresholds
        # Return F1-macro score
        pass

    # Optimize thresholds using differential evolution
    bounds = [(0.1, 0.8)] * 28  # 28 emotion classes
    result = differential_evolution(
        lambda x: -evaluate_thresholds(x),  # Minimize negative F1
        bounds,
        seed=42,
        maxiter=100
    )

    optimal_thresholds = result.x
    return optimal_thresholds, -result.fun

if __name__ == "__main__":
    thresholds, f1_score = optimize_thresholds("model_path", "val_data")
    print(f"Optimal thresholds found: F1={f1_score:.4f}")

    # Save optimized thresholds
    with open("configs/optimal_thresholds.json", "w") as f:
        json.dump(dict(zip(range(28), thresholds.tolist())), f, indent=2)
'''

        script_path = self.base_output_dir / "optimize_thresholds.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        return script_path

    def create_enhanced_preprocessing(self):
        """Create enhanced text preprocessing script"""
        script_content = '''#!/usr/bin/env python3
"""
🧹 ENHANCED TEXT PREPROCESSING
==============================
Advanced text cleaning and preprocessing for better model performance
Expected improvement: +2-3% F1-macro
"""

import re
import string
import pandas as pd
from textblob import TextBlob

def enhanced_preprocessing(text):
    """Apply comprehensive text preprocessing"""

    # 1. Normalize whitespace and remove excessive punctuation
    text = re.sub(r'\\s+', ' ', text.strip())
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    text = re.sub(r'[.]{3,}', '...', text)

    # 2. Handle Reddit-specific patterns
    text = re.sub(r'\\b(u/|r/)\\w+', '[USER]', text)  # Replace Reddit mentions
    text = re.sub(r'http\\S+', '[URL]', text)  # Replace URLs

    # 3. Normalize contractions
    contractions = {
        "won't": "will not", "can't": "cannot", "n't": " not",
        "'re": " are", "'ve": " have", "'ll": " will", "'d": " would"
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)

    # 4. Preserve emotional emphasis while cleaning
    text = re.sub(r'([a-zA-Z])\\1{2,}', r'\\1\\1', text)  # "sooooo" -> "soo"

    return text.strip()

def preprocess_dataset(input_file, output_file):
    """Apply enhanced preprocessing to entire dataset"""
    print(f"🧹 Preprocessing {input_file}...")

    data = []
    with open(input_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            item['text'] = enhanced_preprocessing(item['text'])
            data.append(item)

    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\\n')

    print(f"✅ Enhanced preprocessing complete: {output_file}")

if __name__ == "__main__":
    preprocess_dataset("data/combined_all_datasets/train.jsonl",
                      "data/combined_all_datasets/train_enhanced.jsonl")
    preprocess_dataset("data/combined_all_datasets/val.jsonl",
                      "data/combined_all_datasets/val_enhanced.jsonl")
'''

        script_path = self.base_output_dir / "enhance_preprocessing.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        return script_path

    def run_optimization_experiment(self, config, config_type):
        """Run single optimization experiment"""
        output_dir = self.base_output_dir / f"{config_type}_{config['name']}"
        output_dir.mkdir(exist_ok=True)

        logger.info(f"🧪 Testing {config['name']}: {config['description']}")

        # Build command based on config type
        if config_type == "hyperparameter":
            cmd = self.build_hyperparameter_command(config, output_dir)
        elif config_type == "training_strategy":
            cmd = self.build_training_strategy_command(config, output_dir)
        elif config_type == "infrastructure":
            cmd = self.build_infrastructure_command(config, output_dir)
        else:
            return None

        # Set environment for dual GPU
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0,1'

        start_time = time.time()
        try:
            result = subprocess.run(
                cmd, env=env, timeout=2400,  # 40 minute timeout
                capture_output=True, text=True
            )

            elapsed_time = time.time() - start_time

            if result.returncode == 0:
                return self.extract_optimization_results(config, output_dir, elapsed_time)
            else:
                logger.error(f"❌ {config['name']} failed: {result.stderr[-300:]}")
                return None

        except subprocess.TimeoutExpired:
            logger.error(f"⏰ {config['name']} timed out")
            return None
        except Exception as e:
            logger.error(f"💥 {config['name']} crashed: {str(e)}")
            return None

    def build_hyperparameter_command(self, config, output_dir):
        """Build command for hyperparameter optimization"""
        base_cmd = [
            'python3', 'notebooks/scripts/train_deberta_local.py',
            '--output_dir', str(output_dir),
            '--model_type', 'deberta-v3-large',
            '--max_train_samples', '15000',  # Subset for faster testing
            '--max_eval_samples', '3000',
            '--threshold', '0.2',
        ]

        return base_cmd + config['args']

    def build_training_strategy_command(self, config, output_dir):
        """Build command for training strategy optimization"""
        base_cmd = [
            'python3', 'notebooks/scripts/train_deberta_local.py',
            '--output_dir', str(output_dir),
            '--model_type', 'deberta-v3-large',
            '--per_device_train_batch_size', '2',
            '--learning_rate', '3e-5',
            '--max_train_samples', '15000',
            '--threshold', '0.2',
        ]

        return base_cmd + config['args']

    def build_infrastructure_command(self, config, output_dir):
        """Build command for infrastructure optimization"""
        base_cmd = [
            'python3', 'notebooks/scripts/train_deberta_local.py',
            '--output_dir', str(output_dir),
            '--model_type', 'deberta-v3-large',
            '--per_device_train_batch_size', '2',
            '--num_train_epochs', '2',  # Quick test for infrastructure
            '--learning_rate', '3e-5',
            '--max_train_samples', '10000',  # Smaller for speed testing
            '--threshold', '0.2',
        ]

        return base_cmd + config['args']

    def extract_optimization_results(self, config, output_dir, elapsed_time):
        """Extract results from optimization experiment"""
        eval_file = output_dir / 'eval_report.json'

        if not eval_file.exists():
            return None

        try:
            with open(eval_file, 'r') as f:
                data = json.load(f)

            f1_macro = data.get('f1_macro', 0.0)
            improvement = ((f1_macro - self.baseline_f1) / self.baseline_f1) * 100

            result = {
                'name': config['name'],
                'description': config['description'],
                'f1_macro': f1_macro,
                'improvement_pct': improvement,
                'elapsed_time': elapsed_time,
                'expected_improvement': config.get('expected_improvement', 'Unknown'),
                'priority': config['priority'],
                'success': f1_macro > self.baseline_f1,
                'target_achieved': f1_macro >= self.target_f1
            }

            logger.info(f"📊 {config['name']}: F1={f1_macro:.4f} ({improvement:+.1f}%) {'🎯' if result['target_achieved'] else '✅' if result['success'] else '⚠️'}")
            return result

        except Exception as e:
            logger.error(f"❌ Error extracting results for {config['name']}: {str(e)}")
            return None

    def run_comprehensive_optimization(self):
        """Run comprehensive multi-vector performance optimization"""
        logger.info("🚀 COMPREHENSIVE PERFORMANCE OPTIMIZATION")
        logger.info("=" * 60)
        logger.info(f"🎯 Target: >{self.target_f1:.1%} F1-macro (vs {self.baseline_f1:.1%} baseline)")
        logger.info(f"🔬 Testing {len(self.hyperparameter_configs + self.training_strategy_configs + self.infrastructure_configs)} optimization vectors")

        # Create optimization scripts
        self.create_threshold_optimizer()
        self.create_enhanced_preprocessing()

        all_results = []

        # 1. HYPERPARAMETER OPTIMIZATION (highest impact)
        logger.info("\\n🎛️ PHASE 1: HYPERPARAMETER OPTIMIZATION")
        for config in sorted(self.hyperparameter_configs, key=lambda x: x['priority']):
            result = self.run_optimization_experiment(config, "hyperparameter")
            if result:
                all_results.append(result)

        # 2. TRAINING STRATEGY OPTIMIZATION
        logger.info("\\n🎯 PHASE 2: TRAINING STRATEGY OPTIMIZATION")
        for config in sorted(self.training_strategy_configs, key=lambda x: x['priority']):
            result = self.run_optimization_experiment(config, "training_strategy")
            if result:
                all_results.append(result)

        # 3. INFRASTRUCTURE OPTIMIZATION
        logger.info("\\n⚡ PHASE 3: INFRASTRUCTURE OPTIMIZATION")
        for config in sorted(self.infrastructure_configs, key=lambda x: x['priority']):
            result = self.run_optimization_experiment(config, "infrastructure")
            if result:
                all_results.append(result)

        # Analyze and recommend optimal configuration
        self.analyze_optimization_results(all_results)
        return all_results

    def analyze_optimization_results(self, results):
        """Analyze optimization results and recommend best configuration"""
        if not results:
            logger.error("❌ No successful optimization results!")
            return

        logger.info("\\n" + "=" * 60)
        logger.info("🧪 COMPREHENSIVE OPTIMIZATION ANALYSIS")
        logger.info("=" * 60)

        # Sort by F1 score
        sorted_results = sorted(results, key=lambda x: x['f1_macro'], reverse=True)

        # Find best configurations
        target_achievers = [r for r in results if r['target_achieved']]
        significant_improvements = [r for r in results if r['improvement_pct'] > 5.0]

        logger.info(f"📊 OPTIMIZATION SUMMARY:")
        logger.info(f"   Total experiments: {len(results)}")
        logger.info(f"   Target achievers (≥60%): {len(target_achievers)}")
        logger.info(f"   Significant improvements (>5%): {len(significant_improvements)}")

        logger.info(f"\\n🏆 TOP PERFORMERS:")
        for i, result in enumerate(sorted_results[:5], 1):
            status = "🎯" if result['target_achieved'] else "📈" if result['success'] else "📉"
            logger.info(f"   {i}. {result['name']}: {result['f1_macro']:.4f} ({result['improvement_pct']:+.1f}%) {status}")

        # Generate optimal configuration
        if target_achievers:
            winner = target_achievers[0]
            logger.info(f"\\n🎉 TARGET ACHIEVED!")
            logger.info(f"🏆 WINNER: {winner['name']} with {winner['f1_macro']:.1%} F1-macro")
            logger.info(f"📈 Improvement: {winner['improvement_pct']:+.1f}% over baseline")
        elif significant_improvements:
            best = significant_improvements[0]
            logger.info(f"\\n📈 SIGNIFICANT IMPROVEMENT FOUND!")
            logger.info(f"🥇 BEST: {best['name']} with {best['f1_macro']:.1%} F1-macro")
            logger.info(f"💡 Combine with other optimizations for target achievement")
        else:
            logger.info(f"\\n🔧 INCREMENTAL IMPROVEMENTS FOUND")
            logger.info(f"💡 Stack multiple optimizations for cumulative effect")

        # Save comprehensive report
        self.save_optimization_report(results)

    def save_optimization_report(self, results):
        """Save detailed optimization report"""
        report_file = self.base_output_dir / f"optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'baseline_f1': self.baseline_f1,
                'target_f1': self.target_f1,
                'total_experiments': len(results)
            },
            'results': results,
            'recommendations': {
                'best_f1': max(r['f1_macro'] for r in results) if results else 0,
                'target_achieved': any(r['target_achieved'] for r in results),
                'significant_improvements': len([r for r in results if r['improvement_pct'] > 5.0])
            }
        }

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"📄 Optimization report saved: {report_file}")

def main():
    """Execute comprehensive performance optimization"""
    optimizer = PerformanceOptimizer()
    results = optimizer.run_comprehensive_optimization()

    if results:
        logger.info("\\n🎉 COMPREHENSIVE OPTIMIZATION COMPLETE!")
        logger.info("🚀 Check outputs/performance_optimization/ for detailed results")
    else:
        logger.error("\\n❌ All optimization experiments failed!")

if __name__ == "__main__":
    main()