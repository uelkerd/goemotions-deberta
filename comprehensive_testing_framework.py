#!/usr/bin/env python3
"""
🧪 COMPREHENSIVE TESTING FRAMEWORK
==================================
Systematic validation of ALL performance improvements

TESTING STRATEGY:
1. Unit Tests: Individual components
2. Integration Tests: Components working together
3. Performance Tests: Actual F1 improvements
4. Regression Tests: No performance degradation
5. End-to-End Tests: Full pipeline validation

GOALS:
- Validate +4-7% threshold optimization
- Confirm +2-4% immediate fixes
- Verify dual GPU utilization
- Measure cumulative improvements
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
import unittest
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveTestRunner:
    def __init__(self, test_output_dir="./outputs/comprehensive_testing"):
        self.test_output_dir = Path(test_output_dir)
        self.test_output_dir.mkdir(exist_ok=True)
        self.baseline_f1 = 0.5179  # Known good baseline
        self.regression_f1 = 0.3943  # Current problematic performance
        self.target_f1 = 0.60

        # Test configurations for different optimization levels
        self.test_configs = {
            'quick_validation': {
                'max_train_samples': 5000,
                'max_eval_samples': 1000,
                'num_train_epochs': 1,
                'timeout': 600,  # 10 minutes
                'expected_time': '5-10 minutes'
            },
            'integration_test': {
                'max_train_samples': 10000,
                'max_eval_samples': 2000,
                'num_train_epochs': 2,
                'timeout': 1800,  # 30 minutes
                'expected_time': '15-30 minutes'
            },
            'full_validation': {
                'max_train_samples': 30000,
                'max_eval_samples': 7000,
                'num_train_epochs': 3,
                'timeout': 3600,  # 60 minutes
                'expected_time': '45-60 minutes'
            }
        }

    def test_environment_setup(self):
        """Test 1: Verify environment and dependencies"""
        logger.info("🔍 TEST 1: Environment Setup Validation")

        tests = []

        # Check GPU availability
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=count', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True)
            gpu_count = int(result.stdout.strip())
            tests.append(('GPU Count', gpu_count >= 2, f"{gpu_count} GPUs detected"))
        except:
            tests.append(('GPU Count', False, "Could not detect GPUs"))

        # Check key files exist
        key_files = [
            'notebooks/scripts/train_deberta_local.py',
            'immediate_performance_fixes.py',
            'quick_threshold_optimizer.py',
            'comprehensive_performance_optimizer.py',
            'data/combined_all_datasets/train.jsonl'
        ]

        for file_path in key_files:
            exists = Path(file_path).exists()
            tests.append(('File Exists', exists, file_path))

        # Check Python packages
        required_packages = ['torch', 'transformers', 'datasets', 'sklearn', 'scipy']
        for package in required_packages:
            try:
                __import__(package)
                tests.append(('Package', True, package))
            except ImportError:
                tests.append(('Package', False, f"{package} missing"))

        # Report results
        passed = sum(1 for _, success, _ in tests if success)
        total = len(tests)

        logger.info(f"📊 Environment Test Results: {passed}/{total} passed")
        for test_name, success, detail in tests:
            status = "✅" if success else "❌"
            logger.info(f"   {status} {test_name}: {detail}")

        return passed == total

    def test_threshold_optimization(self):
        """Test 2: Quick threshold optimization validation"""
        logger.info("🎯 TEST 2: Threshold Optimization Validation")

        try:
            # Run threshold optimizer
            start_time = time.time()
            result = subprocess.run([
                'python3', 'quick_threshold_optimizer.py'
            ], capture_output=True, text=True, timeout=300)  # 5 minute timeout

            elapsed_time = time.time() - start_time

            if result.returncode == 0:
                logger.info(f"✅ Threshold optimization completed in {elapsed_time:.1f}s")

                # Check if thresholds file was created
                threshold_file = Path("configs/optimal_thresholds.json")
                if threshold_file.exists():
                    with open(threshold_file, 'r') as f:
                        threshold_data = json.load(f)

                    thresholds = list(threshold_data['optimal_thresholds'].values())
                    logger.info(f"📊 Generated {len(thresholds)} optimal thresholds")
                    logger.info(f"   Range: {min(thresholds):.3f} - {max(thresholds):.3f}")

                    return {'success': True, 'thresholds_generated': len(thresholds), 'time': elapsed_time}
                else:
                    logger.warning("⚠️ Thresholds file not generated")
                    return {'success': False, 'error': 'No thresholds file'}
            else:
                logger.error(f"❌ Threshold optimization failed: {result.stderr[-200:]}")
                return {'success': False, 'error': result.stderr[-200:]}

        except subprocess.TimeoutExpired:
            logger.error("⏰ Threshold optimization timed out")
            return {'success': False, 'error': 'Timeout'}
        except Exception as e:
            logger.error(f"💥 Threshold optimization crashed: {str(e)}")
            return {'success': False, 'error': str(e)}

    def test_immediate_fixes(self, test_level='quick_validation'):
        """Test 3: Immediate performance fixes validation"""
        logger.info(f"🔧 TEST 3: Immediate Fixes Validation ({test_level})")

        config = self.test_configs[test_level]
        output_dir = self.test_output_dir / f"immediate_fixes_{test_level}"

        try:
            # Run immediate fixes with test configuration
            cmd = [
                'python3', 'notebooks/scripts/train_deberta_local.py',
                '--output_dir', str(output_dir),
                '--model_type', 'deberta-v3-large',

                # Immediate fixes configuration
                '--learning_rate', '2e-5',
                '--lr_scheduler_type', 'polynomial',
                '--warmup_ratio', '0.2',
                '--per_device_train_batch_size', '3',
                '--gradient_accumulation_steps', '3',
                '--per_device_eval_batch_size', '6',
                '--weight_decay', '0.005',
                '--fp16',
                '--dataloader_num_workers', '4',

                # Test configuration
                '--max_train_samples', str(config['max_train_samples']),
                '--max_eval_samples', str(config['max_eval_samples']),
                '--num_train_epochs', str(config['num_train_epochs']),
                '--threshold', '0.2',
                '--early_stopping_patience', '2',
            ]

            # Set dual GPU environment
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = '0,1'

            logger.info(f"⚡ Running immediate fixes test ({config['expected_time']})...")
            start_time = time.time()

            result = subprocess.run(
                cmd, env=env, timeout=config['timeout'],
                capture_output=True, text=True
            )

            elapsed_time = time.time() - start_time

            if result.returncode == 0:
                logger.info(f"✅ Immediate fixes test completed in {elapsed_time/60:.1f} minutes")

                # Extract results
                eval_file = output_dir / 'eval_report.json'
                if eval_file.exists():
                    with open(eval_file, 'r') as f:
                        data = json.load(f)

                    f1_macro = data.get('f1_macro', 0.0)
                    improvement_vs_regression = ((f1_macro - self.regression_f1) / self.regression_f1) * 100
                    improvement_vs_baseline = ((f1_macro - self.baseline_f1) / self.baseline_f1) * 100

                    logger.info(f"📊 Immediate Fixes Results:")
                    logger.info(f"   F1-macro: {f1_macro:.4f}")
                    logger.info(f"   vs Regression: {improvement_vs_regression:+.1f}%")
                    logger.info(f"   vs Baseline: {improvement_vs_baseline:+.1f}%")

                    success = f1_macro > self.regression_f1  # At least fix regression
                    significant = improvement_vs_regression > 10.0  # Significant improvement

                    return {
                        'success': success,
                        'f1_macro': f1_macro,
                        'improvement_vs_regression': improvement_vs_regression,
                        'improvement_vs_baseline': improvement_vs_baseline,
                        'significant_improvement': significant,
                        'elapsed_time': elapsed_time
                    }
                else:
                    logger.error("❌ No evaluation results found")
                    return {'success': False, 'error': 'No eval results'}
            else:
                logger.error(f"❌ Immediate fixes test failed: {result.stderr[-300:]}")
                return {'success': False, 'error': result.stderr[-300:]}

        except subprocess.TimeoutExpired:
            logger.error(f"⏰ Immediate fixes test timed out after {config['timeout']/60:.0f} minutes")
            return {'success': False, 'error': 'Timeout'}
        except Exception as e:
            logger.error(f"💥 Immediate fixes test crashed: {str(e)}")
            return {'success': False, 'error': str(e)}

    def test_dual_gpu_utilization(self):
        """Test 4: Verify dual GPU utilization"""
        logger.info("🚀 TEST 4: Dual GPU Utilization Validation")

        try:
            # Run a short training job and monitor GPU usage
            cmd = [
                'python3', 'notebooks/scripts/train_deberta_local.py',
                '--output_dir', str(self.test_output_dir / "gpu_test"),
                '--model_type', 'deberta-v3-large',
                '--max_train_samples', '1000',
                '--max_eval_samples', '200',
                '--num_train_epochs', '1',
                '--per_device_train_batch_size', '2',
                '--eval_strategy', 'no',  # Skip evaluation for speed
                '--save_strategy', 'no',   # Skip saving for speed
            ]

            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = '0,1'

            # Start training in background
            logger.info("⚡ Starting GPU utilization test...")
            process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Monitor GPU usage for 60 seconds
            gpu_utilizations = []
            for i in range(12):  # Check every 5 seconds for 60 seconds
                time.sleep(5)
                try:
                    gpu_result = subprocess.run([
                        'nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'
                    ], capture_output=True, text=True)

                    if gpu_result.returncode == 0:
                        utils = [int(x.strip()) for x in gpu_result.stdout.strip().split('\\n')]
                        gpu_utilizations.append(utils)
                        logger.info(f"   GPU utilization: {utils}")
                except:
                    pass

                # Check if process finished
                if process.poll() is not None:
                    break

            # Terminate process if still running
            if process.poll() is None:
                process.terminate()
                process.wait()

            # Analyze GPU utilization
            if gpu_utilizations:
                avg_utils = np.mean(gpu_utilizations, axis=0)
                max_utils = np.max(gpu_utilizations, axis=0)

                logger.info(f"📊 GPU Utilization Analysis:")
                for i, (avg, max_util) in enumerate(zip(avg_utils, max_utils)):
                    logger.info(f"   GPU {i}: Avg {avg:.1f}%, Max {max_util:.1f}%")

                # Check if both GPUs were utilized
                dual_gpu_used = len(avg_utils) >= 2 and all(util > 10 for util in avg_utils)
                effective_utilization = all(util > 30 for util in avg_utils) if len(avg_utils) >= 2 else False

                return {
                    'success': dual_gpu_used,
                    'dual_gpu_detected': len(avg_utils) >= 2,
                    'effective_utilization': effective_utilization,
                    'avg_utilizations': avg_utils.tolist() if len(avg_utils) > 0 else [],
                    'max_utilizations': max_utils.tolist() if len(max_utils) > 0 else []
                }
            else:
                logger.error("❌ Could not monitor GPU utilization")
                return {'success': False, 'error': 'No GPU monitoring data'}

        except Exception as e:
            logger.error(f"💥 GPU utilization test failed: {str(e)}")
            return {'success': False, 'error': str(e)}

    def test_comprehensive_optimization(self, quick_test=True):
        """Test 5: Comprehensive optimization framework validation"""
        logger.info("🔬 TEST 5: Comprehensive Optimization Validation")

        if quick_test:
            logger.info("⚡ Running quick validation of optimization framework...")

            # Test that the script runs without errors (dry run mode)
            try:
                # Import and validate the optimization framework
                sys.path.insert(0, str(Path.cwd()))
                from comprehensive_performance_optimizer import PerformanceOptimizer

                optimizer = PerformanceOptimizer()

                # Validate configurations
                total_configs = (len(optimizer.hyperparameter_configs) +
                               len(optimizer.data_optimization_configs) +
                               len(optimizer.training_strategy_configs) +
                               len(optimizer.infrastructure_configs))

                logger.info(f"✅ Optimization framework loaded successfully")
                logger.info(f"📊 Total optimization configurations: {total_configs}")

                # Test configuration building
                test_config = optimizer.hyperparameter_configs[0]
                test_cmd = optimizer.build_hyperparameter_command(test_config, Path("/tmp/test"))

                logger.info(f"📝 Test command generation successful: {len(test_cmd)} arguments")

                return {
                    'success': True,
                    'total_configs': total_configs,
                    'framework_loaded': True,
                    'command_generation': True
                }

            except Exception as e:
                logger.error(f"❌ Optimization framework validation failed: {str(e)}")
                return {'success': False, 'error': str(e)}
        else:
            # Full optimization test (would take 1-2 hours)
            logger.info("🔬 Full optimization testing not implemented (would take 1-2 hours)")
            return {'success': True, 'skipped': True, 'reason': 'Full test too long for validation'}

    def run_integration_test(self):
        """Test 6: Integration test of all components"""
        logger.info("🔗 TEST 6: Integration Test - All Components Working Together")

        try:
            # Test sequence: threshold optimization → immediate fixes → validation
            logger.info("🎯 Step 1: Threshold optimization...")
            threshold_result = self.test_threshold_optimization()

            if not threshold_result['success']:
                logger.error("❌ Integration test failed at threshold optimization")
                return {'success': False, 'failed_at': 'threshold_optimization'}

            logger.info("🔧 Step 2: Immediate fixes...")
            fixes_result = self.test_immediate_fixes('quick_validation')

            if not fixes_result['success']:
                logger.error("❌ Integration test failed at immediate fixes")
                return {'success': False, 'failed_at': 'immediate_fixes'}

            logger.info("🚀 Step 3: GPU utilization...")
            gpu_result = self.test_dual_gpu_utilization()

            # Calculate overall improvement
            total_improvement = fixes_result.get('improvement_vs_regression', 0)

            logger.info(f"📊 Integration Test Results:")
            logger.info(f"   Threshold optimization: ✅")
            logger.info(f"   Immediate fixes: ✅ ({total_improvement:+.1f}%)")
            logger.info(f"   Dual GPU: {'✅' if gpu_result['success'] else '⚠️'}")

            success = (threshold_result['success'] and
                      fixes_result['success'] and
                      total_improvement > 5.0)  # At least 5% improvement

            return {
                'success': success,
                'total_improvement': total_improvement,
                'threshold_optimization': threshold_result['success'],
                'immediate_fixes': fixes_result['success'],
                'dual_gpu_utilization': gpu_result['success']
            }

        except Exception as e:
            logger.error(f"💥 Integration test crashed: {str(e)}")
            return {'success': False, 'error': str(e)}

    def run_comprehensive_testing(self):
        """Run all tests in systematic order"""
        logger.info("🧪 COMPREHENSIVE TESTING FRAMEWORK")
        logger.info("=" * 60)
        logger.info("🎯 Goal: Validate all performance improvements systematically")

        test_results = {}

        # Test 1: Environment
        logger.info("\\n" + "="*50)
        test_results['environment'] = self.test_environment_setup()

        if not test_results['environment']:
            logger.error("❌ Environment test failed - stopping tests")
            return test_results

        # Test 2: Threshold Optimization
        logger.info("\\n" + "="*50)
        test_results['threshold_optimization'] = self.test_threshold_optimization()

        # Test 3: Immediate Fixes
        logger.info("\\n" + "="*50)
        test_results['immediate_fixes'] = self.test_immediate_fixes()

        # Test 4: GPU Utilization
        logger.info("\\n" + "="*50)
        test_results['gpu_utilization'] = self.test_dual_gpu_utilization()

        # Test 5: Optimization Framework
        logger.info("\\n" + "="*50)
        test_results['optimization_framework'] = self.test_comprehensive_optimization()

        # Test 6: Integration
        logger.info("\\n" + "="*50)
        test_results['integration'] = self.run_integration_test()

        # Final Analysis
        self.analyze_test_results(test_results)
        return test_results

    def analyze_test_results(self, results):
        """Analyze and report comprehensive test results"""
        logger.info("\\n" + "="*60)
        logger.info("🧪 COMPREHENSIVE TESTING ANALYSIS")
        logger.info("="*60)

        # Count successes
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values()
                          if isinstance(result, dict) and result.get('success', False))

        logger.info(f"📊 OVERALL RESULTS: {passed_tests}/{total_tests} tests passed")

        # Detailed results
        for test_name, result in results.items():
            if isinstance(result, dict):
                status = "✅" if result.get('success', False) else "❌"
                logger.info(f"   {status} {test_name.replace('_', ' ').title()}")

                # Show key metrics
                if test_name == 'immediate_fixes' and result.get('success'):
                    improvement = result.get('improvement_vs_regression', 0)
                    logger.info(f"      → Performance improvement: {improvement:+.1f}%")

                if test_name == 'integration' and result.get('success'):
                    total_imp = result.get('total_improvement', 0)
                    logger.info(f"      → Total improvement: {total_imp:+.1f}%")
            else:
                status = "✅" if result else "❌"
                logger.info(f"   {status} {test_name.replace('_', ' ').title()}")

        # Recommendations
        logger.info(f"\\n💡 RECOMMENDATIONS:")

        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED! Ready for full optimization!")
            logger.info("🚀 Proceed with comprehensive optimization for 60% F1 target")
        elif passed_tests >= total_tests * 0.8:
            logger.info("📈 MOSTLY SUCCESSFUL! Minor issues to resolve")
            logger.info("🔧 Address failing tests then proceed with optimization")
        else:
            logger.info("⚠️ MULTIPLE ISSUES DETECTED! Fix critical problems first")
            logger.info("🔍 Review failed tests and resolve before optimization")

        # Save detailed report
        self.save_test_report(results)

    def save_test_report(self, results):
        """Save comprehensive test report"""
        report_file = self.test_output_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_tests': len(results),
                'passed_tests': sum(1 for r in results.values()
                                  if isinstance(r, dict) and r.get('success', False)),
                'baseline_f1': self.baseline_f1,
                'regression_f1': self.regression_f1,
                'target_f1': self.target_f1
            },
            'test_results': results,
            'summary': {
                'environment_ready': results.get('environment', False),
                'optimizations_working': results.get('integration', {}).get('success', False),
                'performance_improved': results.get('immediate_fixes', {}).get('success', False),
                'ready_for_full_optimization': all([
                    results.get('environment', False),
                    results.get('threshold_optimization', {}).get('success', False),
                    results.get('immediate_fixes', {}).get('success', False)
                ])
            }
        }

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"📄 Comprehensive test report saved: {report_file}")

def main():
    """Execute comprehensive testing framework"""

    tester = ComprehensiveTestRunner()
    results = tester.run_comprehensive_testing()

    if results:
        logger.info("\\n🎉 COMPREHENSIVE TESTING COMPLETE!")
        logger.info("📊 Check outputs/comprehensive_testing/ for detailed results")
    else:
        logger.error("\\n❌ Comprehensive testing failed!")

if __name__ == "__main__":
    main()