#!/usr/bin/env python3
"""
🧪 COMPREHENSIVE TEST EXECUTION SCRIPT
======================================
Orchestrates all testing phases for our performance improvements

TESTING PHASES:
1. Quick Validation (5-10 minutes) - Immediate feedback
2. Comprehensive Testing (30-60 minutes) - Full validation
3. Performance Benchmarking (60+ minutes) - Production readiness

USAGE:
  python run_all_tests.py --quick           # Quick validation only
  python run_all_tests.py --comprehensive   # Full testing suite
  python run_all_tests.py --benchmark       # Production benchmarking
  python run_all_tests.py --all             # Everything (2+ hours)
"""

import os
import sys
import subprocess
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestOrchestrator:
    def __init__(self):
        self.start_time = time.time()
        self.test_results = {}
        self.outputs_dir = Path("./outputs/test_orchestration")
        self.outputs_dir.mkdir(exist_ok=True)

    def run_quick_validation(self):
        """Run quick validation (5-10 minutes)"""
        logger.info("⚡ PHASE 1: QUICK VALIDATION")
        logger.info("=" * 50)

        try:
            result = subprocess.run([
                'python3', 'quick_validation_test.py'
            ], capture_output=True, text=True, timeout=900)  # 15 min timeout

            success = result.returncode == 0
            self.test_results['quick_validation'] = {
                'success': success,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }

            if success:
                logger.info("✅ Quick validation PASSED")
                return True
            else:
                logger.error("❌ Quick validation FAILED")
                logger.error(f"Error: {result.stderr[-300:]}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("⏰ Quick validation timed out")
            self.test_results['quick_validation'] = {'success': False, 'error': 'timeout'}
            return False
        except Exception as e:
            logger.error(f"💥 Quick validation crashed: {str(e)}")
            self.test_results['quick_validation'] = {'success': False, 'error': str(e)}
            return False

    def run_comprehensive_testing(self):
        """Run comprehensive testing (30-60 minutes)"""
        logger.info("🔬 PHASE 2: COMPREHENSIVE TESTING")
        logger.info("=" * 50)

        try:
            result = subprocess.run([
                'python3', 'comprehensive_testing_framework.py'
            ], capture_output=True, text=True, timeout=4800)  # 80 min timeout

            success = result.returncode == 0
            self.test_results['comprehensive_testing'] = {
                'success': success,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }

            if success:
                logger.info("✅ Comprehensive testing PASSED")
                return True
            else:
                logger.error("❌ Comprehensive testing FAILED")
                logger.error(f"Error: {result.stderr[-300:]}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("⏰ Comprehensive testing timed out")
            self.test_results['comprehensive_testing'] = {'success': False, 'error': 'timeout'}
            return False
        except Exception as e:
            logger.error(f"💥 Comprehensive testing crashed: {str(e)}")
            self.test_results['comprehensive_testing'] = {'success': False, 'error': str(e)}
            return False

    def run_performance_benchmark(self):
        """Run performance benchmarking (60+ minutes)"""
        logger.info("🏆 PHASE 3: PERFORMANCE BENCHMARKING")
        logger.info("=" * 50)

        try:
            # Run the comprehensive optimization with full dataset
            result = subprocess.run([
                'python3', 'comprehensive_performance_optimizer.py'
            ], capture_output=True, text=True, timeout=7200)  # 2 hour timeout

            success = result.returncode == 0
            self.test_results['performance_benchmark'] = {
                'success': success,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }

            if success:
                logger.info("✅ Performance benchmarking PASSED")

                # Try to extract final F1 score
                try:
                    # Look for optimization report
                    report_files = list(Path("outputs/performance_optimization").glob("optimization_report_*.json"))
                    if report_files:
                        with open(report_files[-1], 'r') as f:
                            report_data = json.load(f)

                        best_f1 = report_data.get('recommendations', {}).get('best_f1', 0)
                        logger.info(f"📊 Best F1 achieved: {best_f1:.4f}")

                        if best_f1 >= 0.60:
                            logger.info("🎯 TARGET ACHIEVED! 60% F1-macro reached!")
                        elif best_f1 > 0.52:
                            logger.info("📈 Significant improvement over baseline!")
                        else:
                            logger.info("📊 Incremental improvements measured")

                except Exception as e:
                    logger.warning(f"Could not extract final F1 score: {str(e)}")

                return True
            else:
                logger.error("❌ Performance benchmarking FAILED")
                logger.error(f"Error: {result.stderr[-300:]}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("⏰ Performance benchmarking timed out after 2 hours")
            self.test_results['performance_benchmark'] = {'success': False, 'error': 'timeout'}
            return False
        except Exception as e:
            logger.error(f"💥 Performance benchmarking crashed: {str(e)}")
            self.test_results['performance_benchmark'] = {'success': False, 'error': str(e)}
            return False

    def run_parallel_loss_testing(self):
        """Run parallel loss function testing"""
        logger.info("🚀 BONUS: PARALLEL LOSS FUNCTION TESTING")
        logger.info("=" * 50)

        try:
            result = subprocess.run([
                'python3', 'parallel_loss_testing.py'
            ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout

            success = result.returncode == 0
            self.test_results['parallel_loss_testing'] = {
                'success': success,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }

            if success:
                logger.info("✅ Parallel loss testing PASSED")
                return True
            else:
                logger.error("❌ Parallel loss testing FAILED")
                logger.error(f"Error: {result.stderr[-300:]}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("⏰ Parallel loss testing timed out")
            self.test_results['parallel_loss_testing'] = {'success': False, 'error': 'timeout'}
            return False
        except Exception as e:
            logger.error(f"💥 Parallel loss testing crashed: {str(e)}")
            self.test_results['parallel_loss_testing'] = {'success': False, 'error': str(e)}
            return False

    def generate_final_report(self):
        """Generate comprehensive final report"""
        total_time = time.time() - self.start_time

        logger.info("\\n" + "="*60)
        logger.info("📊 FINAL TEST ORCHESTRATION REPORT")
        logger.info("="*60)

        # Count successes
        total_phases = len(self.test_results)
        passed_phases = sum(1 for result in self.test_results.values() if result.get('success', False))

        logger.info(f"⏱️ Total execution time: {total_time/3600:.2f} hours")
        logger.info(f"📋 Test phases completed: {total_phases}")
        logger.info(f"✅ Phases passed: {passed_phases}/{total_phases}")

        # Detailed results
        logger.info(f"\\n📊 PHASE RESULTS:")
        for phase_name, result in self.test_results.items():
            status = "✅" if result.get('success', False) else "❌"
            phase_display = phase_name.replace('_', ' ').title()
            logger.info(f"   {status} {phase_display}")

            if not result.get('success', False) and 'error' in result:
                logger.info(f"      Error: {result['error']}")

        # Overall assessment
        logger.info(f"\\n🎯 OVERALL ASSESSMENT:")

        if passed_phases == total_phases:
            logger.info("🎉 ALL PHASES PASSED!")
            logger.info("🚀 System fully validated and ready for production!")
            logger.info("💡 Proceed with confidence to achieve 60% F1-macro target!")
        elif passed_phases >= total_phases * 0.8:
            logger.info("📈 MOSTLY SUCCESSFUL!")
            logger.info("🔧 Minor issues detected - system largely functional")
            logger.info("💡 Address specific failures and proceed with optimization")
        elif passed_phases >= total_phases * 0.5:
            logger.info("⚠️ PARTIAL SUCCESS!")
            logger.info("🔍 Significant issues need attention")
            logger.info("💡 Fix critical problems before full optimization")
        else:
            logger.info("❌ MAJOR ISSUES DETECTED!")
            logger.info("🚨 Multiple critical failures")
            logger.info("💡 Comprehensive debugging required before proceeding")

        # Save detailed report
        report_file = self.outputs_dir / f"orchestration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_execution_time_hours': total_time / 3600,
                'total_phases': total_phases,
                'passed_phases': passed_phases,
                'success_rate': passed_phases / total_phases if total_phases > 0 else 0
            },
            'phase_results': self.test_results,
            'summary': {
                'all_passed': passed_phases == total_phases,
                'mostly_successful': passed_phases >= total_phases * 0.8,
                'ready_for_production': passed_phases == total_phases,
                'needs_debugging': passed_phases < total_phases * 0.5
            }
        }

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"\\n📄 Detailed report saved: {report_file}")

        return passed_phases == total_phases

def main():
    """Main orchestration function"""
    parser = argparse.ArgumentParser(description='Comprehensive Test Orchestration')
    parser.add_argument('--quick', action='store_true', help='Quick validation only (5-10 min)')
    parser.add_argument('--comprehensive', action='store_true', help='Comprehensive testing (30-60 min)')
    parser.add_argument('--benchmark', action='store_true', help='Performance benchmarking (60+ min)')
    parser.add_argument('--parallel-loss', action='store_true', help='Parallel loss function testing')
    parser.add_argument('--all', action='store_true', help='Run all test phases (2+ hours)')

    args = parser.parse_args()

    # Default to quick if no specific phase selected
    if not any([args.quick, args.comprehensive, args.benchmark, args.parallel_loss, args.all]):
        args.quick = True

    orchestrator = TestOrchestrator()

    logger.info("🧪 TEST ORCHESTRATION STARTING")
    logger.info("=" * 60)

    # Phase execution based on arguments
    overall_success = True

    if args.quick or args.all:
        success = orchestrator.run_quick_validation()
        overall_success = overall_success and success

        if not success and not args.all:
            logger.error("❌ Quick validation failed - stopping execution")
            return False

    if args.comprehensive or args.all:
        success = orchestrator.run_comprehensive_testing()
        overall_success = overall_success and success

        if not success and not args.all:
            logger.error("❌ Comprehensive testing failed - stopping execution")
            return False

    if args.parallel_loss or args.all:
        success = orchestrator.run_parallel_loss_testing()
        overall_success = overall_success and success

    if args.benchmark or args.all:
        success = orchestrator.run_performance_benchmark()
        overall_success = overall_success and success

    # Generate final report
    final_success = orchestrator.generate_final_report()

    if final_success:
        logger.info("\\n🎉 TEST ORCHESTRATION SUCCESSFUL!")
        logger.info("🚀 System validated and ready for optimization!")
    else:
        logger.info("\\n⚠️ TEST ORCHESTRATION COMPLETED WITH ISSUES")
        logger.info("🔧 Review report and address failures")

    return final_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)