#!/usr/bin/env python3
"""
🔍 COMPREHENSIVE VALIDATION & MONITORING SYSTEM
===============================================
Advanced validation steps and monitoring features for multi-dataset training
Includes debugging, stall detection, and scientific validation
"""

import os
import sys
import json
import time
import psutil
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

class TrainingMonitor:
    """Comprehensive training monitoring system"""

    def __init__(self):
        self.start_time = time.time()
        self.last_check_time = time.time()
        self.stall_threshold = 600  # 10 minutes
        self.monitoring_active = False

    def check_gpu_health(self):
        """Check GPU status and utilization"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu',
                                   '--format=csv,noheader,nounits'], capture_output=True, text=True)

            if result.returncode == 0:
                gpu_info = []
                for line in result.stdout.strip().split('\\n'):
                    parts = line.split(', ')
                    if len(parts) >= 6:
                        gpu_info.append({
                            'index': parts[0],
                            'name': parts[1],
                            'utilization': int(parts[2]) if parts[2] != '[Not Supported]' else 0,
                            'memory_used': int(parts[3]),
                            'memory_total': int(parts[4]),
                            'temperature': int(parts[5]) if parts[5] != '[Not Supported]' else 0
                        })

                return gpu_info
            else:
                return []

        except Exception as e:
            print(f"⚠️ GPU health check failed: {e}")
            return []

    def check_training_processes(self):
        """Check for active training processes"""
        training_processes = []

        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'cpu_percent']):
            try:
                cmdline = proc.info['cmdline']
                if (cmdline and 'python' in proc.info['name'] and
                    any('train_deberta_local.py' in arg for arg in cmdline)):

                    training_processes.append({
                        'pid': proc.info['pid'],
                        'cmdline': ' '.join(cmdline),
                        'cpu_percent': proc.info['cpu_percent'],
                        'running_time': time.time() - proc.info['create_time']
                    })

            except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
                continue

        return training_processes

    def check_disk_space(self):
        """Monitor disk space usage"""
        try:
            disk_usage = psutil.disk_usage('/')
            free_gb = disk_usage.free / (1024 ** 3)
            used_percent = (disk_usage.used / disk_usage.total) * 100

            return {
                'free_gb': free_gb,
                'used_percent': used_percent,
                'warning': free_gb < 10 or used_percent > 85
            }

        except Exception as e:
            print(f"⚠️ Disk space check failed: {e}")
            return {'free_gb': 0, 'used_percent': 100, 'warning': True}

    def analyze_training_logs(self, log_path):
        """Analyze training logs for issues"""
        if not os.path.exists(log_path):
            return {"status": "no_logs", "issues": ["Log file not found"]}

        try:
            with open(log_path, 'r') as f:
                lines = f.readlines()

            # Check for common issues
            issues = []
            warnings = []
            errors = []

            for line in lines[-100:]:  # Check last 100 lines
                line_lower = line.lower()

                if 'error' in line_lower or 'exception' in line_lower:
                    errors.append(line.strip())

                if 'warning' in line_lower or 'warn' in line_lower:
                    warnings.append(line.strip())

                if any(keyword in line_lower for keyword in ['nan', 'inf', 'gradient explosion']):
                    issues.append(f"Numerical instability: {line.strip()}")

                if 'cuda' in line_lower and ('memory' in line_lower or 'out of memory' in line_lower):
                    issues.append(f"GPU memory issue: {line.strip()}")

            return {
                "status": "analyzed",
                "total_lines": len(lines),
                "errors": errors[-5:],  # Last 5 errors
                "warnings": warnings[-3:],  # Last 3 warnings
                "issues": issues,
                "last_modified": os.path.getmtime(log_path)
            }

        except Exception as e:
            return {"status": "error", "issues": [f"Log analysis failed: {e}"]}

    def check_training_progress(self, output_dir):
        """Check training progress from checkpoints and logs"""
        progress_info = {
            "checkpoints": [],
            "latest_metrics": {},
            "stalled": False,
            "estimated_completion": None
        }

        # Check checkpoints
        checkpoint_pattern = f"{output_dir}/checkpoint-*"
        checkpoints = list(Path(output_dir).glob("checkpoint-*"))

        if checkpoints:
            # Sort by step number
            checkpoints.sort(key=lambda x: int(x.name.split('-')[1]))
            latest_checkpoint = checkpoints[-1]

            progress_info["checkpoints"] = [cp.name for cp in checkpoints[-3:]]  # Last 3

            # Check for recent activity
            checkpoint_time = os.path.getmtime(latest_checkpoint)
            time_since_last = time.time() - checkpoint_time

            if time_since_last > self.stall_threshold:
                progress_info["stalled"] = True
                progress_info["stall_duration"] = time_since_last

        # Check evaluation results
        eval_file = f"{output_dir}/eval_report.json"
        if os.path.exists(eval_file):
            try:
                with open(eval_file, 'r') as f:
                    eval_results = json.load(f)

                progress_info["latest_metrics"] = {
                    "f1_macro": eval_results.get('f1_macro', 0.0),
                    "f1_micro": eval_results.get('f1_micro', 0.0),
                    "eval_loss": eval_results.get('eval_loss', 0.0),
                    "timestamp": os.path.getmtime(eval_file)
                }

            except Exception as e:
                progress_info["metrics_error"] = str(e)

        return progress_info

    def generate_status_report(self):
        """Generate comprehensive status report"""
        print("🔍 COMPREHENSIVE TRAINING STATUS REPORT")
        print("=" * 55)

        # System health
        gpu_info = self.check_gpu_health()
        disk_info = self.check_disk_space()
        processes = self.check_training_processes()

        print("\\n🖥️ SYSTEM HEALTH:")
        print("-" * 20)

        if gpu_info:
            for gpu in gpu_info:
                print(f"   GPU {gpu['index']} ({gpu['name']}):")
                print(f"      Utilization: {gpu['utilization']}%")
                print(f"      Memory: {gpu['memory_used']}/{gpu['memory_total']} MB")
                print(f"      Temperature: {gpu['temperature']}°C")
        else:
            print("   ⚠️ No GPU information available")

        print(f"\\n💾 DISK SPACE:")
        print(f"   Free: {disk_info['free_gb']:.1f} GB")
        print(f"   Used: {disk_info['used_percent']:.1f}%")
        if disk_info['warning']:
            print("   ⚠️ WARNING: Low disk space!")

        print(f"\\n🔄 TRAINING PROCESSES:")
        if processes:
            for proc in processes:
                hours = proc['running_time'] / 3600
                print(f"   PID {proc['pid']}: Running {hours:.1f}h, CPU {proc['cpu_percent']:.1f}%")
        else:
            print("   No active training processes")

        # Training progress for different experiments
        experiment_dirs = [
            "checkpoints_comprehensive_multidataset",
            "outputs/phase1_multidataset_bce",
            "outputs/phase1_multidataset_asymmetric",
            "outputs/phase2_best_extended"
        ]

        print("\\n📊 TRAINING PROGRESS:")
        print("-" * 25)

        for exp_dir in experiment_dirs:
            if os.path.exists(exp_dir):
                progress = self.check_training_progress(exp_dir)
                exp_name = os.path.basename(exp_dir)

                print(f"\\n   {exp_name.upper()}:")

                if progress["checkpoints"]:
                    print(f"      Checkpoints: {', '.join(progress['checkpoints'])}")
                else:
                    print("      No checkpoints yet")

                if progress["latest_metrics"]:
                    metrics = progress["latest_metrics"]
                    print(f"      F1-macro: {metrics['f1_macro']:.4f}")
                    print(f"      F1-micro: {metrics['f1_micro']:.4f}")

                if progress["stalled"]:
                    stall_hours = progress["stall_duration"] / 3600
                    print(f"      ⚠️ STALLED: {stall_hours:.1f}h since last update")

        # Log analysis
        log_files = [
            "logs/train_comprehensive_multidataset.log",
            "logs/phases/bce_training.log",
            "logs/phases/asymmetric_training.log"
        ]

        print("\\n📝 LOG ANALYSIS:")
        print("-" * 20)

        for log_file in log_files:
            if os.path.exists(log_file):
                analysis = self.analyze_training_logs(log_file)
                log_name = os.path.basename(log_file)

                print(f"\\n   {log_name}:")
                print(f"      Lines: {analysis.get('total_lines', 0)}")

                if analysis.get('errors'):
                    print(f"      Recent errors: {len(analysis['errors'])}")

                if analysis.get('issues'):
                    print(f"      ⚠️ Issues detected: {len(analysis['issues'])}")

        return {
            "timestamp": datetime.now().isoformat(),
            "system": {"gpu": gpu_info, "disk": disk_info},
            "processes": processes,
            "experiments": {exp: self.check_training_progress(exp) for exp in experiment_dirs if os.path.exists(exp)}
        }

class ValidationSuite:
    """Scientific validation and testing suite"""

    def __init__(self):
        self.baseline_f1 = 0.5179

    def validate_dataset_quality(self):
        """Validate dataset quality and distribution"""
        print("\\n🔬 DATASET QUALITY VALIDATION")
        print("=" * 40)

        datasets_to_check = [
            ("data/goemotions/train.jsonl", "GoEmotions Train"),
            ("data/goemotions/val.jsonl", "GoEmotions Val"),
            ("data/combined_all_datasets/train.jsonl", "Combined Train"),
            ("data/combined_all_datasets/val.jsonl", "Combined Val")
        ]

        validation_results = {}

        for dataset_path, dataset_name in datasets_to_check:
            if not os.path.exists(dataset_path):
                print(f"   ❌ {dataset_name}: File not found")
                validation_results[dataset_name] = {"status": "missing"}
                continue

            try:
                # Load and analyze dataset
                samples = []
                with open(dataset_path, 'r') as f:
                    for line in f:
                        try:
                            sample = json.loads(line)
                            samples.append(sample)
                        except:
                            continue

                if not samples:
                    print(f"   ❌ {dataset_name}: No valid samples")
                    validation_results[dataset_name] = {"status": "empty"}
                    continue

                # Analyze distribution
                label_counts = {}
                text_lengths = []
                sources = {}

                for sample in samples:
                    labels = sample.get('labels', [])
                    if isinstance(labels, int):
                        labels = [labels]

                    for label in labels:
                        if isinstance(label, int) and 0 <= label < 28:
                            label_counts[label] = label_counts.get(label, 0) + 1

                    text = sample.get('text', '')
                    text_lengths.append(len(text.split()))

                    source = sample.get('source', 'unknown')
                    sources[source] = sources.get(source, 0) + 1

                # Calculate statistics
                total_samples = len(samples)
                unique_emotions = len(label_counts)
                avg_text_length = np.mean(text_lengths) if text_lengths else 0
                max_count = max(label_counts.values()) if label_counts else 1
                min_count = min(label_counts.values()) if label_counts else 1
                imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

                validation_results[dataset_name] = {
                    "status": "valid",
                    "samples": total_samples,
                    "unique_emotions": unique_emotions,
                    "avg_text_length": avg_text_length,
                    "imbalance_ratio": imbalance_ratio,
                    "sources": sources
                }

                print(f"   ✅ {dataset_name}:")
                print(f"      Samples: {total_samples}")
                print(f"      Emotions: {unique_emotions}/28")
                print(f"      Avg text length: {avg_text_length:.1f} words")
                print(f"      Imbalance ratio: {imbalance_ratio:.1f}:1")

                if imbalance_ratio > 100:
                    print(f"      ⚠️ High imbalance detected")

                if sources and len(sources) > 1:
                    print(f"      Sources: {list(sources.keys())}")

            except Exception as e:
                print(f"   ❌ {dataset_name}: Validation error: {e}")
                validation_results[dataset_name] = {"status": "error", "error": str(e)}

        return validation_results

    def validate_model_outputs(self, results_dir):
        """Validate model outputs and performance"""
        print(f"\\n🎯 MODEL PERFORMANCE VALIDATION: {results_dir}")
        print("=" * 50)

        if not os.path.exists(results_dir):
            print("   ❌ Results directory not found")
            return {"status": "missing"}

        # Check for evaluation results
        eval_file = f"{results_dir}/eval_report.json"
        if not os.path.exists(eval_file):
            print("   ⏳ No evaluation results yet")
            return {"status": "pending"}

        try:
            with open(eval_file, 'r') as f:
                results = json.load(f)

            f1_macro = results.get('f1_macro', 0.0)
            f1_micro = results.get('f1_micro', 0.0)
            f1_weighted = results.get('f1_weighted', 0.0)

            print(f"   📊 PERFORMANCE METRICS:")
            print(f"      F1-macro: {f1_macro:.4f}")
            print(f"      F1-micro: {f1_micro:.4f}")
            print(f"      F1-weighted: {f1_weighted:.4f}")

            # Validate against baseline
            improvement = ((f1_macro - self.baseline_f1) / self.baseline_f1) * 100

            print(f"\\n   🎯 BASELINE COMPARISON:")
            print(f"      Baseline: {self.baseline_f1:.4f}")
            print(f"      Current: {f1_macro:.4f}")
            print(f"      Improvement: {improvement:+.1f}%")

            # Success criteria
            success_criteria = {
                "beats_baseline": f1_macro > self.baseline_f1,
                "reaches_55_percent": f1_macro >= 0.55,
                "reaches_60_percent": f1_macro >= 0.60,
                "reasonable_micro": f1_micro >= 0.50
            }

            print(f"\\n   ✅ SUCCESS CRITERIA:")
            for criterion, achieved in success_criteria.items():
                status = "✅" if achieved else "❌"
                print(f"      {criterion.replace('_', ' ').title()}: {status}")

            return {
                "status": "validated",
                "metrics": {"f1_macro": f1_macro, "f1_micro": f1_micro, "f1_weighted": f1_weighted},
                "improvement": improvement,
                "success_criteria": success_criteria,
                "overall_success": f1_macro >= 0.55  # Minimum success threshold
            }

        except Exception as e:
            print(f"   ❌ Validation error: {e}")
            return {"status": "error", "error": str(e)}

def main():
    """Main monitoring and validation execution"""
    print("🔍 COMPREHENSIVE VALIDATION & MONITORING SYSTEM")
    print("=" * 60)
    print("🎯 Purpose: Ensure scientific rigor and robust training")
    print("🔬 Features: System monitoring + Dataset validation + Performance analysis")
    print("=" * 60)

    # Initialize systems
    monitor = TrainingMonitor()
    validator = ValidationSuite()

    # Generate comprehensive status report
    status_report = monitor.generate_status_report()

    # Validate datasets
    dataset_validation = validator.validate_dataset_quality()

    # Validate model outputs (if available)
    model_validations = {}
    result_dirs = [
        "checkpoints_comprehensive_multidataset",
        "outputs/phase1_multidataset_bce",
        "outputs/phase2_best_extended"
    ]

    for result_dir in result_dirs:
        if os.path.exists(result_dir):
            model_validations[result_dir] = validator.validate_model_outputs(result_dir)

    # Save comprehensive report
    comprehensive_report = {
        "timestamp": datetime.now().isoformat(),
        "system_status": status_report,
        "dataset_validation": dataset_validation,
        "model_validations": model_validations
    }

    with open("comprehensive_validation_report.json", "w") as f:
        json.dump(comprehensive_report, f, indent=2)

    print("\\n📄 Comprehensive report saved: comprehensive_validation_report.json")

    # Final recommendations
    print("\\n🎯 MONITORING RECOMMENDATIONS:")
    print("   1. Run this script every 30 minutes during training")
    print("   2. Check for stalled processes and GPU utilization")
    print("   3. Monitor disk space and clean up old checkpoints")
    print("   4. Validate model performance against baseline regularly")

    print("\\n✅ VALIDATION & MONITORING COMPLETE!")
    return True

if __name__ == "__main__":
    main()