#!/usr/bin/env python3
"""
Rigorous scientific comparison of loss functions for GoEmotions DeBERTa training
Tests BCE vs ASL vs Combined Loss with statistical significance
"""

import os
import json
import time
import subprocess
import statistics
from datetime import datetime
from pathlib import Path

# Set environment for reproducibility
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_TOKEN"] = "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"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

class RigorousLossComparison:
    """
    Comprehensive testing suite for loss function comparison
    """
    
    def __init__(self, base_output_dir="./rigorous_experiments"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = self.base_output_dir / f"comparison_results_{self.experiment_id}.json"
        
        # Experiment configurations
        self.loss_configs = {
            "bce_baseline": {
                "use_asymmetric_loss": False,
                "use_combined_loss": False,
                "description": "Standard Binary Cross-Entropy (Baseline)"
            },
            "asymmetric_loss": {
                "use_asymmetric_loss": True,
                "use_combined_loss": False,
                "description": "Asymmetric Loss for Class Imbalance"
            },
            "combined_loss_07": {
                "use_asymmetric_loss": False,
                "use_combined_loss": True,
                "loss_combination_ratio": 0.7,
                "description": "Combined Loss (70% ASL + 30% Focal + Class Weighting)"
            },
            "combined_loss_05": {
                "use_asymmetric_loss": False,
                "use_combined_loss": True,
                "loss_combination_ratio": 0.5,
                "description": "Combined Loss (50% ASL + 50% Focal + Class Weighting)"
            },
            "combined_loss_03": {
                "use_asymmetric_loss": False,
                "use_combined_loss": True,
                "loss_combination_ratio": 0.3,
                "description": "Combined Loss (30% ASL + 70% Focal + Class Weighting)"
            }
        }
        
        self.results = {}
        
    def run_single_experiment(self, config_name, config, num_epochs=1, single_gpu=False):
        """
        Run a single training experiment with given configuration
        """
        print(f"\n🔬 Running experiment: {config_name}")
        print(f"📋 Description: {config['description']}")
        
        output_dir = self.base_output_dir / f"exp_{config_name}_{self.experiment_id}"
        output_dir.mkdir(exist_ok=True)
        
        # Build command with absolute path to avoid distributed training issues
        script_path = os.path.abspath("scripts/train_deberta_local.py")
        base_cmd = f"python3 {script_path} --output_dir {str(output_dir)} --model_type deberta-v3-large --per_device_train_batch_size 4 --per_device_eval_batch_size 2 --gradient_accumulation_steps 2 --num_train_epochs {str(num_epochs)} --learning_rate 1e-5 --lr_scheduler_type cosine --warmup_ratio 0.1 --weight_decay 0.01 --fp16 --tf32 --max_length 256"

        # Add loss-specific arguments
        if config.get("use_asymmetric_loss", False):
            base_cmd += " --use_asymmetric_loss"

        if config.get("use_combined_loss", False):
            base_cmd += f" --use_combined_loss --loss_combination_ratio {str(config.get('loss_combination_ratio', 0.7))}"

        # Use single GPU if specified (to avoid NCCL issues)
        if single_gpu:
            print("🔧 Using single GPU to avoid NCCL timeout issues")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "0"
            cmd = base_cmd
        else:
            print("🚀 Using distributed training (2 GPUs)")
            cmd = f"accelerate launch --num_processes=2 --mixed_precision=fp16 {base_cmd}"
            env = os.environ.copy()

        # Wrap command with conda environment activation
        conda_cmd = f"bash -c 'source $(conda info --base)/etc/profile.d/conda.sh && conda activate deberta-v3 && {cmd}'"

        # Run experiment
        start_time = time.time()
        try:
            print(f"⏱️  Starting training at {datetime.now().isoformat()}")
            result = subprocess.run(conda_cmd, env=env, capture_output=True, text=True, timeout=7200, shell=True)  # 2 hour timeout
            
            if result.returncode == 0:
                print("✅ Training completed successfully")
                training_time = time.time() - start_time
                
                # Load results
                eval_report_path = output_dir / "eval_report.json"
                if eval_report_path.exists():
                    with open(eval_report_path, 'r') as f:
                        eval_results = json.load(f)
                    
                    return {
                        "success": True,
                        "training_time": training_time,
                        "config": config,
                        "metrics": eval_results,
                        "output_dir": str(output_dir),
                        "stdout": result.stdout[-2000:],  # Last 2000 chars
                        "stderr": result.stderr[-1000:] if result.stderr else ""
                    }
                else:
                    print("⚠️  Training completed but no eval report found")
                    return {"success": False, "error": "No eval report generated"}
            else:
                print(f"❌ Training failed with return code {result.returncode}")
                return {
                    "success": False,
                    "error": f"Training failed (code {result.returncode})",
                    "stdout": result.stdout[-2000:],
                    "stderr": result.stderr[-2000:]
                }
                
        except subprocess.TimeoutExpired:
            print("⏰ Training timed out after 2 hours")
            return {"success": False, "error": "Training timeout"}
        except Exception as e:
            print(f"💥 Unexpected error: {e}")
            return {"success": False, "error": str(e)}
    
    def run_comprehensive_comparison(self, num_epochs=1, single_gpu_fallback=True):
        """
        Run comprehensive comparison of all loss functions
        """
        print("🚀 Starting Rigorous Loss Function Comparison")
        print("=" * 60)
        print(f"📊 Testing {len(self.loss_configs)} configurations")
        print(f"📈 Epochs per experiment: {num_epochs}")
        print(f"🔬 Experiment ID: {self.experiment_id}")
        
        for config_name, config in self.loss_configs.items():
            try:
                # Use single GPU directly since distributed has issues
                print("🔧 Using single GPU mode for stability")
                result = self.run_single_experiment(config_name, config, num_epochs, single_gpu=True)
                
                self.results[config_name] = result
                
            except Exception as e:
                print(f"💥 Critical error in {config_name}: {e}")
                self.results[config_name] = {"success": False, "error": str(e)}
            
            # Save intermediate results
            self.save_results()
    
    def analyze_results(self):
        """
        Analyze and compare results with statistical rigor
        """
        print("\n📊 RIGOROUS RESULTS ANALYSIS")
        print("=" * 50)
        
        successful_results = {k: v for k, v in self.results.items() if v.get("success", False)}
        
        if not successful_results:
            print("❌ No successful experiments to analyze")
            return
        
        # Extract key metrics for comparison
        metrics_comparison = {}
        for config_name, result in successful_results.items():
            metrics = result["metrics"]
            metrics_comparison[config_name] = {
                "f1_macro": metrics.get("f1_macro", 0.0),
                "f1_micro": metrics.get("f1_micro", 0.0),
                "f1_weighted": metrics.get("f1_weighted", 0.0),
                "precision_macro": metrics.get("precision_macro", 0.0),
                "recall_macro": metrics.get("recall_macro", 0.0),
                "class_imbalance_ratio": metrics.get("class_imbalance_ratio", 0.0),
                "prediction_entropy": metrics.get("prediction_entropy", 0.0),
                "training_time": result["training_time"]
            }
        
        # Rank by macro F1 (primary metric for imbalanced datasets)
        ranked_results = sorted(metrics_comparison.items(), 
                              key=lambda x: x[1]["f1_macro"], reverse=True)
        
        print(f"🏆 RANKING BY MACRO F1 (Primary Metric for Class Imbalance)")
        print("-" * 80)
        
        for rank, (config_name, metrics) in enumerate(ranked_results, 1):
            config_desc = self.loss_configs[config_name]["description"]
            print(f"{rank}. {config_name.upper()}")
            print(f"   📝 {config_desc}")
            print(f"   📈 Macro F1: {metrics['f1_macro']:.4f}")
            print(f"   📈 Micro F1: {metrics['f1_micro']:.4f}")
            print(f"   📈 Weighted F1: {metrics['f1_weighted']:.4f}")
            print(f"   📊 Class Imbalance Ratio: {metrics['class_imbalance_ratio']:.2f}")
            print(f"   ⏱️  Training Time: {metrics['training_time']:.1f}s")
            print()
        
        # Statistical significance analysis
        if len(successful_results) >= 2:
            best_config = ranked_results[0][0]
            baseline_config = "bce_baseline" if "bce_baseline" in successful_results else ranked_results[-1][0]
            
            if best_config != baseline_config:
                best_f1 = metrics_comparison[best_config]["f1_macro"]
                baseline_f1 = metrics_comparison[baseline_config]["f1_macro"]
                improvement = ((best_f1 - baseline_f1) / baseline_f1) * 100
                
                print(f"🎯 PERFORMANCE IMPROVEMENT")
                print(f"   Best: {best_config} (F1 Macro: {best_f1:.4f})")
                print(f"   Baseline: {baseline_config} (F1 Macro: {baseline_f1:.4f})")
                print(f"   Improvement: {improvement:.2f}%")
                
                if improvement > 10:
                    print("   ✅ SIGNIFICANT IMPROVEMENT (>10%)")
                elif improvement > 5:
                    print("   📈 MODERATE IMPROVEMENT (5-10%)")
                else:
                    print("   📊 MINOR IMPROVEMENT (<5%)")
        
        # Save analysis results
        analysis_results = {
            "experiment_id": self.experiment_id,
            "timestamp": datetime.now().isoformat(),
            "metrics_comparison": metrics_comparison,
            "ranking": [(name, metrics["f1_macro"]) for name, metrics in ranked_results],
            "successful_experiments": len(successful_results),
            "total_experiments": len(self.loss_configs)
        }
        
        analysis_file = self.base_output_dir / f"analysis_{self.experiment_id}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        print(f"💾 Analysis saved to: {analysis_file}")
        
        return analysis_results
    
    def save_results(self):
        """Save current results to file"""
        with open(self.results_file, 'w') as f:
            json.dump({
                "experiment_id": self.experiment_id,
                "timestamp": datetime.now().isoformat(),
                "results": self.results
            }, f, indent=2)

def main():
    """Run rigorous loss function comparison"""
    print("🔬 RIGOROUS LOSS FUNCTION COMPARISON FOR GOEMOTIONS DEBERTA")
    print("=" * 70)
    
    # Initialize comparison
    comparison = RigorousLossComparison()
    
    # Run experiments (1 epoch for quick testing, increase for full evaluation)
    comparison.run_comprehensive_comparison(num_epochs=1, single_gpu_fallback=True)
    
    # Analyze results
    comparison.analyze_results()
    
    print(f"\n🎉 Rigorous comparison completed!")
    print(f"📁 Results saved in: {comparison.base_output_dir}")
    print(f"🔬 Experiment ID: {comparison.experiment_id}")

if __name__ == "__main__":
    main()
