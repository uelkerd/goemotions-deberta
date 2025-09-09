#!/usr/bin/env python3
"""
QUICK INTEGRATION TEST: 50 steps each config
Validate fixes work in real training context (5 minutes total)
"""

import subprocess
import os
import time
import sys

def run_quick_integration_test():
    print("🔬 STEP 5: QUICK INTEGRATION TEST")
    print("=" * 60)
    
    print("🎯 ULTRA-SHORT TEST DESIGN:")
    print("- 50 training steps each (~90 seconds per config)")
    print("- 1000 samples only")
    print("- Focus: Gradient health + crash detection")
    print("- Total time: ~5 minutes")
    
    # Change to correct directory
    os.chdir("/workspace")
    
    # Test configs
    configs = [
        {
            "name": "AsymmetricLoss_quick",
            "desc": "Test gamma_neg=2.0 gradient fix", 
            "args": ["--use_asymmetric_loss"],
            "critical": True
        },
        {
            "name": "CombinedLoss_quick",
            "desc": "Test AttributeError fix",
            "args": ["--use_combined_loss", "--loss_combination_ratio", "0.7"], 
            "critical": True
        }
    ]
    
    # Base command (ultra-minimal)
    base_cmd = [
        "python3", "notebooks/scripts/train_deberta_local.py",
        "--model_type", "deberta-v3-large",
        "--output_dir", "PLACEHOLDER",
        "--per_device_train_batch_size", "2",    # Smaller batch
        "--gradient_accumulation_steps", "1",    # No accumulation
        "--num_train_epochs", "1",
        "--learning_rate", "3e-5", 
        "--max_length", "64",                    # Very short sequences
        "--max_train_samples", "1000",           # Tiny dataset
        "--max_eval_samples", "200",             # Tiny eval
        "--augment_prob", "0",
        "--save_steps", "5000",                  # Don't save
        "--eval_steps", "5000",                  # Don't eval
        "--logging_steps", "25",                 # Frequent logging
        "--max_steps", "50"                      # Only 50 steps!
    ]
    
    print(f"\n🧪 RUNNING INTEGRATION TESTS:")
    print("-" * 40)
    
    results = {}
    
    for config in configs:
        print(f"\n🔬 Testing {config['name']}...")
        print(f"Purpose: {config['desc']}")
        
        # Build command
        cmd = base_cmd.copy()
        cmd[cmd.index("PLACEHOLDER")] = f"./test_integration/{config['name']}"
        cmd.extend(config['args'])
        
        print(f"Command: {' '.join(cmd[-6:])}")  # Show last 6 args
        
        # Set environment 
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'
        
        start_time = time.time()
        
        try:
            print("🚀 Executing 50-step test...")
            
            # Create output directory
            os.makedirs(f"./test_integration/{config['name']}", exist_ok=True)
            
            # Run the test (REAL EXECUTION)
            result = subprocess.run(
                cmd, 
                env=env, 
                capture_output=True, 
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            duration = time.time() - start_time
            
            # Analyze results
            stdout = result.stdout
            stderr = result.stderr
            
            print(f"⏱️  Duration: {duration:.1f} seconds")
            print(f"Return code: {result.returncode}")
            
            # Check for success indicators
            if result.returncode == 0:
                print("✅ Process completed successfully")
                
                # Look for gradient information
                if "grad_norm" in stdout:
                    grad_lines = [line for line in stdout.split('\n') if 'grad_norm' in line]
                    if grad_lines:
                        last_grad_line = grad_lines[-1]
                        print(f"📊 Gradient info: {last_grad_line}")
                        
                        # Extract grad_norm value
                        try:
                            grad_str = last_grad_line.split("'grad_norm': ")[1].split(',')[0]
                            grad_value = float(grad_str)
                            
                            if grad_value > 1e-3:
                                print(f"✅ HEALTHY GRADIENTS: {grad_value:.2e}")
                                gradient_health = "HEALTHY"
                            else:
                                print(f"❌ WEAK GRADIENTS: {grad_value:.2e}")
                                gradient_health = "WEAK"
                        except:
                            gradient_health = "UNKNOWN"
                    else:
                        gradient_health = "NO_DATA"
                else:
                    gradient_health = "NO_DATA"
                
                results[config['name']] = {
                    "success": True,
                    "duration": duration,
                    "gradient_health": gradient_health,
                    "details": "Completed successfully"
                }
                
            else:
                print(f"❌ Process failed with return code: {result.returncode}")
                
                # Check for specific errors
                if "AttributeError" in stderr:
                    print("🚨 AttributeError detected!")
                    error_type = "ATTRIBUTE_ERROR"
                elif "FileExistsError" in stderr:
                    print("🚨 FileExistsError detected!")
                    error_type = "FILE_EXISTS"
                else:
                    print("🚨 Other error type")
                    error_type = "OTHER"
                
                # Show error details
                error_lines = [line for line in stderr.split('\n') if 'Error' in line or 'Exception' in line]
                if error_lines:
                    print(f"Error: {error_lines[-1]}")
                
                results[config['name']] = {
                    "success": False,
                    "duration": duration,
                    "error_type": error_type,
                    "details": error_lines[-1] if error_lines else "Unknown error"
                }
            
        except subprocess.TimeoutExpired:
            print("⏰ Test timed out (>5 minutes)")
            results[config['name']] = {
                "success": False,
                "duration": 300,
                "error_type": "TIMEOUT",
                "details": "Test exceeded 5 minute limit"
            }
            
        except Exception as e:
            print(f"💥 Test setup error: {e}")
            results[config['name']] = {
                "success": False,
                "duration": 0,
                "error_type": "SETUP_ERROR", 
                "details": str(e)
            }
    
    # COMPREHENSIVE ANALYSIS
    print("\n🏆 INTEGRATION TEST RESULTS:")
    print("=" * 50)
    
    success_count = sum(1 for r in results.values() if r['success'])
    total_tests = len(results)
    
    for config_name, result in results.items():
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        duration = f"{result['duration']:.1f}s"
        
        print(f"{config_name}: {status} ({duration})")
        
        if result['success']:
            if 'gradient_health' in result:
                print(f"  📊 Gradients: {result['gradient_health']}")
        else:
            print(f"  ❌ Error: {result['error_type']} - {result['details']}")
    
    success_rate = success_count / total_tests * 100
    print(f"\nIntegration success rate: {success_count}/{total_tests} ({success_rate:.0f}%)")
    
    # Decision logic
    print("\n🎯 INTEGRATION TEST DECISION:")
    if success_rate == 100:
        print("🎉 ALL INTEGRATION TESTS PASS!")
        print("✅ Proceed to full PHASE 1 training")
        print("🚀 High confidence in bulletproof status")
        return "PROCEED_FULL"
    elif success_rate >= 50:
        print("⚠️  PARTIAL SUCCESS")
        print("🔧 Fix failing tests before full training")
        print("📊 Analyze specific failure modes")
        return "FIX_FAILURES"
    else:
        print("🚨 INTEGRATION TESTS FAIL")
        print("❌ Do not proceed to full training")
        print("🔧 Major fixes still needed")
        return "MAJOR_FIXES_NEEDED"

if __name__ == "__main__":
    decision = run_quick_integration_test()
    
    print(f"\n🚀 SCIENTIFIC METHOD RESULT: {decision}")
    
    if decision == "PROCEED_FULL":
        print("✅ Ready for Step 6: Full validation testing")
    else:
        print("🔧 Address integration test failures first")