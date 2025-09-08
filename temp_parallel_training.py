import subprocess
import threading
import os
import time
import queue
from datetime import datetime

def run_config_live(gpu_id, config_name, use_asym=False, ratio=None):
    """Run training on specific GPU with live monitoring"""
    print(f"🚀 LIVE MONITOR: Starting {config_name} on GPU {gpu_id} at {datetime.now()}")
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Fixed command with quota safety
    cmd = [
        'python3', 'notebooks/scripts/train_deberta_local.py',
        '--output_dir', f'./outputs/parallel_{config_name}',
        '--model_type', 'deberta-v3-large',
        '--per_device_train_batch_size', '4',
        '--per_device_eval_batch_size', '8',
        '--gradient_accumulation_steps', '4',
        '--num_train_epochs', '2',
        '--learning_rate', '3e-5',
        '--lr_scheduler_type', 'cosine',
        '--warmup_ratio', '0.15',
        '--weight_decay', '0.01',
        '--fp16',
        '--max_length', '256',
        '--max_train_samples', '20000',
        '--max_eval_samples', '3000',
        '--save_total_limit', '1'
    ]
    if use_asym: 
        cmd += ['--use_asymmetric_loss']
    if ratio is not None: 
        cmd += ['--use_combined_loss', '--loss_combination_ratio', str(ratio)]
    
    print(f"Command for {config_name}: {' '.join(cmd)}")
    
    # Non-blocking Popen for live output
    try:
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                cwd='/home/user/goemotions-deberta', universal_newlines=True, bufsize=1)
    except Exception as e:
        print(f"ERROR starting {config_name} on GPU {gpu_id}: {e}")
        return 1
    
    output_queue = queue.Queue()
    
    def read_output():
        """Thread to read stdout line-by-line and queue/print live"""
        for line in iter(proc.stdout.readline, ''):
            output_queue.put(line)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] GPU {gpu_id} [{config_name}]: {line.strip()}")
    
    output_thread = threading.Thread(target=read_output, daemon=True)
    output_thread.start()
    
    def monitor_live():
        """Live monitoring: GPU status every 30s while running"""
        while proc.poll() is None:
            time.sleep(30)
            try:
                gpu_result = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu', 
                                             '--format=csv,noheader,nounits'], capture_output=True, text=True)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 🖥️ GPU STATUS (overall): {gpu_result.stdout.strip()}")
            except Exception as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 🖥️ GPU STATUS ERROR: {e}")
    
    monitor_thread = threading.Thread(target=monitor_live, daemon=True)
    monitor_thread.start()
    
    # Wait for completion
    return_code = proc.wait()
    output_thread.join(timeout=5)
    monitor_thread.join(timeout=5)
    
    # Drain queue and print final lines
    final_lines = []
    try:
        while True:
            line = output_queue.get_nowait()
            final_lines.append(line)
    except queue.Empty:
        pass
    if final_lines:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📋 FINAL LINES ({config_name}, last 10):")
        for line in final_lines[-10:]:
            print(f"    {line.strip()}")
    
    print(f"✅ {config_name} complete on GPU {gpu_id} (return code: {return_code}) at {datetime.now()}")
    return return_code

# Kill any existing processes
subprocess.run(['pkill', '-f', 'train_deberta_local'], capture_output=True)
time.sleep(2)

print("🚀 PHASE 1.5: Parallel Dual-GPU Training - 5 Configs WITH LIVE MONITORING (Fixed Args)")
print("=" * 70)
print("GPU 0 & 1 concurrent: 50% faster than sequential")
print("Fixes maintained: pos_weight, oversampling, threshold=0.2")
print("LIVE: Loss/grad_norm/f1_macro/epoch printed in real-time + GPU status every 30s")
print("FIXED: Added --save_total_limit 1 for quota safety")
print("=" * 70)

# FULL PAIR 1: BCE (GPU0) + Asymmetric (GPU1) with live monitoring
print("\n📍 PAIR 1 LIVE: BCE (GPU0) + Asymmetric (GPU1)")
t1 = threading.Thread(target=run_config_live, args=(0, 'BCE_Parallel'))
t2 = threading.Thread(target=run_config_live, args=(1, 'Asymmetric_Parallel', True))
t1.start()
t2.start()
t1.join()
t2.join()

# FULL PAIR 2: Combined 0.7 (GPU0) + 0.5 (GPU1) with live monitoring
print("\n📍 PAIR 2 LIVE: Combined 0.7 (GPU0) + 0.5 (GPU1)")
t3 = threading.Thread(target=run_config_live, args=(0, 'Combined_07_Parallel', False, 0.7))
t4 = threading.Thread(target=run_config_live, args=(1, 'Combined_05_Parallel', False, 0.5))
t3.start()
t4.start()
t3.join()
t4.join()

# FULL SINGLE: Combined 0.3 (GPU0) with live monitoring
print("\n📍 SINGLE LIVE: Combined 0.3 (GPU0)")
run_config_live(0, 'Combined_03_Parallel', False, 0.3)

print("\n🎉 PHASE 1.5 PARALLEL COMPLETE with FULL LIVE VERBOSE MONITORING!")
print("📊 Outputs: ./outputs/parallel_BCE_Parallel/, ./outputs/parallel_Asymmetric_Parallel/, etc.")
print("🔍 All training output printed live above - check eval_report.json for F1@0.2 comparison vs baseline 42.18% (target >50%)")
print("⏱️ Total time: ~1.5 hours with dual-GPU parallel execution")
print("🖥️ GPU monitoring: Utilization/memory/temp shown every 30s during runs")