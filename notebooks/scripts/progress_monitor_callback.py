"""
Progress Monitor Callback for Transformers training
Detects stalls in training and checks disk space
"""

import time
import shutil
import traceback
from datetime import datetime
from transformers import TrainerCallback, TrainerState, TrainerControl

class ProgressMonitorCallback(TrainerCallback):
    """Monitors progress and detects stalls in training"""
    
    def __init__(self, stall_timeout=600, disk_quota_check=True, min_free_space_gb=10):
        """Initialize ProgressMonitor
        
        Args:
            stall_timeout: Seconds without progress before considering training stalled
            disk_quota_check: Whether to check disk space
            min_free_space_gb: Minimum free disk space in GB before warning
        """
        self.last_step = 0
        self.last_progress_time = time.time()
        self.stall_timeout = stall_timeout
        self.disk_quota_check = disk_quota_check
        self.min_free_space_gb = min_free_space_gb
        self.check_disk_space()
    
    def check_disk_space(self):
        """Check available disk space"""
        if not self.disk_quota_check:
            return True
        
        try:
            # Get disk usage of current directory
            disk = shutil.disk_usage('.')
            free_gb = disk.free / (1024 ** 3)
            total_gb = disk.total / (1024 ** 3)
            used_percent = (disk.used / disk.total) * 100
            
            print(f"💾 Disk space: {free_gb:.1f}GB free / {total_gb:.1f}GB total ({used_percent:.1f}% used)")
            
            if free_gb < self.min_free_space_gb:
                print(f"⚠️ LOW DISK SPACE WARNING: Only {free_gb:.1f}GB free. Training may fail with disk quota error.")
                if used_percent > 85:
                    print("🔥 CRITICAL: Disk usage above 85%. Try to free up space immediately!")
                return False
                
            return True
        except Exception as e:
            print(f"⚠️ Error checking disk space: {e}")
            return True
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called after each step"""
        # Update progress tracker if step increased
        if state.global_step > self.last_step:
            self.last_step = state.global_step
            self.last_progress_time = time.time()
        
        # Check if we're stalled
        time_since_progress = time.time() - self.last_progress_time
        if time_since_progress > self.stall_timeout:
            print(f"⚠️ WARNING: No progress for {time_since_progress:.1f} seconds (stall detected)")
            print(f"🔍 Last progress at step {self.last_step}")
            
            # Check disk space on stall detection
            if not self.check_disk_space():
                print("❌ TRAINING STOPPED: Disk space critical. Clean up files before continuing.")
                # We cannot set control.should_training_stop because it's read-only
                # Instead we'll raise an exception to stop training
                raise RuntimeError("Training stopped due to disk quota issue")
        
        # Check disk space every 100 steps
        if state.global_step % 100 == 0 and self.disk_quota_check:
            self.check_disk_space()
        
        return control
        
def check_system_status():
    """Check system status and resources"""
    print("\n🔍 Checking system status...")
    
    # Check disk space
    try:
        disk = shutil.disk_usage('.')
        free_gb = disk.free / (1024 ** 3)
        total_gb = disk.total / (1024 ** 3)
        used_percent = (disk.used / disk.total) * 100
        print(f"💾 Disk space: {free_gb:.1f}GB free / {total_gb:.1f}GB total ({used_percent:.1f}% used)")
        
        if free_gb < 10:
            print(f"⚠️ LOW DISK SPACE WARNING: Only {free_gb:.1f}GB free")
        if used_percent > 85:
            print("🔥 CRITICAL: Disk usage above 85%. Training may fail!")
    except Exception as e:
        print(f"⚠️ Error checking disk space: {e}")
    
    # Check GPU status
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"   Memory: {torch.cuda.memory_allocated(i)/1024**2:.1f}MB allocated, "
                      f"{torch.cuda.memory_reserved(i)/1024**2:.1f}MB reserved")
        else:
            print("❌ No GPUs available")
    except Exception as e:
        print(f"⚠️ Error checking GPU status: {e}")
    
    print("✅ System check complete\n")

def handle_exception(e, scientific_logger=None):
    """Handle exceptions with comprehensive error logging"""
    error_msg = f"❌ ERROR: {str(e)}"
    print(error_msg)
    print("Stack trace:")
    traceback_str = traceback.format_exc()
    print(traceback_str)
    
    # Log to scientific logger if available
    if scientific_logger:
        error_log = {
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "traceback": traceback_str,
        }
        try:
            scientific_logger._write_log(error_log)
        except:
            pass
    
    return 1